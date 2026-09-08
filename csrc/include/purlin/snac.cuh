//
// Created by Osayamen on 8/16/26.
//

#ifndef PURLIN_SNAC_CUH
#define PURLIN_SNAC_CUH
#include "atom.cuh"
#include "base.cuh"
#include "context.cuh"
#include "epoch.cuh"
#include "partition.cuh"
#include "transfer.cuh"

namespace purlin {
  // Every purlin collective is implemented with SNAC: Stage, Notify, And
  // Consume. First, each rank copies its contribution into local symmetric
  // memory. It then notifies the other ranks, which either gather or reduce the
  // data they need. In the latency regime, staging and notification are fused
  // into a remote packet write whose completion flag travels with the payload.
  //
  // SNAC connects the collective-level decisions (buffer layouts, work
  // geometry, and block roles) to the architecture-specific Atom that performs
  // copies and reductions.
  enum class ConsumeOp {
    gather,
    reduce
  };

  enum class Notify {
    allPeersDirect, // Signal every peer once after staging the complete payload.
    pointerList, // Broadcast each completed chunk through a shared-memory pointer list.
    listEntry, // Signal one peer for each chunk through an entry in the shared list.
    one // Signal one peer for each chunk through a direct pointer.
  };

  template<typename PurlinAtom, typename CollConfig, ConsumeOp op,
    DataLayout inputLayout, DataLayout outputLayout>
  struct SnacTopology {
    static constexpr auto MEMTYPE = PurlinAtom::BaseConfig::MEMTYPE;
    // Zero-staging reads the payload out of the producer's own buffer. There
    // is no staging window to fill in pieces and no capacity to recycle, so
    // the protocol collapses to one notification followed by one bulk read.
    static constexpr bool ZERO_STAGING = residencyOf<CollConfig> == Staging::zero;
    // Whether the protocol publishes its payload in chunks. Every shape
    // decision below, and in SNAC itself, is derived from this rather than
    // read from the collective configuration directly.
    static constexpr bool CHUNKED = CollConfig::COLLECTIVE_TYPE == CollectiveType::chunked;
    static constexpr bool PER_DEST_V = op == ConsumeOp::gather &&
                                       inputLayout == DataLayout::scatteredV &&
                                       outputLayout == DataLayout::transposedV;
    // Select the protocol independently for each stream based on its size.
    // Small streams use packets; larger streams use a fixed staging window for
    // their destination and wrap only if they exceed that window. Both peers
    // know the values used by this decision, so they always choose the same
    // protocol and window layout.
    static constexpr bool PER_STREAM = PER_DEST_V && CHUNKED &&
                                       CollConfig::PER_STREAM_THRESHOLD > 0;
    // Per-stream staging reuses cyclic windows and per-slot counters. It waits
    // for drain notifications only when a stream is large enough to wrap.
    static constexpr bool CYCLIC = CollConfig::STAGING_MODE == StagingMode::cyclic || PER_STREAM;
    static constexpr bool PER_DEST = op == ConsumeOp::gather &&
                                     (inputLayout == DataLayout::scattered || inputLayout == DataLayout::scatteredV);
    // With no staging region, every peer reads the same source buffer, so one
    // broadcast notification replaces the per-destination variants.
    static constexpr Notify NOTIFY = ZERO_STAGING ? Notify::allPeersDirect :
      !CHUNKED ?
        (PER_DEST ? Notify::one : Notify::allPeersDirect) :
        (PER_DEST ? Notify::one :
          (op == ConsumeOp::reduce && inputLayout != DataLayout::packed ?
            Notify::listEntry : Notify::pointerList));
    static_assert(!PER_DEST_V || PER_STREAM || ZERO_STAGING,
      "scatteredV -> transposedV requires the per-stream protocol or zero-staging");
    // The source is the caller's buffer, not purlin's double-buffered staging,
    // so nothing separates one call's readers from the next call's writers.
    // Every block arrives, and the last one holds the kernel open until every
    // peer reports that it has finished reading this rank's buffer.
    static constexpr bool RENDEZVOUS = ZERO_STAGING;
    // Consumers derive a producer's layout locally for every layout pair but
    // one: scatteredV -> transposedV partitions by a world x world matrix, and
    // a consumer holds only its own row and column. There the producer sends
    // the offset with the notification.
    static constexpr bool OFFSET_IN_NOTIFY = ZERO_STAGING && PER_DEST_V;
    // Every rank must advance its epoch by the same amount. Variable-size
    // layouts therefore calculate the next epoch from a globally agreed size,
    // rather than from a rank's local notification count.
    static constexpr bool UNIFORM_ADVANCE =
        inputLayout == DataLayout::packedV || inputLayout == DataLayout::scatteredV;

    enum class Drain {
      none, // Resident staging needs no drain beyond its sense-bit double buffer.
      allRanks, // Every rank consumes the staged region and reports its drain.
      single, // One designated rank consumes the region and reports its drain.
      composedUnicast, // In a composed unicast reduce-then-gather path, the gather-ready
                 // broadcast also drains remote input regions. Gather consumers
                 // drain the local region that carries the reduced result.
      localRegion // With multimem, local gather blocks drain their shard region.
    };
    static constexpr Drain DRAIN = !CYCLIC ? Drain::none :
      (op == ConsumeOp::gather ? (PER_DEST ? Drain::single : Drain::allRanks) :
        (inputLayout == DataLayout::packed ? Drain::allRanks :
          (outputLayout == DataLayout::packed || outputLayout == DataLayout::packedV ? Drain::single :
            (MEMTYPE == MemType::multimem ? Drain::localRegion : Drain::composedUnicast))));
  };

  // The collective resolves this staging geometry before calling SNAC.
  struct StageArgs {
    const cuda::std::byte *const src; // Base of the source buffer.
    const size_t srcOffset = 0; // Offset of this region within the original source.
    cuda::std::byte *const staging; // Local staging base with epoch and region offsets applied.
    const size_t bytes; // Number of bytes staged by this block set.
    const PeerBlock block; // Assigned peer or shard and this block's position in its set.
    uint32_t *const putCounter; // First completion counter used by this block set.
    uint64_t **const signalList = nullptr; // Shared-memory list of notification targets.
    uint64_t *const signal = nullptr; // Direct notification target for one peer.
  };

  template<typename BT = int>
  struct SnacArgs {
    cuda::std::byte *const dst;
    const cuda::std::byte *const src;
    const size_t bytes = 0; // Payload size; variable-size collectives derive it per rank.
    cuda::std::byte *const workspace; // Shared memory, reinterpreted as needed by reductions.
    const size_t *const sizes = nullptr; // Partition sizes for a variable output layout.
    const size_t *const inSizes = nullptr; // Input partitions for scatteredV -> transposedV.
    const BT blocks;
    const int collBlocks; // Blocks covered by epoch bookkeeping; normally equal to blocks.
    const int bIdx = static_cast<int>(blockIdx.x);
  };

  template<DataLayout inputLayout, DataLayout outputLayout, typename BT>
  __device__ __forceinline__
  static size_t vExtent(const SnacArgs<BT> &args, const Context &ctx) {
    if constexpr (inputLayout == DataLayout::packedV ||
                  (inputLayout == DataLayout::scatteredV && outputLayout == DataLayout::packedV)) {
      return args.sizes[ctx.rank];
    }
    return args.bytes;
  }

  template<typename PurlinAtom>
  __device__ __forceinline__
  static void packetPut(cuda::std::byte *__restrict__ const&window,
                        const cuda::std::byte *__restrict__ const&src,
                        const size_t &bytes, const uint64_t &flag,
                        const PeerBlock &block) {
    auto *__restrict__ packets = reinterpret_cast<LRP*>(window);
    const auto *__restrict__ payload = reinterpret_cast<const uint64_t*>(src);
    const auto count = bytes / sizeof(uint64_t);
    const auto stride = static_cast<size_t>(block.blockSetSize) * PurlinAtom::THREADS;
    for (auto i = static_cast<size_t>(block.intraIdx) * PurlinAtom::THREADS + threadIdx.x;
         i < count; i += stride) {
      packets[i].write(payload[i], flag);
    }
  }
  template<typename PurlinAtom>
  __device__ __forceinline__
  static void packetGet(cuda::std::byte *__restrict__ const&dst,
                        const cuda::std::byte *__restrict__ const&window,
                        const size_t &bytes, const uint64_t &flag,
                        const PeerBlock &block) {
    const auto *__restrict__ packets = reinterpret_cast<const LRP*>(window);
    auto *__restrict__ payload = reinterpret_cast<uint64_t*>(dst);
    const auto count = bytes / sizeof(uint64_t);
    const auto stride = static_cast<size_t>(block.blockSetSize) * PurlinAtom::THREADS;
    for (auto i = static_cast<size_t>(block.intraIdx) * PurlinAtom::THREADS + threadIdx.x;
         i < count; i += stride) {
      payload[i] = packets[i].read(flag);
    }
  }

  // Publish this rank's largest variable-size contribution. The exchange is
  // completed later, after all block roles have finished using shared memory.
  template<typename PurlinAtom>
  __device__ __forceinline__
  static void postExtent(const Context &ctx, const uint64_t &senseBit,
                         const uint64_t &nextEpoch, const int &bIdx) {
    if (bIdx == 0) {
      const auto sigPrefix = senseBit * ctx.world;
      const auto payload = static_cast<unsigned long long>(ctx.vState.maxBytes);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
        auto *__restrict__ varSigs = ctx.varLenSignals[i] + (sigPrefix + ctx.rank);
        varSigs->write(payload, nextEpoch);
      }
    }
  }
  template<typename PurlinAtom>
  __device__ __forceinline__
  static size_t awaitExtent(cuda::std::byte *__restrict__ const&workspace,
                            const Context &ctx, const uint64_t &senseBit,
                            const uint64_t &nextEpoch) {
    __syncthreads(); // All block roles must release the shared workspace first.
    auto *__restrict__ maxBytes = reinterpret_cast<unsigned long long*>(workspace);
    if (!threadIdx.x) {
      *maxBytes = 0;
    }
    __syncthreads();
    const auto sigPrefix = senseBit * ctx.world;
    auto *__restrict__ vSigs = ctx.varLenSignals[ctx.rank] + sigPrefix;
    for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
      const auto currentPacket = vSigs[i].wait(nextEpoch);
      atomicMax_block(maxBytes, currentPacket.data);
    }
    __syncthreads();
    const auto globalMax = *maxBytes;
    __syncthreads(); // Finish reading the result before the workspace is reused.
    return static_cast<size_t>(globalMax);
  }

  // SNAC and the Atom share one shared-memory allocation. The Atom's pipeline
  // comes first and SNAC's collective state follows it. These helpers calculate
  // the total launch allocation. The latency protocol does not use the Atom's
  // copy or reduction pipelines.
  template<typename PurlinAtom, Regime regime = Regime::throughput>
  consteval int copySmemBytes() {
    return COLLECTIVE_STATE_BYTES +
      (regime == Regime::throughput ? PurlinAtom::COPY_PIPELINE_SMEM_BYTES : 0);
  }
  template<typename PurlinAtom, Regime regime = Regime::throughput>
  consteval int redSmemBytes() {
    return COLLECTIVE_STATE_BYTES +
      (regime == Regime::throughput ? PurlinAtom::RED_PIPELINE_SMEM_BYTES : 0);
  }
  // A collective that both copies and reduces needs the larger of the two
  // shared-memory allocations because the phases reuse the same workspace.
  template<typename PurlinAtom, Regime regime = Regime::throughput>
  consteval int snacSmemBytes() {
    return cuda::std::max(copySmemBytes<PurlinAtom, regime>(), redSmemBytes<PurlinAtom, regime>());
  }

  template<typename PurlinAtom, typename CollConfig, ConsumeOp op,
    DataLayout inputLayout, DataLayout outputLayout, ReduceOp ro = ReduceOp::add>
  struct SNAC {
    using Topology = SnacTopology<PurlinAtom, CollConfig, op, inputLayout, outputLayout>;
    using Drain = typename Topology::Drain;

    // Stage and notify one producer block. The block copies its assigned region
    // into local staging, either all at once or chunk by chunk through cyclic
    // slots, waiting for consumers when a slot is still in use. It then sends
    // the notification required by the topology. BLOCK_SET fixes the producer
    // set size at compile time when known. ACTIVE_BLOCKS similarly fixes the
    // number of blocks covered by epoch bookkeeping.
    template<int BLOCK_SET = AUTO, int ACTIVE_BLOCKS = AUTO>
    __device__ __forceinline__
    static void stage(const StageArgs &a,
                      cuda::std::byte *__restrict__ const&workspace,
                      const Context &ctx,
                      const uint64_t &epoch,
                      const uint64_t &nextEpoch,
                      const int &bIdx,
                      const int &collBlocks,
                      const int &activeBlocks = 0) {
      constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      const auto blockSetSize = BLOCK_SET == AUTO ? a.block.blockSetSize : BLOCK_SET;
      if constexpr (!Topology::CHUNKED) {
        const auto [bytesPut, putStartOffset] = partition<alignmentBytes>(a.bytes, blockSetSize, a.block.intraIdx);
        const auto *__restrict__ srcP = a.src + (putStartOffset + a.srcOffset);
        auto *__restrict__ dstP = a.staging + putStartOffset;
        PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        if (threadIdx.x / WARP_SIZE == 0) {
          const auto laneId = static_cast<int>(threadIdx.x % WARP_SIZE);
          if (lastArrival(a.putCounter, blockSetSize, laneId)) {
            if constexpr (Topology::NOTIFY == Notify::allPeersDirect) {
              signalAllPeers(ctx.signals, ctx.rank, ctx.world, nextEpoch, laneId);
            } else {
              if (!laneId) {
                auto *__restrict__ signal = ctx.signals[a.block.peer] + ctx.rank;
                signalOne(signal, nextEpoch);
              }
            }
            __syncwarp();
          }
        }
        const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
        markEpoch(ctx, bIdx, nextEpoch);
        if constexpr (ACTIVE_BLOCKS == AUTO) {
          markUnusedEpochs<PurlinAtom>(ctx, collBlocks, activeBlocks, nextEpoch, tid);
        } else {
          markUnusedEpochs<PurlinAtom, ACTIVE_BLOCKS>(ctx, collBlocks, nextEpoch, tid);
        }
      } else {
        constexpr auto CHUNK_SIZE = CollConfig::CHUNK_SIZE;
        constexpr auto cyclic = Topology::CYCLIC;
        const auto chunks = static_cast<int>(a.bytes / CHUNK_SIZE);
        const auto cutoff = CHUNK_SIZE * chunks;
        auto flag = epoch;
        const auto [bytesPut, putStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(blockSetSize, a.block.intraIdx);
        const auto *__restrict__ srcP = a.src + (putStartOffset + a.srcOffset);
        auto *__restrict__ dstBase = a.staging;
        auto *__restrict__ dstP = dstBase + putStartOffset;
        const int laneId = static_cast<int>(threadIdx.x % WARP_SIZE);
        // Before reusing a cyclic slot, wait for the consumer responsible for
        // this region to report that it has finished with the previous chunk.
        const int slots = cyclic ? static_cast<int>(ctx.cyclicSlots) : 0;
        const auto awaitDrain = [&](const uint64_t &target) {
          if constexpr (Topology::DRAIN == Drain::allRanks) {
            waitPeerArrivals<PurlinAtom>(ctx.consumedSignals[ctx.rank], ctx.world, target);
          } else if constexpr (Topology::DRAIN == Drain::single || Topology::DRAIN == Drain::localRegion) {
            if (!threadIdx.x) {
              waitUntilAtLeast(ctx.consumedSignals[ctx.rank] + a.block.peer, target);
            }
          } else if constexpr (Topology::DRAIN == Drain::composedUnicast) {
            if (a.block.peer == ctx.rank) {
              waitPeerArrivals<PurlinAtom>(ctx.consumedSignals[ctx.rank], ctx.world, target);
            } else if (!threadIdx.x) {
              waitUntilAtLeast(ctx.gatherSignals[ctx.rank] + a.block.peer, target);
            }
          }
          __syncthreads();
        };
        // The last producer block to finish a chunk notifies its consumers using
        // the mechanism selected by the topology.
        const auto notifyStaged = [&](const uint64_t &flagV, const int counterIdx) {
          if (threadIdx.x / WARP_SIZE == 0) {
            if (lastArrival(a.putCounter + counterIdx, blockSetSize, laneId)) {
              if constexpr (Topology::NOTIFY == Notify::pointerList) {
                signalPointerList(a.signalList, ctx.world, flagV, laneId);
              } else if constexpr (Topology::NOTIFY == Notify::listEntry) {
                if (!laneId) {
                  signalOne(a.signalList[a.block.peer], flagV);
                }
              } else {
                if (!laneId) {
                  signalOne(a.signal, flagV);
                }
              }
              __syncwarp();
            }
          }
        };
        // Map a chunk to its cyclic slot and, when wrapping, wait before
        // overwriting the previous occupant. Completion counters are maintained
        // per slot rather than per logical chunk.
        const auto enterSlot = [&](const int chunkIdx, const size_t &putOffset) {
          int counterIdx = chunkIdx;
          if constexpr (cyclic) {
            const int slot = chunkIdx % ctx.cyclicSlots;
            counterIdx = slot;
            if (chunkIdx >= slots) {
              awaitDrain(flag + 1 - static_cast<uint64_t>(slots));
            }
            dstP = dstBase + (static_cast<size_t>(slot) * CHUNK_SIZE + putOffset);
          }
          return counterIdx;
        };
        for (int chunk = 0; chunk < chunks; ++chunk) {
          const auto counterIdx = enterSlot(chunk, putStartOffset);
          PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
          __syncthreads();
          flag++;
          notifyStaged(flag, counterIdx);
          dstP += CHUNK_SIZE;
          srcP += CHUNK_SIZE;
        }
        if (a.bytes > cutoff) {
          const auto residue = a.bytes - cutoff;
          const auto [bytesPutLeft, putStartOffsetLeft] = partition<alignmentBytes>(
            residue, blockSetSize, a.block.intraIdx);
          srcP = a.src + ((CHUNK_SIZE * chunks + putStartOffsetLeft) + a.srcOffset);
          dstP = dstBase + (CHUNK_SIZE * chunks + putStartOffsetLeft);
          const auto counterIdx = enterSlot(chunks, putStartOffsetLeft);
          PurlinAtom::copy(dstP, srcP, bytesPutLeft, workspace);
          __syncthreads();
          flag++;
          notifyStaged(flag, counterIdx);
        }
        // In per-stream mode, defer all epoch updates until the global extent
        // exchange has completed.
        if constexpr (!Topology::PER_STREAM) {
          const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
          const auto chunkedEpoch = chunkedNextEpoch(epoch,
            Topology::UNIFORM_ADVANCE ?
            cuda::ceil_div(ctx.vState.maxBytes, CHUNK_SIZE) :
            flag - epoch);
          markEpoch(ctx, bIdx, chunkedEpoch);
          if constexpr (ACTIVE_BLOCKS == AUTO) {
            markUnusedEpochs<PurlinAtom>(ctx, collBlocks, activeBlocks, chunkedEpoch, tid);
          } else {
            markUnusedEpochs<PurlinAtom, ACTIVE_BLOCKS>(ctx, collBlocks, chunkedEpoch, tid);
          }
        }
      }
    }

    struct ScatterMap {
      PeerBlock peerBlock;
      size_t bytesFor; // Size of the partition assigned to this peer.
      size_t offsetFor; // Offset of that partition in the original buffer.
    };
    // Map a block from the three-role grid to its peer and buffer partition.
    template<bool out>
    __device__ __forceinline__
    static ScatterMap mapScatterPeer(const int idx, const int blockCount, const size_t &bytes,
                                     const size_t *__restrict__ const&splits,
                                     cuda::std::byte *__restrict__ const&workspace,
                                     const Context &ctx) {
      auto *sizesP = reinterpret_cast<size_t *>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
      auto *shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
      auto *offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
      if constexpr (inputLayout == DataLayout::scatteredV) {
        for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
          sizesP[i] = splits[i];
          // This rank's own slice uses the local-copy path and sends no staging
          // notification. A weighted block mapped to that slice would therefore
          // wait forever. Rotating the sizes moves the local rank to the last
          // slot; setting that slot to zero makes weighted mapping skip it, just
          // as uniform mapping excludes the local rank through actualWorld.
          shiftedSizes[i] = i == ctx.world - 1 ? 0 : splits[(i + ctx.rank + 1) % ctx.world];
        }
        prefixSum<PurlinAtom::THREADS>(splits, offsets, workspace, ctx.world);
      }
      __syncthreads();
      static_assert(inputLayout != DataLayout::scatteredV ||
                    PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
      const auto skewed = inputLayout == DataLayout::scatteredV &&
        (out ? isSkewed(ctx.vState.totalOutBytes, ctx.vState.maxOutBytes, ctx.world_l)
             : isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l));
      // Staged consumers exclude this rank, because a separate block role copies
      // its slice locally. Zero-staging has no such role, so the rotation covers
      // every peer and this rank's slice is just a local read.
      constexpr bool selfIncluded = Topology::ZERO_STAGING;
      auto peerBlock = skewed
                         ? mapWeightedPeerBlock(idx, blockCount,
                             selfIncluded ? sizesP : shiftedSizes, workspace, ctx.world)
                         : (selfIncluded
                              ? mapPeerBlock(idx, blockCount / ctx.world, ctx.rank, ctx.world)
                              : mapPeerBlock(idx, blockCount / ctx.actualWorld, ctx.rank, ctx.world));
      peerBlock.peer = (skewed && !selfIncluded)
                         ? (peerBlock.peer + ctx.rank + 1) % ctx.world : peerBlock.peer;
      return ScatterMap{
        .peerBlock = peerBlock,
        .bytesFor = inputLayout == DataLayout::scatteredV ? sizesP[peerBlock.peer] : bytes,
        .offsetFor = inputLayout == DataLayout::scatteredV ? offsets[peerBlock.peer] : bytes * peerBlock.peer,
      };
    }

    template<typename BT>
    __device__ __forceinline__
    static void runPerStream(const SnacArgs<BT> &args, const Context &ctx) {
      auto *__restrict__ const dst = args.dst;
      const auto *__restrict__ const src = args.src;
      auto *__restrict__ const workspace = args.workspace;
      const auto *__restrict__ const sizes = args.sizes;
      const auto *__restrict__ const inSizes = args.inSizes;
      const int bIdx = args.bIdx;
      const int collBlocks = args.collBlocks;
      const auto epochState = makeEpochState(ctx, bIdx);
      constexpr auto THRESHOLD = CollConfig::PER_STREAM_THRESHOLD;
      constexpr auto CHUNK_SIZE = CollConfig::CHUNK_SIZE;
      static_assert(CHUNK_SIZE >= MIN_CHUNK_SIZE);
      postExtent<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, bIdx);
      const auto windowBytes = static_cast<size_t>(static_cast<int>(ctx.cyclicSlots)) * CHUNK_SIZE;
      const int stagingBlocks = ctx.stagingBlocks;
      const auto totalPutBlocks = stagingBlocks + CollConfig::LOCAL_PUT_BLOCKS;
      if (bIdx < stagingBlocks) {
        const auto m = mapScatterPeer<false>(bIdx, stagingBlocks, size_t{0}, inSizes, workspace, ctx);
        if (m.bytesFor <= THRESHOLD) {
          auto *__restrict__ window = ctx.stagingLR[m.peerBlock.peer] +
            (epochState.lrStagingPrefix + static_cast<size_t>(ctx.rank) * PACKET_BUFFER_SIZE);
          packetPut<PurlinAtom>(window, src + m.offsetFor, m.bytesFor, epochState.nextEpoch, m.peerBlock);
        }
        else {
          auto *__restrict__ signal = ctx.signals[m.peerBlock.peer] + ctx.rank;
          const auto stagingIntraOffset = windowBytes * static_cast<size_t>(m.peerBlock.peer);
          SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, inputLayout, outputLayout>::
              template stage<>(
                StageArgs{
                  .src = src,
                  .srcOffset = m.offsetFor,
                  .staging = ctx.staging[ctx.rank] + (epochState.trStagingPrefix + stagingIntraOffset),
                  .bytes = m.bytesFor,
                  .block = m.peerBlock,
                  .putCounter = ctx.putCounter + m.peerBlock.peer * MAX_CHUNKS,
                  .signal = signal,
                }, workspace, ctx, epochState.epoch, epochState.nextEpoch, bIdx, collBlocks, stagingBlocks);
        }
      }
      else if (bIdx < totalPutBlocks) {
        // Copy this rank's own slice directly from source to destination; it
        // does not need staging or notification.
        const auto myBytes = sizes[ctx.rank];
        auto *inOffsets = reinterpret_cast<size_t *>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES) +
                          MAX_RANKS_PER_DOMAIN;
        auto *outOffsets = inOffsets + MAX_RANKS_PER_DOMAIN;
        prefixSum<PurlinAtom::THREADS>(inSizes, inOffsets, workspace, ctx.world);
        prefixSum<PurlinAtom::THREADS>(sizes, outOffsets, workspace, ctx.world);
        __syncthreads();
        const auto lBIdx = bIdx - stagingBlocks;
        const auto *__restrict__ srcP = src + inOffsets[ctx.rank];
        auto *__restrict__ dstP = dst + outOffsets[ctx.rank];
        constexpr auto stageBytes = static_cast<size_t>(PurlinAtom::STAGE_BYTES);
        constexpr auto partGranularity = stageBytes * CollConfig::LOCAL_PUT_BLOCKS;
        const auto paddedBytes = alignUp(myBytes, partGranularity);
        const auto [bytesP, startOffset] = partition<CollConfig::LOCAL_PUT_BLOCKS, static_cast<int>(stageBytes)>(
          paddedBytes, lBIdx);
        const auto actualBytes = startOffset >= myBytes
                                   ? size_t{0}
                                   : cuda::std::min(bytesP, myBytes - startOffset);
        PurlinAtom::copy(dstP + startOffset, srcP + startOffset, actualBytes, workspace);
      }
      else {
        const auto cBIdx = bIdx - totalPutBlocks;
        const auto consumerBlocks = static_cast<int>(args.blocks - totalPutBlocks);
        const auto m = mapScatterPeer<true>(cBIdx, consumerBlocks, size_t{0}, sizes, workspace, ctx);
        if (m.bytesFor <= THRESHOLD) {
          auto *__restrict__ window = ctx.stagingLR[ctx.rank] +
            (epochState.lrStagingPrefix + static_cast<size_t>(m.peerBlock.peer) * PACKET_BUFFER_SIZE);
          packetGet<PurlinAtom>(dst + m.offsetFor, window, m.bytesFor, epochState.nextEpoch, m.peerBlock);
        }
        else {
          consume(dst + m.offsetFor, m.bytesFor, workspace, ctx, epochState, bIdx,
                  m.peerBlock, ctx.signals[ctx.rank], epochState.trStagingPrefix);
        }
      }
      // Complete the deferred size exchange, then advance every rank by an
      // epoch increment derived from the same global maximum.
      const auto globalMax = awaitExtent<PurlinAtom>(workspace, ctx, epochState.senseBit, epochState.nextEpoch);
      const auto chunkedEpoch = chunkedNextEpoch(epochState.epoch, cuda::ceil_div(globalMax, CHUNK_SIZE));
      markEpoch(ctx, bIdx, chunkedEpoch);
      const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
      markUnusedEpochs<PurlinAtom>(ctx, collBlocks, collBlocks, chunkedEpoch, tid);
    }

    // Stage narrows to an announcement: the payload is already in the caller's
    // buffer, so block 0 only says so, and its release pairs with a consumer's
    // acquire. scatteredV -> transposedV also sends an address, being the one
    // layout pair whose producer-side offset a consumer cannot derive: it
    // partitions by a world x world matrix and a consumer holds one row and one
    // column, so it knows the length but not where the bytes begin.
    template<typename BT>
    __device__ __forceinline__
    static void publishEntry(const SnacArgs<BT> &args, const Context &ctx,
                             const EpochState &epochState, const int &bIdx) {
      const auto &nextEpoch = epochState.nextEpoch;
      if (bIdx != 0) {
        return;
      }
      if constexpr (Topology::OFFSET_IN_NOTIFY) {
        auto *__restrict__ sendOffsets = reinterpret_cast<size_t *>(
          args.workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        // Where each destination's slice starts in this rank's own buffer.
        prefixSum<PurlinAtom::THREADS>(args.inSizes, sendOffsets, args.workspace, ctx.world);
        for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world;
             peer += PurlinAtom::THREADS) {
          ctx.varLenSignals[peer][epochState.senseBit * ctx.world + ctx.rank].writeRelease(
            static_cast<uint64_t>(sendOffsets[peer]), nextEpoch);
        }
      }
      else if (threadIdx.x / WARP_SIZE == 0) {
        signalAllPeers(ctx.signals, ctx.rank, ctx.world, nextEpoch,
          static_cast<int>(threadIdx.x % WARP_SIZE));
        __syncwarp();
      }
      // This block stands in for the producer role, so it also carries that
      // role's epoch bookkeeping. The entries past the grid have to advance
      // with everything else, or a later launch with a wider grid reads stale
      // values from them. One block covers all of them in a single pass.
      markUnusedEpochs<PurlinAtom, 1>(ctx, args.collBlocks, nextEpoch,
        static_cast<int>(threadIdx.x));
    }

    // Run the throughput protocol for one block. Its grid position determines
    // whether it stages data as a producer or retrieves data as a consumer.
    template<typename BT>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::gather) {
      if constexpr (Topology::ZERO_STAGING) {
        publishEntry(args, ctx, makeEpochState(ctx, args.bIdx), args.bIdx);
      }
      if constexpr (Topology::PER_STREAM) {
        runPerStream(args, ctx);
      }
      else {
        auto *__restrict__ const dst = args.dst;
        const auto *__restrict__ const src = args.src;
        auto *__restrict__ const workspace = args.workspace;
        const auto *__restrict__ const sizes = args.sizes;
        const auto *__restrict__ const inSizes = args.inSizes;
        const auto &blocks = args.blocks;
        const int bIdx = args.bIdx;
        const int collBlocks = args.collBlocks;
        const auto bytes = vExtent<inputLayout, outputLayout>(args, ctx);
        const auto epochState = makeEpochState(ctx, bIdx);
        if constexpr (!Topology::CHUNKED) {
          static_assert(!Topology::CYCLIC);
        } else {
          static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
        }
        constexpr auto chunked = Topology::CHUNKED;
        // The input layout alone determines how the grid is divided into roles.
        if constexpr (inputLayout == DataLayout::packed || inputLayout == DataLayout::packedV) {
          // A packed input uses two roles. Producers stage this rank's entire
          // contribution, while each consumer copies one peer's contribution out
          // of staging.
          if constexpr (!Topology::ZERO_STAGING) {
            if (bIdx < CollConfig::PUT_BLOCKS) {
              uint64_t **signals = nullptr;
              if constexpr (chunked) {
                // Build the shared pointer list used to notify every peer when a
                // chunk is ready.
                signals = reinterpret_cast<uint64_t **>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
                for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
                  signals[i] = ctx.signals[i] + ctx.rank;
                }
                __syncthreads();
              }
              SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, inputLayout, outputLayout>::
                  template stage<CollConfig::PUT_BLOCKS, CollConfig::PUT_BLOCKS>(
                    StageArgs{
                      .src = src,
                      .staging = ctx.staging[ctx.rank] + epochState.trStagingPrefix,
                      .bytes = bytes,
                      .block = PeerBlock{.peer = 0, .intraIdx = bIdx, .blockSetSize = CollConfig::PUT_BLOCKS},
                      .putCounter = ctx.putCounter,
                      .signalList = signals,
                    }, workspace, ctx, epochState.epoch, epochState.nextEpoch, bIdx, collBlocks);
              return;
            }
          }
          const auto cBIdx = bIdx - CollConfig::PUT_BLOCKS;
          const auto consumerBlocks = static_cast<int>(blocks) - CollConfig::PUT_BLOCKS;
          const auto skewed = inputLayout != DataLayout::packed &&
                              isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l);
          static_assert(inputLayout != DataLayout::packedV ||
                        PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
          auto *__restrict__ sizesP = reinterpret_cast<size_t *>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
          auto *__restrict__ offsets = sizesP + MAX_RANKS_PER_DOMAIN;
          if constexpr (inputLayout == DataLayout::packedV) {
            for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
              sizesP[i] = sizes[i];
            }
            prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
            __syncthreads();
          }
          const auto peerBlock = skewed
                                   ? mapWeightedPeerBlock(cBIdx, consumerBlocks, sizesP, workspace, ctx.world)
                                   : mapPeerBlock(cBIdx, consumerBlocks / ctx.world);
          const auto peerBytes = inputLayout == DataLayout::packedV ? sizesP[peerBlock.peer] : bytes;
          const auto offset = inputLayout == DataLayout::packedV ? offsets[peerBlock.peer] : bytes * peerBlock.peer;
          consume(
            dst + offset,
            peerBytes,
            workspace,
            ctx,
            epochState,
            bIdx,
            peerBlock,
            ctx.signals[ctx.rank],
            epochState.trStagingPrefix);
        }
        else {
          // A scattered input uses three roles: producers that stage one
          // destination's data, a direct-copy path for this rank's own shard, and
          // consumers that each retrieve one peer's staged region. Only the
          // fixed-size (scattered -> transposed) path executes here. The
          // (scatteredV -> transposedV) path returns through the per-stream path
          // above
          static_assert(inputLayout == DataLayout::scattered || Topology::PER_STREAM || Topology::ZERO_STAGING);
          const int stagingBlocks = Topology::ZERO_STAGING ? 0 : ctx.stagingBlocks;
          if constexpr (!Topology::ZERO_STAGING) {
            if (bIdx < stagingBlocks) {
              const auto m = mapScatterPeer<false>(bIdx, stagingBlocks, bytes, inSizes, workspace, ctx);
              if constexpr (!chunked) {
                SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, inputLayout, outputLayout>::
                    template stage<>(
                      StageArgs{
                        .src = src,
                        .srcOffset = m.offsetFor,
                        .staging = ctx.staging[ctx.rank] + (epochState.trStagingPrefix + m.offsetFor),
                        .bytes = m.bytesFor,
                        .block = m.peerBlock,
                        .putCounter = ctx.putCounter + m.peerBlock.peer,
                      }, workspace, ctx, epochState.epoch, epochState.nextEpoch, bIdx, collBlocks, stagingBlocks);
              } else {
                auto *__restrict__ signal = ctx.signals[m.peerBlock.peer] + ctx.rank;
                constexpr auto cyclic = Topology::CYCLIC;
                // Cyclic staging preserves offsets in the source buffer but maps
                // each destination to a fixed staging window. The consumer can
                // calculate that window without additional metadata.
                const int slots = cyclic ? static_cast<int>(ctx.cyclicSlots) : 0;
                const auto stagingIntraOffset = cyclic
                                                  ? static_cast<size_t>(slots) * CollConfig::CHUNK_SIZE * static_cast<
                                                      size_t>(m.peerBlock.peer)
                                                  : m.offsetFor;
                SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, inputLayout, outputLayout>::
                    template stage<>(
                      StageArgs{
                        .src = src,
                        .srcOffset = m.offsetFor,
                        .staging = ctx.staging[ctx.rank] + (epochState.trStagingPrefix + stagingIntraOffset),
                        .bytes = m.bytesFor,
                        .block = m.peerBlock,
                        // Chunked publication needs one counter row per peer. The
                        // non-chunked path above uses only one counter per peer.
                        .putCounter = ctx.putCounter + m.peerBlock.peer * MAX_CHUNKS,
                        .signal = signal,
                      }, workspace, ctx, epochState.epoch, epochState.nextEpoch, bIdx, collBlocks, stagingBlocks);
              }
              return;
            }
          }
          const auto totalPutBlocks = stagingBlocks + CollConfig::LOCAL_PUT_BLOCKS;
          if constexpr (!Topology::ZERO_STAGING) {
            if (bIdx < totalPutBlocks) {
              // Copy this rank's own shard directly from source to destination,
              // bypassing staging and notification.
              const auto lBIdx = bIdx - stagingBlocks;
              const auto selfOffset = bytes * ctx.rank;
              superCopy<PurlinAtom, CollConfig::LOCAL_PUT_BLOCKS>(
                dst + selfOffset, src + selfOffset, bytes, workspace, lBIdx);
              if constexpr (chunked) {
                const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
                  static_cast<size_t>(cuda::ceil_div(bytes, CollConfig::CHUNK_SIZE)));
                markEpoch(ctx, bIdx, nextEpoch);
              } else {
                markEpoch(ctx, bIdx, epochState.nextEpoch);
              }
              return;
            }
          }
          // The remaining blocks consume one peer's staged region each.
          const auto cBIdx = bIdx - totalPutBlocks;
          const auto consumerBlocks = static_cast<int>(blocks - totalPutBlocks);
          const auto m = mapScatterPeer<true>(cBIdx, consumerBlocks, bytes, sizes, workspace, ctx);
          consume(
            dst + m.offsetFor,
            m.bytesFor,
            workspace,
            ctx,
            epochState,
            bIdx,
            m.peerBlock,
            ctx.signals[ctx.rank],
            epochState.trStagingPrefix,
            chunked ? bytes : size_t{0}
          );
        }
        if constexpr (Topology::ZERO_STAGING) {
          rendezvous(ctx, collBlocks, epochState.nextEpoch);
        }
      }
    }

    template<typename Element, typename BT>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::reduce) {
      // Zero-staging replaces the producer role with a single announcement:
      // the payload is already in place, so all Stage has left to do is say so.
      // The staging branch below is then empty, because a zero-staged
      // configuration has no producer blocks.
      if constexpr (Topology::ZERO_STAGING) {
        publishEntry(args, ctx, makeEpochState(ctx, args.bIdx), args.bIdx);
      }
      auto *__restrict__ const dst = args.dst;
      const auto *__restrict__ const src = args.src;
      auto *__restrict__ const typedWorkspace = reinterpret_cast<Element *>(args.workspace);
      const auto *__restrict__ const sizes = args.sizes;
      const auto &blocks = args.blocks;
      const int bIdx = args.bIdx;
      const int collBlocks = args.collBlocks;
      const auto bytes = vExtent<inputLayout, outputLayout>(args, ctx);
      const auto epochState = makeEpochState(ctx, bIdx);
      constexpr auto PUT_BLOCKS = CollConfig::PUT_BLOCKS;
      constexpr auto multimem = Topology::MEMTYPE == MemType::multimem;
      static_assert(!multimem || (inputLayout == DataLayout::scattered &&
                                  (outputLayout == DataLayout::scattered || outputLayout == DataLayout::packed)),
                    "the multimem datapath serves shard-partitioned staging reductions only");
      if constexpr (!Topology::CHUNKED) {
        const auto &nextEpoch = epochState.nextEpoch;
        const auto &stagingPrefix = epochState.trStagingPrefix;

        // No producer role when zero-staged; elided rather than left
        // unreachable, which would still cost registers.
        if constexpr (!Topology::ZERO_STAGING) {
          if (bIdx < PUT_BLOCKS) {
            const auto globalBytes = inputLayout == DataLayout::scatteredV ? ctx.vState.totalBytes
            : (inputLayout == DataLayout::scattered ? bytes * ctx.world : bytes);
            auto *__restrict__ workspace = reinterpret_cast<cuda::std::byte *>(typedWorkspace);
            stage<PUT_BLOCKS, PUT_BLOCKS>(
              StageArgs{
                .src = src,
                .staging = ctx.staging[ctx.rank] + stagingPrefix,
                .bytes = globalBytes,
                .block = PeerBlock{.peer = 0, .intraIdx = bIdx, .blockSetSize = PUT_BLOCKS},
                .putCounter = ctx.putCounter,
              }, workspace, ctx, nextEpoch, nextEpoch, bIdx, collBlocks);
            return;
          }
        }
        // The remaining blocks reduce the staged inputs.
        consume(
          dst, bytes, typedWorkspace, ctx, blocks - PUT_BLOCKS, bIdx - PUT_BLOCKS, bIdx,
          nextEpoch, nextEpoch, stagingPrefix);
        // A scattered output feeds a following gather, which already waits on
        // every peer's signal -- and a peer raises that only once its own
        // reduction has finished reading. Completion therefore already implies
        // no peer is reading, so a rendezvous here would be a barrier for
        // nothing.
        if constexpr (Topology::ZERO_STAGING && outputLayout != DataLayout::scattered) {
          rendezvous(ctx, collBlocks, nextEpoch);
        }
      }
      else {
        constexpr auto CHUNK_SIZE = CollConfig::CHUNK_SIZE;
        static_assert(CHUNK_SIZE >= MIN_CHUNK_SIZE);
        constexpr auto cyclic = Topology::CYCLIC;
        const auto &epoch = epochState.epoch;
        const auto &stagingPrefix = epochState.trStagingPrefix;

        // In the chunked throughput path, producer blocks are assigned to
        // individual shards before staging them.
        if (bIdx < PUT_BLOCKS) {
          auto *__restrict__ workspace = reinterpret_cast<cuda::std::byte *>(typedWorkspace);
          static_assert(
            PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES + MAX_RANKS_PER_DOMAIN * sizeof(
              size_t));
          auto *__restrict__ sizesP = reinterpret_cast<size_t *>(workspace + WEIGHTED_PEER_BLOCK_STATE_BYTES);
          auto *__restrict__ signals = reinterpret_cast<uint64_t **>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
          auto *__restrict__ offsets = reinterpret_cast<size_t *>(signals + MAX_RANKS_PER_DOMAIN);
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
            signals[i] = ctx.signals[i] + ctx.rank;
            if constexpr (inputLayout == DataLayout::scatteredV) {
              sizesP[i] = sizes[i];
            }
          }
          __syncthreads();
          // Use uneven mapping when the number of producer blocks is not
          // divisible by the world size. A uniform mapping would assign the
          // trailing blocks to a peer that does not exist.
          const auto uniformPeerBlock = inputLayout == DataLayout::packed ? mapPeerBlock(bIdx, PUT_BLOCKS)
          : mapPeerBlockUneven(bIdx, PUT_BLOCKS, ctx.world);
          const auto peerBlock = inputLayout == DataLayout::scatteredV ?
          (isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l) ?
            mapWeightedPeerBlock(bIdx, PUT_BLOCKS, sizesP, workspace, ctx.world) : uniformPeerBlock) : uniformPeerBlock;
          const auto peer = inputLayout == DataLayout::packed ? 0 : peerBlock.peer;
          const auto intraBIdx = inputLayout == DataLayout::packed ? bIdx : peerBlock.intraIdx;
          const auto blockSetSize = inputLayout == DataLayout::packed ? PUT_BLOCKS : peerBlock.blockSetSize;
          const auto putBytes = inputLayout == DataLayout::scatteredV ? sizesP[peer] : bytes;
          if constexpr (inputLayout == DataLayout::scatteredV) {
            prefixSum<PurlinAtom::THREADS>(sizesP, offsets, workspace, ctx.world);
            __syncthreads();
          }
          const auto intraOffset = inputLayout == DataLayout::scatteredV ? offsets[peer]
          : inputLayout == DataLayout::packed ? 0 : peer * bytes;
          // Cyclic staging preserves source offsets but replaces each staged
          // shard with a fixed-size window.
          const int slots = cyclic ? static_cast<int>(ctx.cyclicSlots) : 0;
          const auto stagingIntraOffset = cyclic ?
            (inputLayout == DataLayout::packed ? size_t{0}
              : static_cast<size_t>(slots) * CHUNK_SIZE * static_cast<size_t>(peer)) : intraOffset;
          stage<(inputLayout == DataLayout::packed ? PUT_BLOCKS : AUTO), PUT_BLOCKS>(
            StageArgs{
              .src = src,
              .srcOffset = intraOffset,
              .staging = ctx.staging[ctx.rank] + (stagingPrefix + stagingIntraOffset),
              .bytes = putBytes,
              .block = PeerBlock{.peer = peer, .intraIdx = intraBIdx, .blockSetSize = blockSetSize},
              .putCounter = inputLayout == DataLayout::packed ? ctx.putCounter : ctx.putCounter + peer * MAX_CHUNKS,
              .signalList = signals,
            }, workspace, ctx, epoch, epoch, bIdx, collBlocks);
          return;
        }

        // The remaining blocks reduce the chunks published by the producers.
        consume(
          dst, bytes, typedWorkspace, ctx, blocks - PUT_BLOCKS, bIdx - PUT_BLOCKS, bIdx,
          epoch, epoch, stagingPrefix);
      }
    }

    // Consume a gather region with one block. Wait until the producer publishes
    // the whole payload or next chunk, copy the assigned region from staging to
    // the destination, and, for cyclic staging, report when each slot can be
    // reused.
    __device__ __forceinline__
    static void consume(cuda::std::byte *__restrict__ const&dst,
                        const size_t &bytes,
                        cuda::std::byte *__restrict__ const&workspace,
                        const Context &ctx,
                        const EpochState &epochState,
                        const int &bIdx,
                        const PeerBlock &peerBlock,
                        uint64_t *__restrict__ const&signalBase,
                        const size_t &stagingPrefix, const size_t &globalMaxBytes = 0,
                        cuda::std::byte *const *__restrict__ const&peerBase = nullptr)
      requires (op == ConsumeOp::gather) {
      // In the multimem (scattered -> scattered) path, the reduction broadcasts
      // every reduced shard to every staging replica. The following gather can
      // therefore read this rank's local replica instead of accessing the
      // producer remotely.
      constexpr auto localGather = Topology::MEMTYPE == MemType::multimem &&
          inputLayout == DataLayout::scattered && outputLayout == DataLayout::scattered;
      size_t sourceOffset = 0; // Packed contributions begin at the buffer base.
      if constexpr (inputLayout == DataLayout::scattered) {
        sourceOffset = outputLayout == DataLayout::scattered
                         ? bytes * peerBlock.peer // The preceding reduction stored shard r in region r.
                         : bytes * ctx.rank; // Read my slice from this peer's destination regions.
      }
      if constexpr (!Topology::CHUNKED) {
        if constexpr (Topology::OFFSET_IN_NOTIFY) {
          // This layout pair cannot derive where its share sits in the
          // producer's buffer, so the producer sent the offset with the flag.
          auto *__restrict__ shared = reinterpret_cast<size_t *>(
            workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES) + 3 * MAX_RANKS_PER_DOMAIN;
          if (!threadIdx.x) {
            *shared = static_cast<size_t>(
              ctx.varLenSignals[ctx.rank][epochState.senseBit * ctx.world + peerBlock.peer]
                .waitUntilAtLeastAcquire(epochState.nextEpoch).data);
          }
          __syncthreads();
          sourceOffset = *shared;
        }
        else {
          if (!threadIdx.x) {
            auto *__restrict__ signal = signalBase + peerBlock.peer;
            waitUntilAtLeast(signal, epochState.nextEpoch);
          }
          __syncthreads();
        }
        // Where the peer's contribution sits: the caller's table when one was
        // supplied, else the producer's own buffer for zero-staging, else the
        // copy the producer placed in its staging region.
        const auto *__restrict__ srcBase = peerBase != nullptr
          ? peerBase[peerBlock.peer] + sourceOffset
          : (Topology::ZERO_STAGING
               ? ctx.peerSrc[peerBlock.peer] + sourceOffset
               : ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + sourceOffset));
        auto *__restrict__ dstP = dst;
        superCopy<PurlinAtom>(dstP, srcBase, bytes, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        markEpoch(ctx, bIdx, epochState.nextEpoch);
      }
      else if constexpr (Topology::CYCLIC) {
        constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
        const int slots = ctx.cyclicSlots;
        const auto windowBytes = static_cast<size_t>(slots) * chunkSize;
        // Cyclic mode uses fixed windows instead of payload-sized staging regions.
        size_t regionOffset = 0; // A packed contribution cycles through one window.
        if constexpr (inputLayout == DataLayout::scattered && outputLayout == DataLayout::scattered) {
          // The preceding reduction placed each shard in its own window.
          regionOffset = windowBytes * static_cast<size_t>(peerBlock.peer);
        } else if constexpr (outputLayout == DataLayout::transposed || outputLayout == DataLayout::transposedV) {
          // Select this rank's destination window in the peer's staging buffer.
          regionOffset = windowBytes * static_cast<size_t>(ctx.rank);
        }
        const auto *__restrict__ srcBase =
            ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + regionOffset);
        auto *__restrict__ dstP = dst;
        const auto chunks = static_cast<int>(bytes / chunkSize);
        const auto chunkCutoff = chunkSize * chunks;
        auto flag = epochState.epoch;
        auto *__restrict__ signal = signalBase + peerBlock.peer;
        // Consumers of remote staging notify the rank that owns that staging.
        // A multimem gather reads a local replica instead, so its producer waits
        // on the local drain entry associated with the shard region.
        auto *__restrict__ consumedSignal = localGather
                                              ? ctx.consumedSignals[ctx.rank] + peerBlock.peer
                                              : ctx.consumedSignals[peerBlock.peer] + ctx.rank;
        auto *__restrict__ consumedCounter = ctx.consumedCounter + peerBlock.peer * MAX_CHUNKS;
        // In per-stream mode, only a stream that wraps waits for drain signals.
        // Do not publish drains for a stream that fits in one window because no
        // producer will wait for them. This predicate is uniform across the
        // block, which keeps the block-wide counter and signal operations safe.
        const bool publishDrains = !Topology::PER_STREAM || bytes > windowBytes;
        for (int i = 0; i < chunks; ++i) {
          flag++;
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          const auto slot = i % ctx.cyclicSlots;
          const auto *__restrict__ srcP = srcBase + static_cast<size_t>(slot) * chunkSize;
          superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
                                                        peerBlock.blockSetSize, peerBlock.intraIdx);
          if (publishDrains) {
            signalConsumed(consumedCounter + slot, consumedSignal, peerBlock.blockSetSize, flag);
          }
          dstP += chunkSize;
        }
        if (bytes > chunkCutoff) {
          flag++;
          const auto residue = bytes - chunkCutoff;
          const auto slot = chunks % ctx.cyclicSlots;
          dstP = dst + chunkCutoff;
          const auto *__restrict__ srcP = srcBase + static_cast<size_t>(slot) * chunkSize;
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          superCopy<PurlinAtom>(dstP, srcP, residue, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
          if (publishDrains) {
            signalConsumed(consumedCounter + slot, consumedSignal, peerBlock.blockSetSize, flag);
          }
        }
        if constexpr (!Topology::PER_STREAM) {
          const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
            outputLayout == DataLayout::transposedV ? cuda::ceil_div(globalMaxBytes, chunkSize) :
            (inputLayout == DataLayout::packedV ?
              cuda::ceil_div(ctx.vState.maxBytes, chunkSize) : flag - epochState.epoch));
          markEpoch(ctx, bIdx, nextEpoch);
        }
      }
      else {
        auto *__restrict__ srcBase =
            ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + sourceOffset);
        auto *__restrict__ srcP = srcBase;
        auto *__restrict__ dstP = dst;
        constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
        const auto chunks = static_cast<int>(bytes / CollConfig::CHUNK_SIZE);
        const auto chunkCutoff = CollConfig::CHUNK_SIZE * chunks;
        auto flag = epochState.epoch;
        auto *__restrict__ signal = signalBase + peerBlock.peer;
        const auto waitChunk = [&](const uint64_t &flagV) {
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flagV);
          }
          __syncthreads();
        };
        for (int i = 0; i < chunks; ++i) {
          flag++;
          waitChunk(flag);
          superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
                                                        peerBlock.blockSetSize, peerBlock.intraIdx);
          srcP += CollConfig::CHUNK_SIZE;
          dstP += CollConfig::CHUNK_SIZE;
        }
        if (bytes > chunkCutoff) {
          flag++;
          const auto residue = bytes - chunkCutoff;
          dstP = dst + (CollConfig::CHUNK_SIZE * chunks);
          srcP = srcBase + (CollConfig::CHUNK_SIZE * chunks);
          waitChunk(flag);
          superCopy<PurlinAtom>(dstP, srcP, residue, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        }
        const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
          outputLayout == DataLayout::transposedV ? cuda::ceil_div(globalMaxBytes, chunkSize) :
          (inputLayout == DataLayout::packedV ?
            cuda::ceil_div(ctx.vState.maxBytes, chunkSize) : flag - epochState.epoch));
        markEpoch(ctx, bIdx, nextEpoch);
      }
    }

    // Consume a reduction region with one block. Wait until every producer has
    // published the whole payload or next chunk, reduce the assigned slice
    // across all staging replicas, and publish the result. A result needed by a
    // later gather is announced through the gather signals. A cyclic reduction
    // that writes directly to its destination instead announces a drain so the
    // producers can reuse their slots.
    template<typename Element, typename RB>
    __device__ __forceinline__
    static void consume(cuda::std::byte *__restrict__ const&dst,
                        const size_t &bytes,
                        Element *__restrict__ const&typedWorkspace,
                        const Context &ctx,
                        const RB &reduceBlocks,
                        const int &reduceBIdx,
                        const int &bIdx,
                        const uint64_t &epoch,
                        const uint64_t &nextEpoch,
                        const size_t &stagingPrefix)
      requires (op == ConsumeOp::reduce) {
      // Multicast results that a following gather will read from staging.
      // Unicast results that are written directly to their final destination.
      constexpr auto reduceResult = outputLayout == DataLayout::scattered ? ReduceResult::multicast :
      ReduceResult::unicast;
      constexpr auto multimem = Topology::MEMTYPE == MemType::multimem;
      constexpr auto cyclic = Topology::CYCLIC;
      constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      if constexpr (!Topology::CHUNKED) {
        const auto [bytesRed, redStartOffset] = partition<alignmentBytes>(bytes, reduceBlocks, reduceBIdx);
        auto *__restrict__ workspace = reinterpret_cast<cuda::std::byte *>(typedWorkspace);
        auto *__restrict__ staging = reinterpret_cast<cuda::std::byte **>(
          workspace + PurlinAtom::RED_PIPELINE_SMEM_BYTES);
        auto *__restrict__ gatherSignals = reinterpret_cast<uint64_t **>(staging + MAX_RANKS_PER_DOMAIN);
        static_assert(
          sizeof(cuda::std::byte **) == sizeof(uint64_t **) && alignof(cuda::std::byte **) == alignof(uint64_t **));
        // Where this rank's shard sits inside a contribution, whatever holds it.
        const auto shardOffset = inputLayout == DataLayout::scatteredV ? ctx.vState.offset :
            inputLayout == DataLayout::scattered ? bytes * ctx.rank : size_t{0};
        for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
          if constexpr (!multimem) {
            // Zero-staging reduces straight out of every peer's own buffer.
            staging[peer] = Topology::ZERO_STAGING
              ? ctx.peerSrc[peer] + (redStartOffset + shardOffset)
              : ctx.staging[peer] + ((stagingPrefix + redStartOffset) + shardOffset);
          }
          gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
        }
        cuda::std::byte *__restrict__ dstP = dst + redStartOffset;
        // The staging replica this rank's result is broadcast into. A following
        // gather reads it from there, so it stays the store target even when the
        // reduction no longer loads from it.
        auto *__restrict__ const mcStagingShard = multimem
          ? ctx.mcStagingTR + (stagingPrefix + redStartOffset + bytes * ctx.rank) : nullptr;
        // Reduce-broadcast: with dst's multicast alias the result lands in every
        // rank's destination directly, shard r at region r, and no gather copies.
        const bool directDst = Topology::ZERO_STAGING && multimem && ctx.mcDst != nullptr;
        auto *__restrict__ const mcResultTarget = directDst
          ? ctx.mcDst + (bytes * ctx.rank + redStartOffset) : mcStagingShard;
        const ReduceTRArgs redArgs{
          .sources = staging,
          // Zero-staging load-reduces out of the caller's buffer through its
          // multicast alias; the staged path reduces out of staging. Either way
          // the result is broadcast into staging, which is what mcResult is.
          .mcSource = multimem
            ? (Topology::ZERO_STAGING ? ctx.mcSrc + (redStartOffset + shardOffset) : mcStagingShard)
            : nullptr,
          .mcResult = mcResultTarget,
          .dst = dstP,
          .bytesRed = bytesRed,
          .world = ctx.world,
        };
        const auto warpId = threadIdx.x / WARP_SIZE;
        const auto laneId = threadIdx.x % WARP_SIZE;
        waitPeerArrivals<PurlinAtom>(ctx.signals[ctx.rank], redArgs.world, nextEpoch);
        __syncthreads();
        PurlinAtom::template reduce<reduceResult, ro>(redArgs, typedWorkspace);
        if constexpr (outputLayout == DataLayout::scattered) {
          __syncthreads();
          // The final reducer block announces that the complete result is
          // ready for the gather phase.
          if (warpId == 0) {
            if (lastArrival(ctx.redCounter, static_cast<int>(reduceBlocks), static_cast<int>(laneId))) {
              signalPointerList(gatherSignals, ctx.world, nextEpoch, laneId);
            }
          }
        }
        markEpoch(ctx, bIdx, nextEpoch);
      }
      else {
        constexpr auto CHUNK_SIZE = CollConfig::CHUNK_SIZE;
        const auto chunks = static_cast<int>(bytes / CHUNK_SIZE);
        const auto cutoff = CHUNK_SIZE * chunks;
        const auto [bytesRed, redStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(reduceBlocks, reduceBIdx);
        auto *__restrict__ workspace = reinterpret_cast<cuda::std::byte *>(typedWorkspace);
        auto *__restrict__ signals = reinterpret_cast<uint64_t **>(workspace + PurlinAtom::RED_PIPELINE_SMEM_BYTES);
        auto *__restrict__ gatherSignals = signals + MAX_RANKS_PER_DOMAIN;
        static_assert(
          sizeof(cuda::std::byte **) == sizeof(uint64_t **) && alignof(cuda::std::byte **) == alignof(uint64_t **));
        auto *__restrict__ staging = reinterpret_cast<cuda::std::byte **>(gatherSignals + MAX_RANKS_PER_DOMAIN);
        auto *__restrict__ consumed = reinterpret_cast<uint64_t **>(staging + MAX_RANKS_PER_DOMAIN);
        const int slots = cyclic ? static_cast<int>(ctx.cyclicSlots) : 0;
        for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
          // Cache the producer, gather, and drain signal addresses used by
          // this block.
          signals[peer] = ctx.signals[ctx.rank] + peer;
          gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
          if constexpr (cyclic && (outputLayout == DataLayout::packed || outputLayout == DataLayout::packedV)) {
            consumed[peer] = ctx.consumedSignals[peer] + ctx.rank;
          }
          if constexpr (!multimem) {
            // Cache the address of this rank's shard in every peer's staging
            // buffer.
            const auto offset = stagingPrefix + redStartOffset;
            const auto regionOffset = cyclic ?
              (inputLayout == DataLayout::packed ? size_t{0} :
                static_cast<size_t>(slots) * CHUNK_SIZE * static_cast<size_t>(ctx.rank)) :
              (inputLayout == DataLayout::scatteredV ? ctx.vState.offset :
                inputLayout == DataLayout::scattered ? bytes * ctx.rank : 0);
            staging[peer] = ctx.staging[peer] + (offset + regionOffset);
          }
        }
        __syncthreads();
        auto flag = epoch;
        const auto mcRegionOffset = cyclic ? static_cast<size_t>(slots) * CHUNK_SIZE * static_cast<size_t>(ctx.rank)
        : bytes * ctx.rank;
        auto *__restrict__ mcPtr = multimem ? ctx.mcStagingTR + (stagingPrefix + redStartOffset + mcRegionOffset)
        : nullptr;
        cuda::std::byte *__restrict__ dstP = dst + redStartOffset;
        const auto warpId = threadIdx.x / WARP_SIZE;
        const auto laneId = threadIdx.x % WARP_SIZE;
        const auto tidS1 = (((warpId + (PurlinAtom::WARPS - 1)) % PurlinAtom::WARPS) * WARP_SIZE) + laneId;
        const auto tidS2 = PurlinAtom::WARPS == 1 ? threadIdx.x
        : (((warpId + (PurlinAtom::WARPS - 2)) % PurlinAtom::WARPS) * WARP_SIZE) + laneId;
        // Publish completion according to the topology. If a gather follows,
        // notify its consumers that the reduced data is ready. If a cyclic
        // reduction writes directly to the destination, notify producers that
        // the slot can be reused.
        const auto publish = [&](const uint64_t &flagV, const int counterIdx) {
          if constexpr (outputLayout == DataLayout::scattered || cyclic) {
            if (warpId == 0) {
              if (lastArrival(ctx.redCounter + counterIdx, static_cast<int>(reduceBlocks),
                              static_cast<int>(laneId))) {
                if constexpr (outputLayout == DataLayout::scattered) {
                  signalPointerList(gatherSignals, ctx.world, flagV, laneId);
                } else {
                  signalPointerList(consumed, ctx.world, flagV, laneId);
                }
              }
            }
          }
        };
        for (int chunk = 0; chunk < chunks; ++chunk) {
          flag++;
          auto counterIdx = chunk;
          if constexpr (cyclic) {
            const int slot = chunk % ctx.cyclicSlots;
            counterIdx = slot;
            if constexpr (outputLayout == DataLayout::scattered) {
              // Store the reduced chunk back in the local shard's cyclic slot
              // so the gather phase can read it.
              dstP = dst + (static_cast<size_t>(slot) * CHUNK_SIZE + redStartOffset);
            }
          }
          const ReduceTRArgs redArgs{
            .sources = staging,
            .mcSource = mcPtr,
            .mcResult = mcPtr,
            .dst = dstP,
            .bytesRed = bytesRed,
            .world = ctx.world,
          };
          waitPointerList<PurlinAtom>(signals, redArgs.world, flag, tidS2);
          __syncthreads();
          PurlinAtom::template reduce<reduceResult, ro>(redArgs, typedWorkspace);
          __syncthreads();
          publish(flag, counterIdx);
          dstP += CHUNK_SIZE;
          if constexpr (multimem) {
            if constexpr (cyclic) {
              const int nextSlot = (chunk + 1) % ctx.cyclicSlots;
              mcPtr += nextSlot == 0 ? -static_cast<ptrdiff_t>((static_cast<size_t>(slots) - 1) * CHUNK_SIZE)
              : static_cast<ptrdiff_t>(CHUNK_SIZE);
            } else {
              mcPtr += CHUNK_SIZE;
            }
          } else {
            for (int i = static_cast<int>(tidS1); i < ctx.world; i += PurlinAtom::THREADS) {
              if constexpr (cyclic) {
                const int nextSlot = (chunk + 1) % ctx.cyclicSlots;
                staging[i] += nextSlot == 0 ? -static_cast<ptrdiff_t>((static_cast<size_t>(slots) - 1) * CHUNK_SIZE)
                : static_cast<ptrdiff_t>(CHUNK_SIZE);
              } else {
                staging[i] += CHUNK_SIZE;
              }
            }
          }
        }
        if (bytes > cutoff) {
          flag++;
          const auto residue = bytes - cutoff;
          const auto [bytesRedLeft, redStartOffsetLeft] = partition<alignmentBytes>(residue, reduceBlocks, reduceBIdx);
          dstP = dst + (CHUNK_SIZE * chunks + redStartOffsetLeft);
          auto counterIdx = chunks;
          if constexpr (cyclic) {
            counterIdx = chunks % ctx.cyclicSlots;
          }
          if constexpr (!multimem) {
            for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
              if constexpr (cyclic) {
                // The completed-chunk loop has already advanced each staging
                // pointer to the slot that will hold the residue.
                staging[i] = (staging[i] - redStartOffset) + redStartOffsetLeft;
              } else {
                auto *__restrict__ stagingBase = staging[i] - (chunks * CHUNK_SIZE + redStartOffset);
                staging[i] = stagingBase + (chunks * CHUNK_SIZE + redStartOffsetLeft);
              }
            }
          }
          if constexpr (cyclic && outputLayout == DataLayout::scattered) {
            dstP = dst + (static_cast<size_t>(counterIdx) * CHUNK_SIZE + redStartOffsetLeft);
          }
          const ReduceTRArgs redArgs{
            .sources = staging,
            .mcSource = multimem ? (mcPtr - redStartOffset) + redStartOffsetLeft : nullptr,
            .mcResult = multimem ? (mcPtr - redStartOffset) + redStartOffsetLeft : nullptr,
            .dst = dstP,
            .bytesRed = bytesRedLeft,
            .world = ctx.world,
          };
          waitPointerList<PurlinAtom>(signals, redArgs.world, flag);
          __syncthreads();
          PurlinAtom::template reduce<reduceResult, ro>(redArgs, typedWorkspace);
          __syncthreads();
          publish(flag, counterIdx);
        }
        const auto chunkedEpoch = chunkedNextEpoch(epoch, inputLayout == DataLayout::scatteredV ?
          cuda::ceil_div(ctx.vState.maxBytes, CHUNK_SIZE) : flag - epoch);
        markEpoch(ctx, bIdx, chunkedEpoch);
      }
    }
  };

  // In the latency regime, SNAC combines staging and notification into one
  // remote packet write whose completion flag travels with the payload.
  // Consuming means waiting for that flag and reading the packet; reductions
  // also combine the received value with the local contribution.
  template<typename PurlinAtom, ConsumeOp op, DataLayout inputLayout, DataLayout outputLayout,
    ReduceOp ro>
  struct SNAC<PurlinAtom, CollectiveConfigLR, op, inputLayout, outputLayout, ro> {
    template<typename BT = int>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::gather) {
      auto *__restrict__ const dst = args.dst;
      const auto *__restrict__ const src = args.src;
      auto *__restrict__ const workspace = args.workspace;
      const auto *__restrict__ const sizes = args.sizes;
      const auto *__restrict__ const inSizes = args.inSizes;
      const auto &blocks = args.blocks;
      const int bIdx = args.bIdx;
      const auto bytes = vExtent<inputLayout, outputLayout>(args, ctx);
      const auto epochState = makeEpochState(ctx, bIdx);
      const auto &nextEpoch = epochState.nextEpoch;
      const auto &senseBit = epochState.senseBit;
      const auto stagingPrefix = (senseBit * ctx.world * purlin::PACKET_BUFFER_SIZE);
      constexpr auto bufferStride = purlin::PACKET_BUFFER_SIZE;
      const auto rankOffset = ctx.rank * purlin::PACKET_BUFFER_SIZE;
      auto *__restrict__ localStaging = ctx.stagingLR[ctx.rank] + stagingPrefix;
      // Divide shared memory into the peer pointers, sizes, and offsets needed
      // by the latency protocol.
      auto *__restrict__ staging = reinterpret_cast<cuda::std::byte **>(workspace);
      auto *__restrict__ offsets = reinterpret_cast<size_t *>(staging + MAX_RANKS_PER_DOMAIN);
      auto *__restrict__ sizesP = offsets + MAX_RANKS_PER_DOMAIN;
      auto *__restrict__ inSizesP = sizesP + MAX_RANKS_PER_DOMAIN;
      auto *__restrict__ inOffsetsP = inSizesP + MAX_RANKS_PER_DOMAIN;
      static_assert(inputLayout != DataLayout::scatteredV ||
                    (copySmemBytes<PurlinAtom, Regime::latency>() >= sizeof(void *) * 5));
      for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
        staging[peer] = ctx.stagingLR[peer] + (stagingPrefix + rankOffset);
        if constexpr (inputLayout == DataLayout::packedV) {
          sizesP[peer] = sizes[peer];
        } else if constexpr (inputLayout == DataLayout::scatteredV) {
          sizesP[peer] = sizes[peer];
          inSizesP[peer] = inSizes[peer];
        }
      }
      if constexpr (inputLayout == DataLayout::packedV) {
        auto *__restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte *>(sizesP + MAX_RANKS_PER_DOMAIN);
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, scanWorkspace, ctx.world);
      } else if constexpr (inputLayout == DataLayout::scatteredV) {
        auto *__restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte *>(inOffsetsP + MAX_RANKS_PER_DOMAIN);
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, scanWorkspace, ctx.world);
        prefixSum<PurlinAtom::THREADS>(inSizes, inOffsetsP, scanWorkspace, ctx.world);
      }
      const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
      __syncthreads();
      // In-place detection is currently supported only for packed input.
      const auto isInPlace = inputLayout == DataLayout::packed ? src == (dst + ctx.rank * bytes) : false;
      const LRArgs gArgs{
        .src = src,
        .staging = staging,
        .localStaging = localStaging,
        .dst = dst,
        .flag = nextEpoch,
        .bufferStride = bufferStride,
        .bytes = bytes,
        .maxBytes = inputLayout == DataLayout::scatteredV ? ctx.vState.maxOutBytes : ctx.vState.maxBytes,
        .inSizes = inSizes,
        .sizes = sizesP,
        .inOffsets = inOffsetsP,
        .offsets = offsets,
        .blocks = blocks,
        .tIdx = static_cast<int>(tid),
        .world = ctx.world,
        .rank = ctx.rank,
        .isInPlace = isInPlace,
      };
      fascia::gather<typename PurlinAtom::BaseConfig, inputLayout>(gArgs);
      __syncthreads();
      markEpoch(ctx, bIdx, nextEpoch);
      markUnusedEpochs<PurlinAtom>(ctx, blocks, blocks, nextEpoch, tid);
    }

    template<typename Element, LRMode mode = LRMode::fullBuffer, typename BT = int>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::reduce) {
      auto *__restrict__ const dst = args.dst;
      const auto *__restrict__ const src = args.src;
      auto *__restrict__ const typedWorkspace = reinterpret_cast<Element *>(args.workspace);
      const auto *__restrict__ const sizes = args.sizes;
      const auto &blocks = args.blocks;
      const int bIdx = args.bIdx;
      const auto bytes = vExtent<inputLayout, outputLayout>(args, ctx);
      const auto epochState = makeEpochState(ctx, bIdx);
      const auto &nextEpoch = epochState.nextEpoch;
      const auto &senseBit = epochState.senseBit;
      const auto stagingPrefix = (senseBit * ctx.world * purlin::PACKET_BUFFER_SIZE);
      constexpr auto bufferStride = purlin::PACKET_BUFFER_SIZE;
      const auto rankOffset = ctx.rank * purlin::PACKET_BUFFER_SIZE;
      auto *__restrict__ base = ctx.stagingLR[ctx.rank];
      auto *__restrict__ localStaging = base + stagingPrefix;
      cuda::std::byte **staging = nullptr;
      size_t *offsets = nullptr;
      size_t *sizesP = nullptr;
      if constexpr (inputLayout == DataLayout::packed) {
        staging = ctx.stagingLR;
      } else {
        staging = reinterpret_cast<cuda::std::byte **>(typedWorkspace);
        offsets = reinterpret_cast<size_t *>(staging + MAX_RANKS_PER_DOMAIN);
        sizesP = offsets + MAX_RANKS_PER_DOMAIN;
        for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
          const auto peerBase = ctx.stagingLR[peer];
          staging[peer] = peerBase + (stagingPrefix + rankOffset);
          if constexpr (inputLayout == DataLayout::scatteredV) {
            sizesP[peer] = sizes[peer];
          }
        }
      }
      if constexpr (inputLayout == DataLayout::scatteredV) {
        // Compute the byte displacement of each variable-size partition.
        auto *__restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte *>(sizesP + MAX_RANKS_PER_DOMAIN);
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, scanWorkspace, ctx.world);
      }
      if constexpr (inputLayout != DataLayout::packed) {
        __syncthreads();
      }
      const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
      const LRArgs redArgs{
        .src = src,
        .staging = staging,
        .localStaging = localStaging,
        .dst = dst,
        .mcStaging = PurlinAtom::BaseConfig::MEMTYPE == MemType::multimem && inputLayout == DataLayout::packed
                       ? ctx.mcStagingLR + (stagingPrefix + rankOffset)
                       : nullptr,
        .flag = nextEpoch,
        .bufferStride = bufferStride,
        .stagingOffset = inputLayout == DataLayout::packed ? stagingPrefix + rankOffset : 0,
        .bytes = bytes,
        .maxBytes = ctx.vState.maxBytes,
        .sizes = sizesP,
        .offsets = offsets,
        .blocks = blocks,
        .tIdx = static_cast<int>(tid),
        .world = ctx.world,
        .rank = ctx.rank,
        .bIdx = bIdx,
      };
      static_assert(mode == LRMode::fullBuffer || inputLayout == DataLayout::packed);
      PurlinAtom::template reduce<inputLayout, mode, ro>(redArgs, typedWorkspace);
      __syncthreads();
      markEpoch(ctx, bIdx, nextEpoch);
      markUnusedEpochs<PurlinAtom>(ctx, blocks, blocks, nextEpoch, tid);
    }
  };

  // Compose (scattered -> packed) with (packed -> scattered) to produce a fused
  // (scattered -> scattered) operation. The first SNAC reduces each scattered
  // shard and notifies the second SNAC's gather consumers, which distribute the
  // results into the scattered destination. The two phases use different
  // sections of the grid, but every block derives the same epoch state from its
  // own bookkeeping entry.
  //
  // The intermediate lives in staging by default. When the destination is
  // peer-visible it lives there instead, and the composition touches no staging
  // at all. This necessitates an exit barrier, since a destination has no
  // sense-bit double buffer to separate one call's readers from the next call's
  // writes.
  template<typename PurlinAtom, typename CollConfig, ReduceOp ro = ReduceOp::add>
  struct ReduceGatherSNAC {
    // A gather half that reads staging is reading purlin's own buffer, so it
    // stays staged even when the reduce half does not.
    using StagedCollConfig = CollectiveConfig<
      CollConfig::COLLECTIVE_TYPE, CollConfig::PUT_BLOCKS, CollConfig::GATHER_BLOCKS,
      CollConfig::CHUNK_SIZE, CollConfig::LOCAL_PUT_BLOCKS, CollConfig::LATENCY_THRESHOLD,
      CollConfig::STAGING_MODE, CollConfig::PER_STREAM_THRESHOLD, Staging::staged>;
    template<typename Element, typename BT>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx) {
      const auto &bIdx = args.bIdx;
      const auto epochState = makeEpochState(ctx, bIdx);
      const auto stagingPrefix = epochState.trStagingPrefix;
      const auto localBytes = args.bytes / ctx.world_l;
      const auto reduceHalfBlocks = args.blocks - CollConfig::GATHER_BLOCKS;
      // Multimem keeps its intermediate in staging whatever the caller supplies:
      // a multicast reduction writes only through its multicast alias, and the
      // only alias purlin holds is staging's. Honouring peerDst here would leave
      // the destination unwritten and the gather half reading nothing.
      const bool intermediateInDst = residencyOf<CollConfig> == Staging::zero &&
        PurlinAtom::BaseConfig::MEMTYPE == MemType::unicast && ctx.peerDst != nullptr;
      // Reduce-broadcast: the multimem reduce stored shard r into every dst
      // through dst's multicast alias, so the gather half only has to wait for
      // each peer's reducers before the kernel may complete.
      const bool directDst = residencyOf<CollConfig> == Staging::zero &&
        PurlinAtom::BaseConfig::MEMTYPE == MemType::multimem && ctx.mcDst != nullptr;
      // In cyclic mode each shard uses a fixed staging window rather than a
      // region sized to localBytes. A destination is always laid out by shard.
      constexpr auto cyclic = CollConfig::STAGING_MODE == StagingMode::cyclic;
      const auto shardStagingOffset = cyclic ?
        static_cast<size_t>(static_cast<int>(ctx.cyclicSlots)) * CollConfig::CHUNK_SIZE *
          static_cast<size_t>(ctx.rank) :
        localBytes * ctx.rank;
      if (bIdx < reduceHalfBlocks) {
        auto *__restrict__ sDst = (intermediateInDst || directDst) ? args.dst + (localBytes * ctx.rank)
          : ctx.staging[ctx.rank] + (stagingPrefix + shardStagingOffset);
        SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scattered, DataLayout::scattered,
          ro>::template run<Element>(
              SnacArgs<decltype(reduceHalfBlocks)>{
                .dst = sDst,
                .src = args.src,
                .bytes = localBytes,
                .workspace = args.workspace,
                .blocks = reduceHalfBlocks,
                .collBlocks = args.collBlocks,
                .bIdx = bIdx,
              }, ctx);
      }
      else {
        const auto gBIdx = bIdx - reduceHalfBlocks;
        // Use uneven mapping when the gather-block count is not divisible by the
        // world size. A uniform mapping would assign trailing blocks to a peer
        // that does not exist.
        const auto peerBlock = mapPeerBlockUneven(static_cast<int>(gBIdx), CollConfig::GATHER_BLOCKS, ctx.world);
        // The reduce half wrote this rank's shard straight into the destination
        // when the intermediate lives there, so gathering it would copy dst onto
        // itself. Mark the epoch and leave the shard alone.
        if (directDst) {
          if (!threadIdx.x) {
            waitUntilAtLeast(ctx.gatherSignals[ctx.rank] + peerBlock.peer, epochState.nextEpoch);
          }
          __syncthreads();
          markEpoch(ctx, bIdx, epochState.nextEpoch);
        }
        else if (intermediateInDst && peerBlock.peer == ctx.rank) {
          markEpoch(ctx, bIdx, epochState.nextEpoch);
        }
        else {
          SNAC<PurlinAtom, StagedCollConfig, ConsumeOp::gather, DataLayout::scattered,
            DataLayout::scattered>::consume(
            args.dst + (localBytes * peerBlock.peer),
            localBytes,
            args.workspace,
            ctx,
            epochState,
            bIdx,
            peerBlock,
            ctx.gatherSignals[ctx.rank],
            stagingPrefix,
            size_t{0},
            intermediateInDst ? ctx.peerDst : nullptr
          );
        }
      }
      // Peers read this rank's destination only when the intermediate lives
      // there, and nothing else separates that from the next call's writes.
      // Staging's own double buffer covers the other case.
      if (intermediateInDst) {
        rendezvous(ctx, args.collBlocks, epochState.nextEpoch);
      }
    }
  };
}
#endif //PURLIN_SNAC_CUH
