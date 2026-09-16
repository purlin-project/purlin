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

  // A SNAC can be composed with another; Compose, below, joins them. The head
  // of a composition consumes into the staging region its tail reads and
  // announces each completed result as the tail's staged data. The tail has
  // no stage of its own and reads the head's results from staging. A SNAC
  // with no seam is a complete collective by itself. Compose attaches the
  // seams to the SNACs it is given; the parameter is never written by hand.
  enum class Seam {
    none,
    head,
    tail
  };

  enum class Notify {
    allPeersDirect, // Signal every peer once after staging the complete payload.
    pointerList, // Broadcast each completed chunk through a shared-memory pointer list.
    listEntry, // Signal one peer for each chunk through an entry in the shared list.
    one // Signal one peer for each chunk through a direct pointer.
  };

  template<typename PurlinAtom, typename CollConfig, ConsumeOp op,
    DataLayout inputLayout, DataLayout outputLayout, Seam seam>
  struct SnacTopology {
    static constexpr auto MEMTYPE = PurlinAtom::BaseConfig::MEMTYPE;
    static constexpr bool PER_DEST_V = op == ConsumeOp::gather &&
                                       inputLayout == DataLayout::scatteredV &&
                                       outputLayout == DataLayout::transposedV;
    // Select the protocol independently for each stream based on its size.
    // Small streams use packets; larger streams use a fixed staging window for
    // their destination and wrap only if they exceed that window. Both peers
    // know the values used by this decision, so they always choose the same
    // protocol and window layout.
    static constexpr bool PER_STREAM = PER_DEST_V &&
                                       CollConfig::COLLECTIVE_TYPE == CollectiveType::chunked &&
                                       CollConfig::PER_STREAM_THRESHOLD > 0;
    // Per-stream staging reuses cyclic windows and per-slot counters. It waits
    // for drain notifications only when a stream is large enough to wrap.
    static constexpr bool CYCLIC = CollConfig::STAGING_MODE == StagingMode::cyclic || PER_STREAM;
    static constexpr bool PER_DEST = op == ConsumeOp::gather &&
                                     (inputLayout == DataLayout::scattered || inputLayout == DataLayout::scatteredV);
    static constexpr Notify NOTIFY =
      CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked ?
        (PER_DEST ? Notify::one : Notify::allPeersDirect) :
        (PER_DEST ? Notify::one :
          (op == ConsumeOp::reduce && inputLayout != DataLayout::packed ?
            Notify::listEntry : Notify::pointerList));
    static_assert(!PER_DEST_V || PER_STREAM,
      "scatteredV -> transposedV requires the per-stream protocol");
    // Every rank must advance its epoch by the same amount. Variable-size
    // layouts therefore calculate the next epoch from a globally agreed size,
    // rather than from a rank's local notification count.
    static constexpr bool UNIFORM_ADVANCE =
        inputLayout == DataLayout::packedV || inputLayout == DataLayout::scatteredV;
    // Variable-size invocations announce entry to every peer before any work
    // can block and wait for every peer's announcement before returning. The
    // per-stream path folds that exchange into its extent handshake.
    static constexpr bool VARLEN_ARRIVAL = UNIFORM_ADVANCE && !PER_STREAM;

    enum class Drain {
      none, // Resident staging needs no drain beyond its sense-bit double buffer.
      allRanks, // Every rank consumes the staged region and reports its drain.
      single, // One designated rank consumes the region and reports its drain.
      composedUnicast, // At the head of a unicast composition, the gather-ready
                 // broadcast also drains remote input regions. Gather consumers
                 // drain the local region that carries the reduced result.
      localRegion // At the head of a multimem composition, local gather blocks
                 // drain their shard region.
    };
    static constexpr Drain DRAIN = !CYCLIC ? Drain::none :
      (op == ConsumeOp::gather ? (PER_DEST ? Drain::single : Drain::allRanks) :
        (inputLayout == DataLayout::packed ? Drain::allRanks :
          (seam == Seam::head ?
            (MEMTYPE == MemType::multimem ? Drain::localRegion : Drain::composedUnicast) :
            Drain::single)));
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

  template<typename PurlinAtom>
  __device__ __forceinline__
  static void postVarlenSignal(const Context &ctx, const uint64_t &senseBit,
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
  static void awaitVarlenSignal(const Context &ctx, const uint64_t &senseBit,
                                const uint64_t &nextEpoch, const int &bIdx) {
    if (bIdx == 0) {
      const auto *varSigs = ctx.varLenSignals[ctx.rank] + senseBit * ctx.world;
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
        (void)varSigs[i].wait(nextEpoch);
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
  // The latency protocol carves its per-peer arrays out of the collective
  // state region and places prefix-sum scratch after them. The widest layout
  // is the scatteredV gather's: staging pointers, output offsets, output
  // sizes, input sizes, and input offsets. The reduce paths use a prefix of
  // the same layout, so one bound covers every latency path.
  template<typename PurlinAtom>
  consteval int lrStateBytes() {
    constexpr int peerArrays = 5;
    return static_cast<int>(peerArrays * MAX_RANKS_PER_DOMAIN * sizeof(size_t) +
      prefixSumScratchBytes<PurlinAtom::THREADS>());
  }

  template<typename PurlinAtom, typename CollConfig, ConsumeOp op,
    DataLayout inputLayout, DataLayout outputLayout, ReduceOp ro = ReduceOp::add,
    Seam seam = Seam::none>
  struct SNAC {
    using Topology = SnacTopology<PurlinAtom, CollConfig, op, inputLayout, outputLayout, seam>;
    using Drain = typename Topology::Drain;
    // Traits Compose reads to check that two SNACs fit together.
    using AtomType = PurlinAtom;
    using CollType = CollConfig;
    static constexpr ConsumeOp OP = op;
    static constexpr DataLayout INPUT = inputLayout;
    static constexpr DataLayout OUTPUT = outputLayout;
    static constexpr ReduceOp RO = ro;
    static constexpr Seam SEAM = seam;
    // Which seams this SNAC can play. A head's consume must write into the
    // seam and publish the tail's staged signal; a tail's consume must read
    // the seam. Only the reduce consume implements the head hooks and only the
    // gather consume implements the tail reads. Flip these when a hook is
    // added to another op.
    static constexpr bool CAN_HEAD = op == ConsumeOp::reduce;
    static constexpr bool CAN_TAIL = op == ConsumeOp::gather;
    static_assert(seam == Seam::none || (seam == Seam::head ? CAN_HEAD : CAN_TAIL),
      "this SNAC's consume has no hooks for the requested seam");

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
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
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
      auto peerBlock = skewed
                         ? mapWeightedPeerBlock(idx, blockCount, shiftedSizes, workspace, ctx.world)
                         : mapPeerBlock(idx, blockCount / ctx.actualWorld, ctx.rank, ctx.world);
      peerBlock.peer = skewed ? (peerBlock.peer + ctx.rank + 1) % ctx.world : peerBlock.peer;
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
      postVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, bIdx);
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

    // Run the throughput protocol for one block. Variable-size layouts
    // bracket the invocation with the arrival exchange; the per-stream path
    // carries its own.
    template<typename BT>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::gather) {
      if constexpr (Topology::PER_STREAM) {
        runPerStream(args, ctx);
        return;
      }
      const auto epochState = makeEpochState(ctx, args.bIdx);
      if constexpr (Topology::VARLEN_ARRIVAL) {
        postVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, args.bIdx);
      }
      runStaged(args, ctx);
      if constexpr (Topology::VARLEN_ARRIVAL) {
        awaitVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, args.bIdx);
      }
    }

    // Stage or consume one block. Its grid position determines whether it
    // stages data as a producer or retrieves data as a consumer.
    template<typename BT>
    __device__ __forceinline__
    static void runStaged(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::gather) {
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
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        static_assert(CollConfig::STAGING_MODE == StagingMode::resident);
      } else {
        static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
      }
      constexpr auto chunked = CollConfig::COLLECTIVE_TYPE == CollectiveType::chunked;
      // The input layout alone determines how the grid is divided into roles.
      if constexpr (inputLayout == DataLayout::packed || inputLayout == DataLayout::packedV) {
        // A packed input uses two roles. Producers stage this rank's entire
        // contribution, while each consumer copies one peer's contribution out
        // of staging.
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
        static_assert(inputLayout == DataLayout::scattered || Topology::PER_STREAM);
        const int stagingBlocks = ctx.stagingBlocks;
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
            constexpr auto cyclic = CollConfig::STAGING_MODE == StagingMode::cyclic;
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
        const auto totalPutBlocks = stagingBlocks + CollConfig::LOCAL_PUT_BLOCKS;
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
    }

    template<typename Element, typename BT>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::reduce) {
      const auto epochState = makeEpochState(ctx, args.bIdx);
      if constexpr (Topology::VARLEN_ARRIVAL) {
        postVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, args.bIdx);
      }
      runStaged<Element>(args, ctx);
      if constexpr (Topology::VARLEN_ARRIVAL) {
        awaitVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, args.bIdx);
      }
    }

    template<typename Element, typename BT>
    __device__ __forceinline__
    static void runStaged(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::reduce) {
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
      static_assert(!multimem || (inputLayout == DataLayout::scattered && outputLayout == DataLayout::packed),
                    "the multimem datapath serves shard-partitioned staging reductions only");
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        const auto &nextEpoch = epochState.nextEpoch;
        const auto &stagingPrefix = epochState.trStagingPrefix;

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
        // The remaining blocks reduce the staged inputs.
        consume(
          dst, bytes, typedWorkspace, ctx, blocks - PUT_BLOCKS, bIdx - PUT_BLOCKS, bIdx,
          nextEpoch, nextEpoch, stagingPrefix);
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
                        const size_t &stagingPrefix, const size_t &globalMaxBytes = 0)
      requires (op == ConsumeOp::gather) {
      // At the tail of a multimem composition, the head's reduction broadcast
      // every reduced shard to every staging replica. The gather can therefore
      // read this rank's local replica instead of accessing the producer
      // remotely.
      constexpr auto localGather = Topology::MEMTYPE == MemType::multimem && seam == Seam::tail;
      size_t sourceOffset = 0; // Packed contributions begin at the staging base.
      if constexpr (seam == Seam::tail) {
        sourceOffset = bytes * peerBlock.peer; // The head stored shard r in region r.
      } else if constexpr (inputLayout == DataLayout::scattered) {
        sourceOffset = bytes * ctx.rank; // Read my slice from this peer's destination regions.
      }
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        if (!threadIdx.x) {
          auto *__restrict__ signal = signalBase + peerBlock.peer;
          waitUntilAtLeast(signal, epochState.nextEpoch);
        }
        __syncthreads();
        const auto *__restrict__ srcBase =
            ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + sourceOffset);
        const auto *__restrict__ srcP = srcBase;
        auto *__restrict__ dstP = dst;
        superCopy<PurlinAtom>(dstP, srcP, bytes, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        markEpoch(ctx, bIdx, epochState.nextEpoch);
      }
      else if constexpr (Topology::CYCLIC) {
        constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
        const int slots = ctx.cyclicSlots;
        const auto windowBytes = static_cast<size_t>(slots) * chunkSize;
        // Cyclic mode uses fixed windows instead of payload-sized staging regions.
        size_t regionOffset = 0; // A packed contribution cycles through one window.
        if constexpr (seam == Seam::tail) {
          // The head placed each shard in its own window.
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
      constexpr auto multimem = Topology::MEMTYPE == MemType::multimem;
      // At the head of a composition, the tail reads the result from staging.
      // The multimem datapath stores it through the multicast mapping so every
      // replica receives it. A unicast datapath stores it into this rank's
      // shard, which peers read remotely, so it is asked for a unicast store.
      constexpr auto reduceResult = multimem && seam == Seam::head
                                      ? ReduceResult::multicast : ReduceResult::unicast;
      constexpr auto cyclic = Topology::CYCLIC;
      constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        const auto [bytesRed, redStartOffset] = partition<alignmentBytes>(bytes, reduceBlocks, reduceBIdx);
        auto *__restrict__ workspace = reinterpret_cast<cuda::std::byte *>(typedWorkspace);
        auto *__restrict__ staging = reinterpret_cast<cuda::std::byte **>(
          workspace + PurlinAtom::RED_PIPELINE_SMEM_BYTES);
        auto *__restrict__ gatherSignals = reinterpret_cast<uint64_t **>(staging + MAX_RANKS_PER_DOMAIN);
        static_assert(
          sizeof(cuda::std::byte **) == sizeof(uint64_t **) && alignof(cuda::std::byte **) == alignof(uint64_t **));
        for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
          if constexpr (!multimem) {
            const auto offset = stagingPrefix + redStartOffset;
            staging[peer] = ctx.staging[peer] + (offset + (inputLayout == DataLayout::scatteredV ? ctx.vState.offset :
                inputLayout == DataLayout::scattered ? bytes * ctx.rank : 0));
          }
          gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
        }
        cuda::std::byte *__restrict__ dstP = dst + redStartOffset;
        const ReduceTRArgs redArgs{
          .sources = staging,
          .mcSource = multimem ? ctx.mcStagingTR + (stagingPrefix + redStartOffset + bytes * ctx.rank) : nullptr,
          .dst = dstP,
          .bytesRed = bytesRed,
          .world = ctx.world,
        };
        const auto warpId = threadIdx.x / WARP_SIZE;
        const auto laneId = threadIdx.x % WARP_SIZE;
        waitPeerArrivals<PurlinAtom>(ctx.signals[ctx.rank], redArgs.world, nextEpoch);
        __syncthreads();
        PurlinAtom::template reduce<reduceResult, ro>(redArgs, typedWorkspace);
        if constexpr (seam == Seam::head) {
          __syncthreads();
          // The final reducer block announces that the complete result is
          // ready for the tail.
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
          if constexpr (cyclic && seam != Seam::head) {
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
          if constexpr (seam == Seam::head || cyclic) {
            if (warpId == 0) {
              if (lastArrival(ctx.redCounter + counterIdx, static_cast<int>(reduceBlocks),
                              static_cast<int>(laneId))) {
                if constexpr (seam == Seam::head) {
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
            if constexpr (seam == Seam::head) {
              // Store the reduced chunk back in the local shard's cyclic slot
              // so the tail can read it.
              dstP = dst + (static_cast<size_t>(slot) * CHUNK_SIZE + redStartOffset);
            }
          }
          const ReduceTRArgs redArgs{
            .sources = staging,
            .mcSource = mcPtr,
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
          if constexpr (cyclic && seam == Seam::head) {
            dstP = dst + (static_cast<size_t>(counterIdx) * CHUNK_SIZE + redStartOffsetLeft);
          }
          const ReduceTRArgs redArgs{
            .sources = staging,
            .mcSource = multimem ? (mcPtr - redStartOffset) + redStartOffsetLeft : nullptr,
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

  // Arguments shared by the latency protocol's phases. The type lives at
  // namespace scope so the head and tail of a composed latency collective
  // share one instance.
  struct LRArgs {
    const cuda::std::byte* const src;
    cuda::std::byte** const staging;
    cuda::std::byte* const localStaging; // This rank's entry in staging.
    cuda::std::byte* const dst;
    // Multicast alias for this rank's packet region, with the staging prefix
    // and rank offset already applied. This is null when NVLS is unavailable.
    cuda::std::byte* const mcStaging = nullptr;
    const uint64_t flag;
    const size_t bufferStride;
    const size_t stagingOffset = 0;
    const size_t bytes;
    const size_t maxBytes;
    const size_t* const inSizes = nullptr;
    const size_t* const sizes = nullptr;
    const size_t* const inOffsets = nullptr;
    const size_t* const offsets = nullptr;
    const int blocks;
    const int tIdx;
    const cuda::fast_mod_div<int, true> world;
    const int rank;
    const int bIdx;
    const int isInPlace;
  };

  // In the latency regime, staging and notification are one operation. A
  // 16-byte packet carries its payload and its completion flag and is written
  // with a single store into the peer's packet buffer. stage() writes this
  // rank's packets into every peer; consume() waits on the flags that arrive
  // in this rank's own buffer and reads or reduces the payloads. The Atom is
  // not involved: the packet path has no pipeline, and the reduction functor
  // is lowered from the Atom's arch.
  template<typename PurlinAtom, ConsumeOp op, DataLayout inputLayout, DataLayout outputLayout, ReduceOp ro,
    Seam seam>
  struct SNAC<PurlinAtom, CollectiveConfigLR, op, inputLayout, outputLayout, ro, seam> {
    using Config = typename PurlinAtom::BaseConfig; // THREADS, WORLD_UNROLL, MEMTYPE
    using RedOp = typename LoweredReduceOp<ro, PurlinAtom::NARCH>::type;
    using AtomType = PurlinAtom;
    using CollType = CollectiveConfigLR;
    static constexpr ConsumeOp OP = op;
    static constexpr DataLayout INPUT = inputLayout;
    static constexpr DataLayout OUTPUT = outputLayout;
    static constexpr ReduceOp RO = ro;
    static constexpr Seam SEAM = seam;
    // Which seams this SNAC can play. A head's consume must write into the
    // seam and publish the tail's staged signal; a tail's consume must read
    // the seam. stagePartitioned and reducePartitioned are the reduce's head
    // hooks; gatherPartitioned is the gather's tail read. Flip these when a hook is
    // added to another op.
    static constexpr bool CAN_HEAD = op == ConsumeOp::reduce;
    static constexpr bool CAN_TAIL = op == ConsumeOp::gather;
    static_assert(seam == Seam::none || (seam == Seam::head ? CAN_HEAD : CAN_TAIL),
      "this SNAC's consume has no hooks for the requested seam");
    // A composed latency reduction keeps its input packets in the first half
    // of each rank's window and its result packets in the second half. A
    // remote writer cannot know when a reader has finished polling the input
    // half, so the result cannot alias it.
    static constexpr size_t RESULT_OFFSET = PACKET_BUFFER_SIZE / 2;
    // The latency path has no per-stream mode: every variable-size layout
    // brackets its invocation with the arrival exchange.
    static constexpr bool VARLEN_ARRIVAL =
        inputLayout == DataLayout::packedV || inputLayout == DataLayout::scatteredV;
    static_assert(COLLECTIVE_STATE_BYTES >= lrStateBytes<PurlinAtom>(),
      "the collective state region must hold the latency protocol's per-peer arrays and scan scratch");

    // Tiny full-buffer reductions expose too few elements to occupy the grid,
    // so their work is distributed by peer instead of by element. stage and
    // consume must agree on this schedule: the in-place barrier at the end of
    // stageFullBuffer covers only the peer-striped path.
    __device__ __forceinline__
    static bool isPeerStriped(const LRArgs& a) {
      return a.world > 4 && a.bytes <= 16UL * 1024UL;
    }

    template<typename BT = int>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::gather) {
      const auto epochState = makeEpochState(ctx, args.bIdx);
      if constexpr (VARLEN_ARRIVAL) {
        postVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, args.bIdx);
      }
      runPackets(args, ctx);
      if constexpr (VARLEN_ARRIVAL) {
        awaitVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, args.bIdx);
      }
    }

    template<typename BT>
    __device__ __forceinline__
    static void runPackets(const SnacArgs<BT> &args, const Context &ctx)
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
        .inSizes = inSizesP,
        .sizes = sizesP,
        .inOffsets = inOffsetsP,
        .offsets = offsets,
        .blocks = blocks,
        .tIdx = static_cast<int>(tid),
        .world = ctx.world,
        .rank = ctx.rank,
        .isInPlace = isInPlace,
      };
      stage(gArgs);
      consume(gArgs);
      __syncthreads();
      markEpoch(ctx, bIdx, nextEpoch);
      markUnusedEpochs<PurlinAtom>(ctx, blocks, blocks, nextEpoch, tid);
    }

    template<typename Element, typename BT = int>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::reduce) {
      const auto epochState = makeEpochState(ctx, args.bIdx);
      if constexpr (VARLEN_ARRIVAL) {
        postVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, args.bIdx);
      }
      runPackets<Element>(args, ctx);
      if constexpr (VARLEN_ARRIVAL) {
        awaitVarlenSignal<PurlinAtom>(ctx, epochState.senseBit, epochState.nextEpoch, args.bIdx);
      }
    }

    // Build the arguments for a packed contribution: one window per peer,
    // addressed through the context's pointer table without a shared-memory
    // copy. The direct reduction and the head of a composed reduction share
    // this geometry.
    template<typename BT>
    __device__ __forceinline__
    static LRArgs packedArgs(const SnacArgs<BT> &args, const Context &ctx) {
      const int bIdx = args.bIdx;
      const auto epochState = makeEpochState(ctx, bIdx);
      const auto stagingPrefix = (epochState.senseBit * ctx.world * purlin::PACKET_BUFFER_SIZE);
      const auto rankOffset = ctx.rank * purlin::PACKET_BUFFER_SIZE;
      return LRArgs{
        .src = args.src,
        .staging = ctx.stagingLR,
        .localStaging = ctx.stagingLR[ctx.rank] + stagingPrefix,
        .dst = args.dst,
        .mcStaging = Config::MEMTYPE == MemType::multimem
                       ? ctx.mcStagingLR + (stagingPrefix + rankOffset)
                       : nullptr,
        .flag = epochState.nextEpoch,
        .bufferStride = purlin::PACKET_BUFFER_SIZE,
        .stagingOffset = stagingPrefix + rankOffset,
        .bytes = args.bytes,
        .maxBytes = ctx.vState.maxBytes,
        .blocks = args.blocks,
        .tIdx = static_cast<int>(bIdx * PurlinAtom::THREADS + threadIdx.x),
        .world = ctx.world,
        .rank = ctx.rank,
        .bIdx = bIdx,
      };
    }

    template<typename Element, typename BT>
    __device__ __forceinline__
    static void runPackets(const SnacArgs<BT> &args, const Context &ctx)
      requires (op == ConsumeOp::reduce) {
      if constexpr (inputLayout == DataLayout::packed) {
        const auto redArgs = packedArgs(args, ctx);
        stage(redArgs);
        consume<Element>(redArgs);
        __syncthreads();
        markEpoch(ctx, redArgs.bIdx, redArgs.flag);
        markUnusedEpochs<PurlinAtom>(ctx, args.blocks, args.blocks, redArgs.flag, redArgs.tIdx);
        return;
      }
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
      // Cache this rank's window in every peer, plus the variable-size
      // partition table when the input carries one.
      auto **staging = reinterpret_cast<cuda::std::byte **>(typedWorkspace);
      auto *offsets = reinterpret_cast<size_t *>(staging + MAX_RANKS_PER_DOMAIN);
      size_t *sizesP = offsets + MAX_RANKS_PER_DOMAIN;
      for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
        const auto peerBase = ctx.stagingLR[peer];
        staging[peer] = peerBase + (stagingPrefix + rankOffset);
        if constexpr (inputLayout == DataLayout::scatteredV) {
          sizesP[peer] = sizes[peer];
        }
      }
      if constexpr (inputLayout == DataLayout::scatteredV) {
        // Compute the byte displacement of each variable-size partition.
        auto *__restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte *>(sizesP + MAX_RANKS_PER_DOMAIN);
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, scanWorkspace, ctx.world);
      }
      __syncthreads();
      const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
      const LRArgs redArgs{
        .src = src,
        .staging = staging,
        .localStaging = localStaging,
        .dst = dst,
        .flag = nextEpoch,
        .bufferStride = bufferStride,
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
      stage(redArgs);
      consume<Element>(redArgs);
      __syncthreads();
      markEpoch(ctx, bIdx, nextEpoch);
      markUnusedEpochs<PurlinAtom>(ctx, blocks, blocks, nextEpoch, tid);
    }

    // Write this rank's contribution into every peer's packet buffer.
    __device__ __forceinline__
    static void stage(const LRArgs& gArgs)
      requires (op == ConsumeOp::gather) {
      using VT = LRP::RT;
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(gArgs.src);
      const auto gridSize = Config::THREADS * gArgs.blocks;
      const auto elements = gArgs.bytes / sizeof(VT);
      const auto worldTrips = gArgs.world / Config::WORLD_UNROLL;
      const auto cutoff = worldTrips * Config::WORLD_UNROLL;
      if constexpr (inputLayout == DataLayout::packed || inputLayout == DataLayout::packedV) {
        for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
          const auto value = vS[idx];
          for (int t = 0; t < worldTrips; ++t) {
            cuda::std::byte* ptrs[Config::WORLD_UNROLL];
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = t * Config::WORLD_UNROLL + p;
              ptrs[p] = gArgs.staging[peer];
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = t * Config::WORLD_UNROLL + p;
              if (peer != gArgs.rank) {
                auto* __restrict__ packets = reinterpret_cast<LRP*>(ptrs[p]);
                packets[idx].write(value, gArgs.flag);
              }
            });
          }
          if (gArgs.world > cutoff) {
            for (int peer = cutoff; peer < gArgs.world; ++peer) {
              if (peer != gArgs.rank) {
                auto* __restrict__ packets = reinterpret_cast<LRP*>(gArgs.staging[peer]);
                packets[idx].write(value, gArgs.flag);
              }
            }
          }
        }
      }
      else if constexpr (inputLayout == DataLayout::scatteredV) {
        for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
          for (int t = 0; t < worldTrips; ++t) {
            cuda::std::byte* ptrs[Config::WORLD_UNROLL];
            LRP packets[Config::WORLD_UNROLL];
            int peers[Config::WORLD_UNROLL];
            size_t offsets[Config::WORLD_UNROLL];
            size_t sizes[Config::WORLD_UNROLL];
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = t * Config::WORLD_UNROLL + p;
              peers[p] = peer;
              ptrs[p] = gArgs.staging[peer];
              offsets[p] = gArgs.inOffsets[peer] / sizeof(VT);
              sizes[p] = gArgs.inSizes[peer] / sizeof(VT);
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = peers[p];
              const auto offset = offsets[p] + idx;
              const auto value = idx < sizes[p] ? vS[offset] : 0;
              packets[p] = LRP{value, gArgs.flag};
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = peers[p];
              if (peer != gArgs.rank) {
                auto* __restrict__ stagingPackets = reinterpret_cast<LRP*>(ptrs[p]);
                stagingPackets[idx].write(packets[p].data, packets[p].flag);
              }
            });
          }
          if (gArgs.world > cutoff) {
            for (int peer = cutoff; peer < gArgs.world; ++peer) {
              if (peer != gArgs.rank) {
                const auto offset = (gArgs.inOffsets[peer] / sizeof(VT)) + idx;
                const auto value = idx < (gArgs.inSizes[peer] / sizeof(VT)) ? vS[offset] : 0;
                auto* __restrict__ packets = reinterpret_cast<LRP*>(gArgs.staging[peer]);
                packets[idx].write(value, gArgs.flag);
              }
            }
          }
        }
      }
      else {
        for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
          for (int t = 0; t < worldTrips; ++t) {
            cuda::std::byte* ptrs[Config::WORLD_UNROLL];
            LRP packets[Config::WORLD_UNROLL];
            int peers[Config::WORLD_UNROLL];
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = t * Config::WORLD_UNROLL + p;
              peers[p] = peer;
              ptrs[p] = gArgs.staging[peer];
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = peers[p];
              const auto offset = static_cast<size_t>(peer) * elements + idx;
              const auto value = vS[offset];
              packets[p] = LRP{value, gArgs.flag};
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = peers[p];
              if (peer != gArgs.rank) {
                auto* __restrict__ stagingPackets = reinterpret_cast<LRP*>(ptrs[p]);
                stagingPackets[idx].write(packets[p].data, packets[p].flag);
              }
            });
          }
          if (gArgs.world > cutoff) {
            for (int peer = cutoff; peer < gArgs.world; ++peer) {
              if (peer != gArgs.rank) {
                const auto offset = static_cast<size_t>(peer) * elements + idx;
                const auto value = vS[offset];
                auto* __restrict__ packets = reinterpret_cast<LRP*>(gArgs.staging[peer]);
                packets[idx].write(value, gArgs.flag);
              }
            }
          }
        }
      }
    }

    // Gather every rank's contribution into the destination buffer, waiting on
    // each peer's packets as they arrive.
    __device__ __forceinline__
    static void consume(const LRArgs& gArgs)
      requires (op == ConsumeOp::gather) {
      using VT = LRP::RT;
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(gArgs.src);
      auto* __restrict__ vD = reinterpret_cast<VT*>(gArgs.dst);
      const auto gridSize = Config::THREADS * gArgs.blocks;
      const auto elements = gArgs.bytes / sizeof(VT);
      if constexpr (inputLayout == DataLayout::packedV || inputLayout == DataLayout::scatteredV) {
        const auto gatherElements = gArgs.maxBytes / sizeof(VT);
        for (int idx = gArgs.tIdx; idx < gatherElements; idx += gridSize) {
          for (int i = 0; i < gArgs.world; ++i) {
            const auto peer = (gArgs.rank + i) % gArgs.world;
            const auto peerElements = gArgs.sizes[peer] / sizeof(VT);
            if (idx < peerElements) {
              const auto peerOffset = gArgs.offsets[peer] / sizeof(VT);
              const auto offset = peerOffset + idx;
              if (peer == gArgs.rank) {
                if (!gArgs.isInPlace) {
                  const auto sourceOffset = inputLayout == DataLayout::packedV ? idx :
                    (gArgs.inOffsets[peer] / sizeof(VT)) + idx;
                  vD[offset] = vS[sourceOffset];
                }
              }
              else {
                const auto* __restrict__ packets = reinterpret_cast<const LRP*>(
                  gArgs.localStaging + gArgs.bufferStride * peer);
                vD[offset] = packets[idx].read(gArgs.flag);
              }
            }
          }
        }
      }
      else {
        for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
          for (int i = 0; i < gArgs.world; ++i) {
            const auto peer = (gArgs.rank + i) % gArgs.world;
            const auto offset = elements * peer + idx;
            if (peer == gArgs.rank) {
              if (!gArgs.isInPlace) {
                const auto sourceOffset = inputLayout == DataLayout::packed ? idx : offset;
                vD[offset] = vS[sourceOffset];
              }
            }
            else {
              const auto* __restrict__ packets = reinterpret_cast<const LRP*>(
                gArgs.localStaging + gArgs.bufferStride * peer);
              vD[offset] = packets[idx].read(gArgs.flag);
            }
          }
        }
      }
    }

    // Write this rank's input into the peers that reduce it. A direct
    // reduction sends the whole payload to every peer; the head of a composed
    // reduction sends each remote shard to the peer that owns it.
    __device__ __forceinline__
    static void stage(const LRArgs& redArgs)
      requires (op == ConsumeOp::reduce) {
      if constexpr (seam == Seam::head) {
        stagePartitioned(redArgs);
      } else {
        stageFullBuffer(redArgs);
      }
    }

    // Combine the packets received from every peer with the local contribution.
    // The head of a composed reduction reduces this rank's shard and publishes
    // the result as packets for the tail.
    template<typename Element>
    __device__ __forceinline__
    static void consume(const LRArgs& redArgs)
      requires (op == ConsumeOp::reduce) {
      if constexpr (seam == Seam::head) {
        reducePartitioned<Element>(redArgs);
      } else {
        consumeFullBuffer<Element>(redArgs);
      }
    }

    // Gather the head's result packets for this block group's remote peer.
    template<typename Element>
    __device__ __forceinline__
    static void consume(const LRArgs& redArgs)
      requires (op == ConsumeOp::gather && seam == Seam::tail) {
      gatherPartitioned<Element>(redArgs);
    }

    __device__ __forceinline__
    static void stageFullBuffer(const LRArgs& redArgs)
      requires (op == ConsumeOp::reduce) {
      using VT = LRP::RT;
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(redArgs.src);
      const auto gridSize = Config::THREADS * redArgs.blocks;
      const auto elements = redArgs.bytes / sizeof(VT);
      const auto worldTrips = redArgs.world / Config::WORLD_UNROLL;
      const auto peerStriped = isPeerStriped(redArgs);
      const auto cutoff = worldTrips * Config::WORLD_UNROLL;
      // Send this rank's input to the peers that participate in the reduction.
      if constexpr (inputLayout == DataLayout::scatteredV) {
        const auto putElems = redArgs.maxBytes / sizeof(VT);
        for (int idx = redArgs.tIdx; idx < putElems; idx += gridSize) {
          for (int t = 0; t < worldTrips; ++t) {
            cuda::std::byte* ptrs[Config::WORLD_UNROLL];
            LRP packets[Config::WORLD_UNROLL];
            int peers[Config::WORLD_UNROLL];
            size_t peerElems[Config::WORLD_UNROLL];
            size_t offsets[Config::WORLD_UNROLL];
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = t * Config::WORLD_UNROLL + p;
              peers[p] = peer;
              ptrs[p] = redArgs.staging[peer];
              peerElems[p] = redArgs.sizes[peer] / sizeof(VT);
              offsets[p] = redArgs.offsets[peer] / sizeof(VT);
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = peers[p];
              const auto offset = offsets[p] + idx;
              const auto value = idx < peerElems[p] ? vS[offset] : 0;
              packets[p] = LRP{value, redArgs.flag};
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = peers[p];
              if (peer != redArgs.rank) {
                auto* __restrict__ stagingPackets = reinterpret_cast<LRP*>(ptrs[p]);
                stagingPackets[idx].write(packets[p].data, packets[p].flag);
              }
            });
          }
          if (redArgs.world > cutoff) {
            for (int peer = cutoff; peer < redArgs.world; ++peer) {
              if (peer != redArgs.rank) {
                const auto offset = (redArgs.offsets[peer] / sizeof(VT)) + idx;
                const auto peerElem = redArgs.sizes[peer] / sizeof(VT);
                const auto value = idx < peerElem ? vS[offset] : 0;
                auto* __restrict__ packets = reinterpret_cast<LRP*>(redArgs.staging[peer]);
                packets[idx].write(value, redArgs.flag);
              }
            }
          }
        }
      }
      else if constexpr (inputLayout == DataLayout::packed) {
        if constexpr (Config::MEMTYPE == MemType::multimem) {
          auto* __restrict__ mcPackets = reinterpret_cast<LRP*>(redArgs.mcStaging);
          for (size_t idx = redArgs.tIdx; idx < elements; idx += gridSize) {
            multimemStPacket(mcPackets + idx, vS[idx], redArgs.flag);
          }
        }
        else if (!peerStriped) {
          for (size_t idx = redArgs.tIdx; idx < elements; idx += gridSize) {
            const auto value = vS[idx];
            for (int t = 0; t < worldTrips; ++t) {
              cuda::std::byte* ptrs[Config::WORLD_UNROLL];
              int peers[Config::WORLD_UNROLL];
              cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
                const auto peer = t * Config::WORLD_UNROLL + p;
                peers[p] = peer;
                ptrs[p] = redArgs.staging[peer] + redArgs.stagingOffset;
              });
              cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
                if (peers[p] != redArgs.rank) {
                  auto* __restrict__ packets = reinterpret_cast<LRP*>(ptrs[p]);
                  packets[idx].write(value, redArgs.flag);
                }
              });
            }
            if (redArgs.world > cutoff) {
              for (int peer = cutoff; peer < redArgs.world; ++peer) {
                if (peer != redArgs.rank) {
                  auto* __restrict__ packets = reinterpret_cast<LRP*>(
                    redArgs.staging[peer] + redArgs.stagingOffset);
                  packets[idx].write(value, redArgs.flag);
                }
              }
            }
          }
        }
        else {
          const auto laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
          const auto warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
          constexpr int warps = Config::THREADS / WARP_SIZE;
          for (int peerIdx = warpId; peerIdx < redArgs.world - 1; peerIdx += warps) {
            const auto peer = peerIdx < redArgs.rank ? peerIdx : peerIdx + 1;
            auto* __restrict__ packets = reinterpret_cast<LRP*>(
              redArgs.staging[peer] + redArgs.stagingOffset);
            for (size_t idx = laneId + static_cast<size_t>(redArgs.bIdx) * WARP_SIZE;
                 idx < elements; idx += static_cast<size_t>(redArgs.blocks) * WARP_SIZE) {
              packets[idx].write(vS[idx], redArgs.flag);
            }
          }
          // The striped reduce below repartitions elements across warps, so in
          // place a warp could overwrite an element that a sibling warp has not
          // yet sent. Every send that reads an element runs in the block that
          // reduces it, so a block barrier suffices to separate the phases.
          if (redArgs.src == redArgs.dst) {
            __syncthreads();
          }
        }
      }
      else {
        for (int idx = redArgs.tIdx; idx < elements; idx += gridSize) {
          for (int t = 0; t < worldTrips; ++t) {
            cuda::std::byte* ptrs[Config::WORLD_UNROLL];
            LRP packets[Config::WORLD_UNROLL];
            int peers[Config::WORLD_UNROLL];
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = t * Config::WORLD_UNROLL + p;
              peers[p] = peer;
              ptrs[p] = redArgs.staging[peer];
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = peers[p];
              const auto offset = static_cast<size_t>(peer) * elements + idx;
              const auto value = vS[offset];
              packets[p] = LRP{value, redArgs.flag};
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = peers[p];
              if (peer != redArgs.rank) {
                auto* __restrict__ stagingPackets = reinterpret_cast<LRP*>(ptrs[p]);
                stagingPackets[idx].write(packets[p].data, packets[p].flag);
              }
            });
          }
          if (redArgs.world > cutoff) {
            for (int peer = cutoff; peer < redArgs.world; ++peer) {
              if (peer != redArgs.rank) {
                const auto offset = static_cast<size_t>(peer) * elements + idx;
                const auto value = vS[offset];
                auto* __restrict__ packets = reinterpret_cast<LRP*>(redArgs.staging[peer]);
                packets[idx].write(value, redArgs.flag);
              }
            }
          }
        }
      }
    }

    template<typename Element>
    __device__ __forceinline__
    static void consumeFullBuffer(const LRArgs& redArgs)
      requires (op == ConsumeOp::reduce) {
      using VT = LRP::RT;
      constexpr RedOp reduceOp{};
      using VE = PackedElement<Element>::type; // Process elements in their packed vector form.
      using AccumType = PackedElement<ReduceAccumType<Element>>::type;
      using VERaw = DataToRawType<VE>::type;
      static_assert(alignof(VERaw) == alignof(VE) && sizeof(VERaw) == sizeof(VE));
      static_assert(sizeof(VT) % sizeof(VERaw) == 0 && alignof(VT) % alignof(VERaw) == 0);
      constexpr int vectorWidth = sizeof(VT) / sizeof(VERaw);
      using AVT = AlignedArray<AccumType, vectorWidth>;
      using LVT = AlignedArray<VERaw, vectorWidth>;
      static_assert(Config::ALIGNMENT_BYTES % alignof(VT) == 0 && Config::ALIGNMENT_BYTES % sizeof(VT) == 0);
      static_assert(cuda::std::is_trivially_copyable_v<LVT>);

      const auto* __restrict__ vS = reinterpret_cast<const VT*>(redArgs.src);
      auto* __restrict__ vD = reinterpret_cast<LVT*>(redArgs.dst);
      const auto gridSize = Config::THREADS * redArgs.blocks;
      const auto elements = redArgs.bytes / sizeof(VT);
      AVT accumulator{};
      constexpr Converter<AccumType, VE> loadConv{};
      constexpr Converter<VERaw, AccumType> storeConv{};
      constexpr typename RedOp::template Identity<AccumType> clear{};
      const auto peerStriped = isPeerStriped(redArgs);
      cuda::static_for<accumulator.size()>([&](auto i) {
        clear(accumulator[i]);
      });
      size_t firstElement = redArgs.tIdx;
      size_t elementStride = gridSize;
      // Multimem keeps the send-phase striping here: each thread then reduces
      // exactly the elements it multicast, so an in-place destination write
      // cannot race with another thread's send-phase read.
      if constexpr (inputLayout == DataLayout::packed && Config::MEMTYPE != MemType::multimem) {
        if (peerStriped) {
          constexpr int warps = Config::THREADS / WARP_SIZE;
          const auto laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
          const auto warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
          firstElement = laneId + static_cast<size_t>(redArgs.bIdx) * WARP_SIZE +
            static_cast<size_t>(warpId) * WARP_SIZE * redArgs.blocks;
          elementStride = static_cast<size_t>(warps) * WARP_SIZE * redArgs.blocks;
        }
      }
      for (size_t idx = firstElement; idx < elements; idx += elementStride) {
        const auto reducePeer = [&](const int peer) {
          LVT valRaw{};
          if (peer == redArgs.rank) {
            if constexpr (inputLayout == DataLayout::packed) {
              valRaw = cuda::std::bit_cast<LVT>(vS[idx]);
            }
            else if constexpr (inputLayout == DataLayout::scatteredV) {
              const auto peerElements = redArgs.sizes[peer] / sizeof(VT);
              const auto sourceOffset = redArgs.offsets[peer] / sizeof(VT);
              const auto value = idx < peerElements ? vS[sourceOffset + idx] : 0;
              valRaw = cuda::std::bit_cast<LVT>(value);
            }
            else {
              const auto sourceOffset = static_cast<size_t>(peer) * elements + idx;
              valRaw = cuda::std::bit_cast<LVT>(vS[sourceOffset]);
            }
          }
          else {
            const auto* __restrict__ packets = reinterpret_cast<const LRP*>(
              redArgs.localStaging + redArgs.bufferStride * peer);
            valRaw = cuda::std::bit_cast<LVT>(packets[idx].read(redArgs.flag));
          }
          AVT val{};
          cuda::static_for<val.size()>([&](auto i) {
            val[i] = loadConv(valRaw[i]);
          });
          reduceOp(accumulator, val);
        };
        constexpr int worldUnroll = Config::WORLD_UNROLL;
        #pragma unroll worldUnroll
        for (int peer = 0; peer < redArgs.world; ++peer) {
          reducePeer(peer);
        }
        // Convert and store the completed reduction.
        LVT resultRaw{};
        cuda::static_for<resultRaw.size()>([&](auto i) {
          resultRaw[i] = storeConv(accumulator[i]);
        });
        vD[idx] = resultRaw;
        cuda::static_for<resultRaw.size()>([&](auto i) {
          clear(accumulator[i]);
        });
      }
    }

    // Vector types of a composed latency reduction over Element. Each packet
    // payload carries vectorWidth elements.
    template<typename Element>
    struct PartitionedTypes {
      using Payload = LRP::RT;
      using VE = cuda::std::conditional_t<
        (sizeof(Payload) > sizeof(Element)), typename PackedElement<Element>::type, Element>;
      using AccumType = cuda::std::conditional_t<
        (sizeof(Payload) > sizeof(Element)), typename PackedElement<ReduceAccumType<Element>>::type,
        ReduceAccumType<Element>>;
      using VERaw = typename DataToRawType<VE>::type;
      static constexpr int vectorWidth = sizeof(Payload) / sizeof(VERaw);
      using AVT = AlignedArray<AccumType, vectorWidth>;
      using LVT = AlignedArray<VERaw, vectorWidth>;
      static_assert(sizeof(LVT) == sizeof(Payload));
    };
    // At the head of a composed reduction, each block group sends the shard
    // owned by its assigned remote peer. The block-group geometry is computed
    // here and again in gatherPartitioned on purpose: sharing it through a
    // struct changes the composed kernels' code on every architecture.
    __device__ __forceinline__
    static void stagePartitioned(const LRArgs& redArgs)
      requires (op == ConsumeOp::reduce) {
      using Payload = LRP::RT;
      const auto world = static_cast<int>(redArgs.world);
      const auto peers = world - 1;
      const auto packetsPerRank = redArgs.bytes / (static_cast<size_t>(world) * sizeof(LRP::RT));
      const auto blocksPerPeer = redArgs.blocks / peers;
      const auto localBlock = redArgs.bIdx % blocksPerPeer;
      const auto peerIdx = redArgs.bIdx / blocksPerPeer;
      const auto remoteRank = peerIdx < redArgs.rank ? peerIdx : peerIdx + 1;
      const auto peerStride = static_cast<size_t>(Config::THREADS) * blocksPerPeer;
      const auto groupTid = static_cast<size_t>(threadIdx.x) +
        static_cast<size_t>(localBlock) * Config::THREADS;
      const auto* __restrict__ source = reinterpret_cast<const Payload*>(redArgs.src);
      auto* __restrict__ remoteInputPackets = reinterpret_cast<LRP*>(
        redArgs.staging[remoteRank] + redArgs.stagingOffset);
      const auto sourceOffset = static_cast<size_t>(remoteRank) * packetsPerRank;
      for (size_t idx = groupTid; idx < packetsPerRank; idx += peerStride) {
        remoteInputPackets[idx].write(source[sourceOffset + idx], redArgs.flag);
      }
    }

    // The head of a composed reduction. Reduce this rank's shard in
    // deterministic rank order, store it in the destination, and publish it as
    // result packets to every remote peer. That publish is the tail's staged
    // data.
    template<typename Element>
    __device__ __forceinline__
    static void reducePartitioned(const LRArgs& redArgs)
      requires (op == ConsumeOp::reduce) {
      using Types = PartitionedTypes<Element>;
      using Payload = typename Types::Payload;
      using VE = typename Types::VE;
      using AccumType = typename Types::AccumType;
      using VERaw = typename Types::VERaw;
      using AVT = typename Types::AVT;
      using LVT = typename Types::LVT;

      const auto world = static_cast<int>(redArgs.world);
      const auto packetsPerRank = redArgs.bytes / (static_cast<size_t>(world) * sizeof(Payload));
      const auto* __restrict__ source = reinterpret_cast<const Payload*>(redArgs.src);
      auto* __restrict__ destination = reinterpret_cast<LVT*>(redArgs.dst);

      constexpr Converter<AccumType, VE> loadConv{};
      constexpr Converter<VERaw, AccumType> storeConv{};
      constexpr RedOp reduceOp{};
      constexpr typename RedOp::template Identity<AccumType> clear{};
      const auto rankSourceOffset = static_cast<size_t>(redArgs.rank) * packetsPerRank;
      const auto gridTid = static_cast<size_t>(threadIdx.x) +
        static_cast<size_t>(redArgs.bIdx) * Config::THREADS;
      const auto gridStride = static_cast<size_t>(Config::THREADS) * redArgs.blocks;
      for (size_t idx = gridTid; idx < packetsPerRank; idx += gridStride) {
        AVT accumulator{};
        cuda::static_for<accumulator.size()>([&](auto i) {
          clear(accumulator[i]);
        });
        const auto reducePeer = [&](const int peer) {
          LVT valueRaw{};
          if (peer == redArgs.rank) {
            valueRaw = cuda::std::bit_cast<LVT>(source[rankSourceOffset + idx]);
          }
          else {
            const auto* __restrict__ inputPackets = reinterpret_cast<const LRP*>(
              redArgs.localStaging + static_cast<size_t>(peer) * redArgs.bufferStride);
            valueRaw = cuda::std::bit_cast<LVT>(inputPackets[idx].read(redArgs.flag));
          }
          AVT value{};
          cuda::static_for<value.size()>([&](auto i) {
            value[i] = loadConv(valueRaw[i]);
          });
          reduceOp(accumulator, value);
        };
        constexpr int worldUnroll = Config::WORLD_UNROLL;
        #pragma unroll worldUnroll
        for (int peer = 0; peer < world; ++peer) {
          reducePeer(peer);
        }

        LVT result{};
        cuda::static_for<result.size()>([&](auto i) {
          result[i] = storeConv(accumulator[i]);
        });
        destination[rankSourceOffset + idx] = result;

        const auto rawResult = cuda::std::bit_cast<Payload>(result);
        if constexpr (Config::MEMTYPE == MemType::multimem) {
          auto* __restrict__ mcResultPackets =
            reinterpret_cast<LRP*>(redArgs.mcStaging + RESULT_OFFSET);
          multimemStPacket(mcResultPackets + idx, rawResult, redArgs.flag);
        }
        else {
          const auto publishPeer = [&](const int peer) {
            if (peer == redArgs.rank) return;
            auto* __restrict__ remoteResultPackets = reinterpret_cast<LRP*>(
              redArgs.staging[peer] + redArgs.stagingOffset + RESULT_OFFSET);
            remoteResultPackets[idx].write(rawResult, redArgs.flag);
          };
          #pragma unroll worldUnroll
          for (int peer = 0; peer < world; ++peer) {
            publishPeer(peer);
          }
        }
      }
    }

    // The tail of a composed reduction. Each block group reads the result
    // packets for the same remote peer it sent input to and stores that shard
    // in the destination.
    template<typename Element>
    __device__ __forceinline__
    static void gatherPartitioned(const LRArgs& redArgs)
      requires (op == ConsumeOp::gather) {
      using LVT = typename PartitionedTypes<Element>::LVT;
      const auto world = static_cast<int>(redArgs.world);
      const auto peers = world - 1;
      const auto packetsPerRank = redArgs.bytes / (static_cast<size_t>(world) * sizeof(LRP::RT));
      const auto blocksPerPeer = redArgs.blocks / peers;
      const auto localBlock = redArgs.bIdx % blocksPerPeer;
      const auto peerIdx = redArgs.bIdx / blocksPerPeer;
      const auto remoteRank = peerIdx < redArgs.rank ? peerIdx : peerIdx + 1;
      const auto peerStride = static_cast<size_t>(Config::THREADS) * blocksPerPeer;
      const auto groupTid = static_cast<size_t>(threadIdx.x) +
        static_cast<size_t>(localBlock) * Config::THREADS;
      auto* __restrict__ destination = reinterpret_cast<LVT*>(redArgs.dst);
      const auto* __restrict__ resultPackets = reinterpret_cast<const LRP*>(
        redArgs.localStaging + static_cast<size_t>(remoteRank) * redArgs.bufferStride + RESULT_OFFSET);
      const auto destinationOffset = static_cast<size_t>(remoteRank) * packetsPerRank;
      for (size_t idx = groupTid; idx < packetsPerRank; idx += peerStride) {
        destination[destinationOffset + idx] =
          cuda::std::bit_cast<LVT>(resultPackets[idx].read(redArgs.flag));
      }
    }
  };

  // Compose two SNACs into one collective. The tail's stage is the head's
  // consume: the head writes each reduced shard into the staging region the
  // tail reads and announces it as the tail's staged data, so the intermediate
  // never lands in a destination buffer. Today the only composition is reduce
  // (scattered -> packed) followed by gather (packed -> scattered), which is
  // allReduce. The checks below reject halves that cannot fit together; a
  // pair that passes them but is not allReduce has not been exercised.
  //
  // In the throughput regime the head reduces shard r in place over its own
  // input region r of this rank's staging, and the tail reads region r of
  // rank r's staging. The aliasing is safe because region r of rank r's
  // staging is read only by rank r's reducers. The gather-ready broadcast is
  // the tail's staged signal and, under cyclic staging, the drain for the
  // head's remote input regions. The two phases use different sections of the
  // grid, but every block derives the same epoch state from its own entry.
  //
  // In the latency regime the head's result packets are the tail's staged
  // data. Every block runs the head's stage,
  // the head's consume, and the tail's consume in sequence.
  consteval bool fixedSizeLayout(const DataLayout layout) {
    return layout != DataLayout::packedV && layout != DataLayout::scatteredV &&
      layout != DataLayout::transposedV;
  }
  // Attach a seam to an ordinary SNAC. The rebind keeps every other parameter,
  // so the latency specialization is selected the same way as before.
  template<typename S, Seam seam>
  struct WithSeam;
  template<typename PurlinAtom, typename CollConfig, ConsumeOp op, DataLayout inputLayout,
    DataLayout outputLayout, ReduceOp ro, Seam was, Seam seam>
  struct WithSeam<SNAC<PurlinAtom, CollConfig, op, inputLayout, outputLayout, ro, was>, seam> {
    using type = SNAC<PurlinAtom, CollConfig, op, inputLayout, outputLayout, ro, seam>;
  };
  template<typename CollConfig>
  consteval int composeTailBlocks() {
    if constexpr (regimeOf<CollConfig> == Regime::latency) {
      return 0;
    } else {
      return CollConfig::GATHER_BLOCKS;
    }
  }

  template<typename A, typename B>
  struct Compose {
    static_assert(A::SEAM == Seam::none && B::SEAM == Seam::none,
      "compose ordinary SNACs; Compose attaches the seams");
    using Head = typename WithSeam<A, Seam::head>::type;
    using Tail = typename WithSeam<B, Seam::tail>::type;
    using PurlinAtom = typename Head::AtomType;
    using CollConfig = typename Head::CollType;
    // Blocks that run the tail in the throughput regime; the head runs on the
    // rest. The count is a compose parameter that still lives in the
    // collective configuration as GATHER_BLOCKS. The latency regime runs both
    // halves in every block, so it has no tail blocks.
    static constexpr int TAIL_BLOCKS = composeTailBlocks<CollConfig>();
    static_assert(cuda::std::is_same_v<PurlinAtom, typename Tail::AtomType>,
      "a composition runs on one Atom");
    static_assert(cuda::std::is_same_v<CollConfig, typename Tail::CollType>,
      "a composition shares one collective configuration so its chunk flags line up");
    static_assert(Head::OUTPUT == Tail::INPUT,
      "the head's output layout is the tail's input layout");
    // A gather tail never reads its operator, but a composition carries one
    // operator, so a tail that could reduce can never silently differ.
    static_assert(Head::RO == Tail::RO, "the head and tail share one reduction operator");
    static_assert(Head::CAN_HEAD && Tail::CAN_TAIL,
      "the head's consume must publish into the seam and the tail's consume must read it");
    // Assumptions of the bodies below, not of composition in general: the
    // payload splits into equal shards of bytes / world, and the in-place seam
    // is safe only when region r of a rank's staging is read by that rank
    // alone, which a scattered head input guarantees.
    static_assert(fixedSizeLayout(Head::OUTPUT) && fixedSizeLayout(Tail::INPUT) &&
                  fixedSizeLayout(Tail::OUTPUT),
      "Compose splits the payload into equal shards; variable-size layouts are not supported");
    static_assert(Head::INPUT == DataLayout::scattered,
      "the in-place seam needs a scattered head input");

    template<typename Element, typename BT>
    __device__ __forceinline__
    static void run(const SnacArgs<BT> &args, const Context &ctx) {
      if constexpr (regimeOf<CollConfig> == Regime::latency) {
        runPackets<Element>(args, ctx);
      } else {
        runStaged<Element>(args, ctx);
      }
    }

    template<typename Element, typename BT>
    __device__ __forceinline__
    static void runStaged(const SnacArgs<BT> &args, const Context &ctx) {
      const auto &bIdx = args.bIdx;
      const auto epochState = makeEpochState(ctx, bIdx);
      const auto stagingPrefix = epochState.trStagingPrefix;
      // Each rank's shard of the payload: the head produces it and the tail
      // distributes it.
      const auto shardBytes = args.bytes / ctx.world_l;
      const auto headBlocks = args.blocks - TAIL_BLOCKS;
      // The seam is this rank's shard window in its own staging: the head
      // writes its result there and every rank's tail reads it. Cyclic
      // staging gives each shard a fixed window instead of a region sized to
      // shardBytes.
      constexpr auto cyclic = CollConfig::STAGING_MODE == StagingMode::cyclic;
      const auto seamOffset = cyclic ?
        static_cast<size_t>(static_cast<int>(ctx.cyclicSlots)) * CollConfig::CHUNK_SIZE *
          static_cast<size_t>(ctx.rank) :
        shardBytes * ctx.rank;
      if (bIdx < headBlocks) {
        auto *__restrict__ seam = ctx.staging[ctx.rank] + (stagingPrefix + seamOffset);
        Head::template run<Element>(
              SnacArgs<decltype(headBlocks)>{
                .dst = seam,
                .src = args.src,
                .bytes = shardBytes,
                .workspace = args.workspace,
                .blocks = headBlocks,
                .collBlocks = args.collBlocks,
                .bIdx = bIdx,
              }, ctx);
        return;
      }
      const auto tailBIdx = bIdx - headBlocks;
      // Use uneven mapping when the tail-block count is not divisible by the
      // world size. A uniform mapping would assign trailing blocks to a peer
      // that does not exist.
      const auto peerBlock = mapPeerBlockUneven(static_cast<int>(tailBIdx), TAIL_BLOCKS, ctx.world);
      Tail::consume(
        args.dst + (shardBytes * peerBlock.peer),
        shardBytes,
        args.workspace,
        ctx,
        epochState,
        bIdx,
        peerBlock,
        ctx.gatherSignals[ctx.rank],
        stagingPrefix
      );
    }

    template<typename Element, typename BT>
    __device__ __forceinline__
    static void runPackets(const SnacArgs<BT> &args, const Context &ctx) {
      const auto redArgs = Head::packedArgs(args, ctx);
      Head::stage(redArgs);
      Head::template consume<Element>(redArgs);
      Tail::template consume<Element>(redArgs);
      __syncthreads();
      markEpoch(ctx, redArgs.bIdx, redArgs.flag);
      markUnusedEpochs<PurlinAtom>(ctx, args.blocks, args.blocks, redArgs.flag, redArgs.tIdx);
    }
  };
}
#endif //PURLIN_SNAC_CUH
