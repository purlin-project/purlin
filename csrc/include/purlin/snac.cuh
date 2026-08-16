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
  // SNAC: Stage, Notify, And Consume — the protocol every purlin collective
  // lowers to. A collective stages its contribution into symmetric memory
  // (local-to-local), notifies the world, and consumes what it needs from its
  // peers, where consumption is either a gather or a reduce. The latency
  // regime specializes the protocol by fusing stage+notify into one
  // local-to-remote packet transfer whose flag rides with the data.
  //
  // SNAC sits between the collective (semantics: layouts, geometry, block-role
  // split) and the Atom (datapath: how bytes move or reduce per architecture).
  // It receives resolved geometry and never owns the grid-role partitioning.
  enum class ConsumeOp {
    gather,
    reduce
  };

  // How staged data is announced to its consumers.
  enum class Notify {
    allPeersDirect, // one whole-payload signal to every peer (non-chunked flat puts)
    pointerList,    // per-chunk broadcast through a shared-memory pointer list
    listEntry,      // per-chunk signal to one peer through the shared list
    one             // per-chunk signal to one peer through a direct pointer
  };

  // Derived wiring, not free parameters: who is notified of staged chunks and,
  // under ring staging, who publishes the drain that frees a slot. Everything
  // follows from the consume op, the layout pair, the datapath, and the
  // staging mode.
  template<typename PurlinAtom, typename CollConfig, ConsumeOp op,
    DataLayout inputLayout, DataLayout outputLayout>
  struct SnacTopology {
    static constexpr auto DATAPATH = PurlinAtom::BaseConfig::DATAPATH;
    static constexpr bool RING = CollConfig::STAGING_MODE == StagingMode::ring;
    // all2all stages one region per destination, consumed by that rank alone;
    // allGather and the reductions stage for every peer at once.
    static constexpr bool PER_DEST = op == ConsumeOp::gather &&
      (inputLayout == DataLayout::scattered || inputLayout == DataLayout::scatteredV);
    static constexpr Notify NOTIFY =
      CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked ?
        (PER_DEST ? Notify::one : Notify::allPeersDirect) :
        (PER_DEST ? Notify::one :
          (op == ConsumeOp::reduce && inputLayout != DataLayout::packed ?
            Notify::listEntry : Notify::pointerList));
    // Resident a2aV conveys its data-dependent staging offset in the first
    // notification; ring windows are static, so the ring signals plainly.
    static constexpr bool OFFSET_PACKET = PER_DEST &&
      inputLayout == DataLayout::scatteredV && !RING;
    // Epoch advance must be uniform across ranks; variable-length layouts
    // derive it from a globally agreed size rather than the local flag count.
    static constexpr bool UNIFORM_ADVANCE =
      inputLayout == DataLayout::packedV || inputLayout == DataLayout::scatteredV;
    enum class Drain {
      none,        // resident staging: the sense-bit double buffer suffices
      allRanks,    // every rank drains the window (allGather; the direct allReduce)
      single,      // one rank drains the region (reduceScatter, all2all)
      arUnicast,   // allReduce: owner's reducers for remote regions (their gather
                   // broadcast doubles as the drain), every rank's gathers for the
                   // local, result-carrying region
      localRegion  // multimem allReduce: the local gather set for the shard
    };
    static constexpr Drain DRAIN = !RING ? Drain::none :
      (op == ConsumeOp::gather ? (PER_DEST ? Drain::single : Drain::allRanks) :
        (inputLayout == DataLayout::packed ? Drain::allRanks :
          (outputLayout == DataLayout::packed ? Drain::single :
            (DATAPATH == Datapath::multimem ? Drain::localRegion : Drain::arUnicast))));
  };

  // Resolved stage geometry, computed by the collective.
  struct StageArgs {
    const cuda::std::byte* const src;      // source buffer base
    const size_t srcOffset = 0;            // this region's real intra offset within the source
    cuda::std::byte* const staging;        // own staging region base (epoch prefix + staging intra applied)
    const size_t bytes;                    // bytes this block set stages
    const size_t epochBytes = 0;           // uniform-advance driver when it is not vState-derived (a2aV)
    const PeerBlock block;                 // destination/shard identity and set geometry
    uint32_t* const putCounter;            // completion counter base for this set
    uint64_t** const signalList = nullptr; // pointer-list notify targets (shared memory)
    uint64_t* const signal = nullptr;      // direct single-peer notify target
    const size_t vPayload = 0;             // offset conveyed by the a2aV first-chunk packet
  };

  template<typename PurlinAtom, typename CollConfig, ConsumeOp op,
    DataLayout inputLayout, DataLayout outputLayout>
  struct SNAC {
    using Topology = SnacTopology<PurlinAtom, CollConfig, op, inputLayout, outputLayout>;
    using Drain = typename Topology::Drain;

    // Stage + notify for one producer block: copy the region into own staging
    // (whole, or chunk-by-chunk through ring slots under backpressure) and
    // publish per the topology. BLOCK_SET pins a compile-time set size where
    // the collective knows one; ACTIVE_BLOCKS pins the epoch-sweep breadth.
    template<int BLOCK_SET = AUTO, int ACTIVE_BLOCKS = AUTO>
    __device__ __forceinline__
    static void stage(const StageArgs& a,
      cuda::std::byte* __restrict__ const& workspace,
      const Context& ctx,
      const uint64_t& epoch,
      const uint64_t& nextEpoch,
      const int& bIdx,
      const int& collBlocks,
      const int& activeBlocks = 0) {
      constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      const auto blockSetSize = BLOCK_SET == AUTO ? a.block.blockSetSize : BLOCK_SET;
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        const auto [bytesPut, putStartOffset] = partition<alignmentBytes>(a.bytes, blockSetSize, a.block.intraIdx);
        const auto* __restrict__ srcP = a.src + (putStartOffset + a.srcOffset);
        auto* __restrict__ dstP = a.staging + putStartOffset;
        PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        if (threadIdx.x / WARP_SIZE == 0) {
          const auto laneId = threadIdx.x % WARP_SIZE;
          int shouldNotify = blockSetSize == 1 ? 1 : 0;
          if (!threadIdx.x) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*a.putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            if constexpr (Topology::NOTIFY == Notify::allPeersDirect) {
              signalAllPeers(ctx.signals, ctx.rank, ctx.world, nextEpoch, laneId);
            }
            else {
              if (!laneId) {
                if constexpr (Topology::OFFSET_PACKET) {
                  const auto sigPrefix = (epoch % 2) * ctx.world;
                  auto* __restrict__ signal = ctx.varOffsetSignals[a.block.peer] + (sigPrefix + ctx.rank);
                  signal->writeRelease(a.vPayload, nextEpoch);
                }
                else {
                  auto* __restrict__ signal = ctx.signals[a.block.peer] + ctx.rank;
                  signalOne(signal, nextEpoch);
                }
              }
            }
            __syncwarp();
          }
        }
        const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
        markEpoch(ctx, bIdx, nextEpoch);
        if constexpr (ACTIVE_BLOCKS == AUTO) {
          markUnusedEpochs<PurlinAtom>(ctx, collBlocks, activeBlocks, nextEpoch, tid);
        }
        else {
          markUnusedEpochs<PurlinAtom, ACTIVE_BLOCKS>(ctx, collBlocks, nextEpoch, tid);
        }
      }
      else {
        constexpr auto CHUNK_SIZE = CollConfig::CHUNK_SIZE;
        constexpr auto ring = Topology::RING;
        const auto chunks = static_cast<int>(a.bytes / CHUNK_SIZE);
        const auto cutoff = CHUNK_SIZE * chunks;
        auto flag = epoch;
        const auto [bytesPut, putStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(blockSetSize, a.block.intraIdx);
        const auto* __restrict__ srcP = a.src + (putStartOffset + a.srcOffset);
        auto* __restrict__ dstBase = a.staging;
        auto* __restrict__ dstP = dstBase + putStartOffset;
        const int laneId = static_cast<int>(threadIdx.x % WARP_SIZE);
        // ring: block before rewriting a slot until whoever drains this region
        // has published the drain of the slot's previous occupant.
        const int slots = ring ? static_cast<int>(ctx.ringSlots) : 0;
        const auto awaitDrain = [&](const uint64_t& target) {
          if constexpr (Topology::DRAIN == Drain::allRanks) {
            waitPeerArrivals<PurlinAtom>(ctx.consumedSignals[ctx.rank], ctx.world, target);
          }
          else if constexpr (Topology::DRAIN == Drain::single || Topology::DRAIN == Drain::localRegion) {
            if (!threadIdx.x) {
              waitUntilAtLeast(ctx.consumedSignals[ctx.rank] + a.block.peer, target);
            }
          }
          else if constexpr (Topology::DRAIN == Drain::arUnicast) {
            if (a.block.peer == ctx.rank) {
              waitPeerArrivals<PurlinAtom>(ctx.consumedSignals[ctx.rank], ctx.world, target);
            }
            else if (!threadIdx.x) {
              waitUntilAtLeast(ctx.gatherSignals[ctx.rank] + a.block.peer, target);
            }
          }
          __syncthreads();
        };
        for (int chunk = 0; chunk < chunks; ++chunk) {
          auto counterIdx = chunk;
          if constexpr (ring) {
            const int slot = chunk % ctx.ringSlots;
            counterIdx = slot;
            if (chunk >= slots) {
              awaitDrain(flag + 1 - static_cast<uint64_t>(slots));
            }
            dstP = dstBase + (static_cast<size_t>(slot) * CHUNK_SIZE + putStartOffset);
          }
          PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
          __syncthreads();
          flag++;
          if (threadIdx.x / WARP_SIZE == 0) {
            int shouldNotify = blockSetSize == 1 ? 1 : 0;
            if (blockSetSize > 1 && !laneId) {
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(a.putCounter + counterIdx)};
              shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
              if (shouldNotify) {
                s.store(0, cuda::memory_order_relaxed);
              }
            }
            __syncwarp();
            shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
            if (shouldNotify) {
              if constexpr (Topology::NOTIFY == Notify::pointerList) {
                signalPointerList(a.signalList, ctx.world, flag, laneId);
              }
              else if constexpr (Topology::NOTIFY == Notify::listEntry) {
                if (!laneId) {
                  signalOne(a.signalList[a.block.peer], flag);
                }
              }
              else {
                if (!laneId) {
                  if constexpr (Topology::OFFSET_PACKET) {
                    if (chunk == 0) {
                      const auto sigPrefix = (epoch % 2) * ctx.world;
                      auto* __restrict__ vSignal = ctx.varOffsetSignals[a.block.peer] + (sigPrefix + ctx.rank);
                      vSignal->writeRelease(a.vPayload, flag);
                    }
                    else {
                      signalOne(a.signal, flag);
                    }
                  }
                  else {
                    signalOne(a.signal, flag);
                  }
                }
              }
              __syncwarp();
            }
          }
          dstP += CHUNK_SIZE;
          srcP += CHUNK_SIZE;
        }
        if (a.bytes > cutoff) {
          const auto residue = a.bytes - cutoff;
          const auto [bytesPutLeft, putStartOffsetLeft] = partition<alignmentBytes>(residue, blockSetSize, a.block.intraIdx);
          srcP = a.src + ((CHUNK_SIZE * chunks + putStartOffsetLeft) + a.srcOffset);
          dstP = dstBase + (CHUNK_SIZE * chunks + putStartOffsetLeft);
          auto counterIdx = chunks;
          if constexpr (ring) {
            const int slot = chunks % ctx.ringSlots;
            counterIdx = slot;
            if (chunks >= slots) {
              awaitDrain(flag + 1 - static_cast<uint64_t>(slots));
            }
            dstP = dstBase + (static_cast<size_t>(slot) * CHUNK_SIZE + putStartOffsetLeft);
          }
          PurlinAtom::copy(dstP, srcP, bytesPutLeft, workspace);
          __syncthreads();
          flag++;
          if (threadIdx.x / WARP_SIZE == 0) {
            int shouldNotify = blockSetSize == 1 ? 1 : 0;
            if (blockSetSize > 1 && !laneId) {
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(a.putCounter + counterIdx)};
              shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
              if (shouldNotify) {
                s.store(0, cuda::memory_order_relaxed);
              }
            }
            __syncwarp();
            shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
            if (shouldNotify) {
              if constexpr (Topology::NOTIFY == Notify::pointerList) {
                signalPointerList(a.signalList, ctx.world, flag, laneId);
              }
              else if constexpr (Topology::NOTIFY == Notify::listEntry) {
                if (!laneId) {
                  signalOne(a.signalList[a.block.peer], flag);
                }
              }
              else {
                if (!laneId) {
                  if constexpr (Topology::OFFSET_PACKET) {
                    if (chunks == 0) {
                      const auto sigPrefix = (epoch % 2) * ctx.world;
                      auto* __restrict__ vSignal = ctx.varOffsetSignals[a.block.peer] + (sigPrefix + ctx.rank);
                      vSignal->writeRelease(a.vPayload, flag);
                    }
                    else {
                      signalOne(a.signal, flag);
                    }
                  }
                  else {
                    signalOne(a.signal, flag);
                  }
                }
              }
              __syncwarp();
            }
          }
        }
        const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
        const auto chunkedEpoch = chunkedNextEpoch(epoch,
          Topology::UNIFORM_ADVANCE ?
          cuda::ceil_div(Topology::PER_DEST ? a.epochBytes : ctx.vState.maxBytes, CHUNK_SIZE) :
          flag - epoch);
        markEpoch(ctx, bIdx, chunkedEpoch);
        if constexpr (ACTIVE_BLOCKS == AUTO) {
          markUnusedEpochs<PurlinAtom>(ctx, collBlocks, activeBlocks, chunkedEpoch, tid);
        }
        else {
          markUnusedEpochs<PurlinAtom, ACTIVE_BLOCKS>(ctx, collBlocks, chunkedEpoch, tid);
        }
      }
    }


    // The full throughput collective for one block: role-split the grid into
    // producers and consumers, resolve geometry, then stage or consume.
    template<typename BT>
    __device__ __forceinline__
    static void run(cuda::std::byte* __restrict__ const& dst,
      const cuda::std::byte* __restrict__ const& src,
      const size_t& bytes,
      cuda::std::byte* __restrict__ const& workspace, // shared
      const Context& ctx,
      const BT& blocks,
      const int& bIdx,
      const EpochState& epochState,
      const int& collBlocks,
      const size_t* __restrict__ const& sizes = nullptr,
      const size_t* __restrict__ const& inSizes = nullptr)
      requires (op == ConsumeOp::gather) {
      static_assert(PurlinAtom::REGIME == Regime::throughput);
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        static_assert(CollConfig::STAGING_MODE == StagingMode::resident);

      if constexpr ((inputLayout == DataLayout::packed && outputLayout == DataLayout::packed) ||
        (inputLayout == DataLayout::packedV && outputLayout == DataLayout::packedV)) {
        // AllGather
        constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        if (bIdx < CollConfig::PUT_BLOCKS) {
          const auto globalBytes = bytes;
          SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, inputLayout, outputLayout>::
            template stage<CollConfig::PUT_BLOCKS, CollConfig::PUT_BLOCKS>(
            StageArgs{
              .src = src,
              .staging = ctx.staging[ctx.rank] + epochState.trStagingPrefix,
              .bytes = globalBytes,
              .block = PeerBlock{.peer = 0, .intraIdx = bIdx, .blockSetSize = CollConfig::PUT_BLOCKS},
              .putCounter = ctx.putCounter,
            }, workspace, ctx, epochState.epoch, epochState.nextEpoch, bIdx, collBlocks);
          return;
        }
        const auto cBIdx = bIdx - CollConfig::PUT_BLOCKS;
        const auto consumerBlocks = static_cast<int>(blocks) - CollConfig::PUT_BLOCKS;
        const auto skewed = inputLayout != DataLayout::packed &&
          isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l);
        static_assert(inputLayout != DataLayout::packedV ||
          PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
        auto* __restrict__ sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        auto* __restrict__ offsets = sizesP + MAX_RANKS_PER_DOMAIN;
        if constexpr (inputLayout == DataLayout::packedV) {
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
            sizesP[i] = sizes[i];
          }
          prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
          __syncthreads();
        }
        const auto peerBlock = skewed ? mapWeightedPeerBlock(cBIdx, consumerBlocks, sizesP, workspace, ctx.world)
        : mapPeerBlock(cBIdx, consumerBlocks / ctx.world);
        const auto peerBytes = inputLayout == DataLayout::packedV ? sizes[peerBlock.peer] : bytes;
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
        // All2All
        static_assert(inputLayout != DataLayout::packed);
        constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        const int stagingBlocks = ctx.stagingBlocks;
        if (bIdx < stagingBlocks) {
          auto* sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
          auto* shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
          auto* offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
          if constexpr (inputLayout == DataLayout::scatteredV) {
            for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
              sizesP[i] = inSizes[i];
              shiftedSizes[i] = inSizes[(i + ctx.rank + 1) % ctx.world];
            }
            prefixSum<PurlinAtom::THREADS>(inSizes, offsets, workspace, ctx.world);
          }
          __syncthreads();
          static_assert(inputLayout != DataLayout::scatteredV ||
            PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
          const auto skewed = inputLayout == DataLayout::scatteredV &&
            isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l);
          const auto peerBlock =  skewed ? mapWeightedPeerBlock(bIdx, stagingBlocks, shiftedSizes, workspace, ctx.world) :
          mapPeerBlock(bIdx, stagingBlocks / ctx.actualWorld, ctx.rank, ctx.world);
          const auto peer = skewed ? ((peerBlock.peer + ctx.rank + 1) % ctx.world) : peerBlock.peer;
          // no chunking
          const auto myBytes = inputLayout == DataLayout::scattered ? bytes : inSizes[peer];
          const auto shiftOffset = inputLayout == DataLayout::scattered ? bytes * peer : offsets[peer];
          SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, inputLayout, outputLayout>::
            template stage<>(
            StageArgs{
              .src = src,
              .srcOffset = shiftOffset,
              .staging = ctx.staging[ctx.rank] + (epochState.trStagingPrefix + shiftOffset),
              .bytes = myBytes,
              .block = PeerBlock{.peer = peer, .intraIdx = peerBlock.intraIdx,
                .blockSetSize = peerBlock.blockSetSize},
              .putCounter = ctx.putCounter + peer,
              .vPayload = shiftOffset,
            }, workspace, ctx, epochState.epoch, epochState.nextEpoch, bIdx, collBlocks, stagingBlocks);
          return;
        }
        const auto totalPutBlocks = stagingBlocks + CollConfig::LOCAL_PUT_BLOCKS;
        if (bIdx < totalPutBlocks) {
          const auto myBytes = inputLayout == DataLayout::scattered ? bytes : sizes[ctx.rank];
          auto* inOffsets = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES) + MAX_RANKS_PER_DOMAIN;
          auto* outOffsets = inOffsets + MAX_RANKS_PER_DOMAIN;
          auto inOffset = bytes * ctx.rank;
          auto outOffset = bytes * ctx.rank;
          if constexpr (inputLayout == DataLayout::scatteredV) {
            prefixSum<PurlinAtom::THREADS>(inSizes, inOffsets, workspace, ctx.world);
            prefixSum<PurlinAtom::THREADS>(sizes, outOffsets, workspace, ctx.world);
            __syncthreads();
            inOffset = inOffsets[ctx.rank];
            outOffset = outOffsets[ctx.rank];
          }
          const auto lBIdx = bIdx - stagingBlocks;
          auto* __restrict__ srcP = src + inOffset;
          auto* __restrict__ dstP = dst + outOffset;
          superCopy<PurlinAtom, CollConfig::LOCAL_PUT_BLOCKS>(dstP, srcP, myBytes, workspace, lBIdx);
          markEpoch(ctx, bIdx, epochState.nextEpoch);
          return;
        }
        // consumers
        auto* sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        auto* shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
        auto* offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
        if constexpr (inputLayout == DataLayout::scatteredV) {
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
            sizesP[i] = sizes[i];
            shiftedSizes[i] = sizes[(i + ctx.rank + 1) % ctx.world];
          }
          prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
          __syncthreads();
        }
        const auto cBIdx = bIdx - totalPutBlocks;
        const auto consumerBlocks = static_cast<int>(blocks - totalPutBlocks);
        const auto skewed = (inputLayout == DataLayout::scatteredV &&
            isSkewed(ctx.vState.totalOutBytes, ctx.vState.maxOutBytes, ctx.world_l));
        auto peerBlock = skewed ?
          mapWeightedPeerBlock(cBIdx, consumerBlocks, shiftedSizes, workspace, ctx.world) :
          mapPeerBlock(cBIdx, consumerBlocks / ctx.actualWorld, ctx.rank, ctx.world);
        peerBlock.peer = skewed ? (peerBlock.peer + ctx.rank + 1) % ctx.world : peerBlock.peer;
        const auto offset = inputLayout==DataLayout::scatteredV ? offsets[peerBlock.peer]: bytes * peerBlock.peer;
        const auto peerBytes = inputLayout == DataLayout::scatteredV ? sizes[peerBlock.peer] : bytes;
        auto* __restrict__ sigBase = inputLayout == DataLayout::scatteredV ? nullptr : ctx.signals[ctx.rank];
        consume(
          dst + offset,
          peerBytes,
          workspace,
          ctx,
          epochState,
          bIdx,
          peerBlock,
          sigBase,
          epochState.trStagingPrefix
        );
      }
      }
      else {
        static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);

      constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      if constexpr ((inputLayout == DataLayout::packed && outputLayout == DataLayout::packed) ||
        (inputLayout == DataLayout::packedV && outputLayout == DataLayout::packedV)) {
        // AllGather
        if (bIdx < CollConfig::PUT_BLOCKS) {
          auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
            signals[i] = ctx.signals[i] + ctx.rank;
          }
          __syncthreads();
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
        auto* __restrict__ sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        auto* __restrict__ offsets = sizesP + MAX_RANKS_PER_DOMAIN;
        if constexpr (inputLayout == DataLayout::packedV) {
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
            sizesP[i] = sizes[i];
          }
          prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
          __syncthreads();
        }
        const auto peerBlock = skewed ? mapWeightedPeerBlock(cBIdx, consumerBlocks, sizesP, workspace, ctx.world)
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
          epochState.trStagingPrefix
        );
      }
      else {
        constexpr auto inPlace = false;
        // All2All
        const int stagingBlocks = ctx.stagingBlocks;
        if (bIdx < stagingBlocks) {
          auto* sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
          auto* shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
          auto* offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
          if constexpr (inputLayout == DataLayout::scatteredV) {
            for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
              sizesP[i] = inSizes[i];
              shiftedSizes[i] = inSizes[(i + ctx.rank + 1) % ctx.world];
            }
            prefixSum<PurlinAtom::THREADS>(inSizes, offsets, workspace, ctx.world);
          }
          __syncthreads();
          static_assert(inputLayout != DataLayout::scatteredV ||
            PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
          const auto skewed = (inputLayout == DataLayout::scatteredV &&
          isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l));
          auto peerBlock = skewed ?
          mapWeightedPeerBlock(bIdx, stagingBlocks, shiftedSizes, workspace, ctx.world) :
          mapPeerBlock(bIdx, stagingBlocks / ctx.actualWorld, ctx.rank, ctx.world);
          if constexpr (inputLayout == DataLayout::scatteredV) {
            peerBlock.peer = skewed ? ((peerBlock.peer + ctx.rank + 1) % ctx.world) : peerBlock.peer;
          }
          const auto myBytes = inputLayout == DataLayout::scattered ? bytes : inSizes[peerBlock.peer];
          auto* __restrict__ signal = ctx.signals[peerBlock.peer] + ctx.rank;
          const auto intraOffset = inputLayout == DataLayout::scattered ? bytes * peerBlock.peer :
          offsets[peerBlock.peer];
          constexpr auto ring = CollConfig::STAGING_MODE == StagingMode::ring;
          // ring: the source keeps its real offsets while staging is windowed per
          // destination; the consumer statically knows its window, so no offset
          // conveyance is needed and every chunk signals plainly.
          const int slots = ring ? static_cast<int>(ctx.ringSlots) : 0;
          const auto stagingIntraOffset = ring ?
            static_cast<size_t>(slots) * CollConfig::CHUNK_SIZE * static_cast<size_t>(peerBlock.peer) :
            intraOffset;
          SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, inputLayout, outputLayout>::
            template stage<>(
            StageArgs{
              .src = src,
              .srcOffset = intraOffset,
              .staging = ctx.staging[ctx.rank] + (epochState.trStagingPrefix + stagingIntraOffset),
              .bytes = myBytes,
              .epochBytes = bytes,
              .block = peerBlock,
              .putCounter = ctx.putCounter + peerBlock.peer * MAX_CHUNKS,
              .signal = signal,
              .vPayload = intraOffset,
            }, workspace, ctx, epochState.epoch, epochState.nextEpoch, bIdx, collBlocks, stagingBlocks);
          return;
        }
        const auto totalPutBlocks = stagingBlocks + (inPlace ? 0 : CollConfig::LOCAL_PUT_BLOCKS);
        if (!inPlace && bIdx < totalPutBlocks) {
          const auto myBytes = inputLayout == DataLayout::scattered ? bytes : sizes[ctx.rank];
          auto* inOffsets = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES) + MAX_RANKS_PER_DOMAIN;
          auto* outOffsets = inOffsets + MAX_RANKS_PER_DOMAIN;
          auto inOffset = bytes * ctx.rank;
          auto outOffset = bytes * ctx.rank;
          if constexpr (inputLayout == DataLayout::scatteredV) {
            prefixSum<PurlinAtom::THREADS>(inSizes, inOffsets, workspace, ctx.world);
            prefixSum<PurlinAtom::THREADS>(sizes, outOffsets, workspace, ctx.world);
            __syncthreads();
            inOffset = inOffsets[ctx.rank];
            outOffset = outOffsets[ctx.rank];
          }
          const auto lBIdx = bIdx - stagingBlocks;
          auto* __restrict__ srcP = src + inOffset;
          auto* __restrict__ dstP = dst + outOffset;
          if constexpr (inputLayout == DataLayout::scatteredV) {
            constexpr auto stageBytes = static_cast<size_t>(PurlinAtom::STAGE_BYTES);
            constexpr auto partGranularity = stageBytes * CollConfig::LOCAL_PUT_BLOCKS;
            const auto paddedBytes = alignUp(myBytes, partGranularity);
            const auto [bytesP, startOffset] = partition<CollConfig::LOCAL_PUT_BLOCKS, static_cast<int>(stageBytes)>(paddedBytes, lBIdx);
            const auto actualBytes = startOffset >= myBytes ? size_t{0} :
              cuda::std::min(bytesP, myBytes - startOffset);
            PurlinAtom::copy(dstP + startOffset, srcP + startOffset, actualBytes, workspace);
          }
          else {
            superCopy<PurlinAtom, CollConfig::LOCAL_PUT_BLOCKS>(dstP, srcP, myBytes, workspace, lBIdx);
          }
          constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
          const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
            inputLayout == DataLayout::scatteredV ?
            cuda::ceil_div(bytes, chunkSize) :
            static_cast<size_t>(cuda::ceil_div(myBytes, chunkSize)));
          markEpoch(ctx, bIdx, nextEpoch);
          return;
        }
        // consumers
        const auto cBIdx = bIdx - totalPutBlocks;
        const auto consumerBlocks = static_cast<int>(blocks - totalPutBlocks);
        auto* sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        auto* shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
        auto* offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
        if constexpr (inputLayout == DataLayout::scatteredV) {
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
            sizesP[i] = sizes[i];
            shiftedSizes[i] = sizes[(i + ctx.rank + 1) % ctx.world];
          }
          prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
          __syncthreads();
        }
        const auto skewed = (inputLayout == DataLayout::scatteredV &&
            isSkewed(ctx.vState.totalOutBytes, ctx.vState.maxOutBytes, ctx.world_l));
        auto peerBlock = skewed ?
          mapWeightedPeerBlock(cBIdx, consumerBlocks, shiftedSizes, workspace, ctx.world) :
          mapPeerBlock(cBIdx, consumerBlocks / ctx.actualWorld, ctx.rank, ctx.world);
        if constexpr (inputLayout == DataLayout::scatteredV) {
          peerBlock.peer = skewed ? (peerBlock.peer + ctx.rank + 1) % ctx.world : peerBlock.peer;
        }
        const auto offset = inputLayout==DataLayout::scatteredV ? offsets[peerBlock.peer]: bytes * peerBlock.peer;
        const auto peerBytes = inputLayout == DataLayout::scatteredV ? sizes[peerBlock.peer] : bytes;
        consume(
          dst + offset,
          peerBytes,
          workspace,
          ctx,
          epochState,
          bIdx,
          peerBlock,
          ctx.signals[ctx.rank],
          epochState.trStagingPrefix, bytes
        );
      }
      }
    }

    template<typename Element, typename BT>
    __device__ __forceinline__
    static void run(cuda::std::byte* __restrict__ const& dst,
      const cuda::std::byte* __restrict__ const& src,
      const size_t& bytes,
      Element* __restrict__ const& typedWorkspace, // shared
      const Context& ctx,
      const BT& blocks,
      const int& bIdx,
      const EpochState& epochState,
      const int& collBlocks,
      const size_t* __restrict__ const& sizes = nullptr)
      requires (op == ConsumeOp::reduce) {
      static_assert(PurlinAtom::REGIME == Regime::throughput);
      constexpr auto PUT_BLOCKS = CollConfig::PUT_BLOCKS;
      constexpr auto multimem = Topology::DATAPATH == Datapath::multimem;
      static_assert(!multimem || (inputLayout == DataLayout::scattered &&
        (outputLayout == DataLayout::scattered || outputLayout == DataLayout::packed)),
        "the multimem datapath serves shard-partitioned staging reductions only");
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        const auto& nextEpoch = epochState.nextEpoch;
        const auto& stagingPrefix = epochState.trStagingPrefix;

      if (bIdx < PUT_BLOCKS) {
        const auto globalBytes = inputLayout == DataLayout::scatteredV ? ctx.vState.totalBytes :
          (inputLayout == DataLayout::scattered ? bytes * ctx.world : bytes);
        auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
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
      // reducers
      consume(
        dst, bytes, typedWorkspace, ctx, blocks - PUT_BLOCKS, bIdx - PUT_BLOCKS, bIdx,
        nextEpoch, nextEpoch, stagingPrefix);
      }
      else {
        constexpr auto CHUNK_SIZE = CollConfig::CHUNK_SIZE;
        static_assert(CHUNK_SIZE >= MIN_CHUNK_SIZE);
        constexpr auto ring = Topology::RING;
        const auto& epoch = epochState.epoch;
        const auto& stagingPrefix = epochState.trStagingPrefix;

      //chunked-throughput regime
      if (bIdx < PUT_BLOCKS) {
        auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
        static_assert(PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES + MAX_RANKS_PER_DOMAIN * sizeof(size_t));
        auto* __restrict__ sizesP = reinterpret_cast<size_t*>(workspace + WEIGHTED_PEER_BLOCK_STATE_BYTES);
        auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        auto* __restrict__ offsets = reinterpret_cast<size_t*>(signals + MAX_RANKS_PER_DOMAIN);
        for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
          signals[i] = ctx.signals[i] + ctx.rank;
          if constexpr (inputLayout == DataLayout::scatteredV) {
            sizesP[i] = sizes[i];
          }
        }
        __syncthreads();
        // uneven split: worlds that do not divide the put-block count would
        // otherwise map trailing blocks to a nonexistent peer
        const auto uniformPeerBlock = inputLayout == DataLayout::packed ?
          mapPeerBlock(bIdx, PUT_BLOCKS) : mapPeerBlockUneven(bIdx, PUT_BLOCKS, ctx.world);
        const auto peerBlock = inputLayout == DataLayout::scatteredV ?
        (isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l) ?
          mapWeightedPeerBlock(bIdx, PUT_BLOCKS, sizesP, workspace, ctx.world) :
          uniformPeerBlock) : uniformPeerBlock;
        const auto peer = inputLayout == DataLayout::packed ? 0 : peerBlock.peer;
        const auto intraBIdx = inputLayout == DataLayout::packed ? bIdx : peerBlock.intraIdx;
        const auto blockSetSize = inputLayout == DataLayout::packed ? PUT_BLOCKS : peerBlock.blockSetSize;
        const auto putBytes = inputLayout == DataLayout::scatteredV ? sizesP[peer] : bytes;
        if constexpr (inputLayout == DataLayout::scatteredV) {
          prefixSum<PurlinAtom::THREADS>(sizesP, offsets, workspace, ctx.world);
          __syncthreads();
        }
        const auto intraOffset = inputLayout == DataLayout::scatteredV ? offsets[peer] :
        inputLayout == DataLayout::packed ? 0 : peer * bytes;
        // ring: the source keeps its real offsets while staging is windowed
        const int slots = ring ? static_cast<int>(ctx.ringSlots) : 0;
        const auto stagingIntraOffset = ring ?
          (inputLayout == DataLayout::packed ? size_t{0} :
            static_cast<size_t>(slots) * CHUNK_SIZE * static_cast<size_t>(peer)) : intraOffset;
        stage<(inputLayout == DataLayout::packed ? PUT_BLOCKS : AUTO), PUT_BLOCKS>(
          StageArgs{
            .src = src,
            .srcOffset = intraOffset,
            .staging = ctx.staging[ctx.rank] + (stagingPrefix + stagingIntraOffset),
            .bytes = putBytes,
            .block = PeerBlock{.peer = peer, .intraIdx = intraBIdx, .blockSetSize = blockSetSize},
            .putCounter = inputLayout == DataLayout::packed ?
              ctx.putCounter : ctx.putCounter + peer * MAX_CHUNKS,
            .signalList = signals,
          }, workspace, ctx, epoch, epoch, bIdx, collBlocks);
        return;
      }

      // reducer blocks
      consume(
        dst, bytes, typedWorkspace, ctx, blocks - PUT_BLOCKS, bIdx - PUT_BLOCKS, bIdx,
        epoch, epoch, stagingPrefix);
      }
    }

    // Gather consume for one consumer block: wait for the producer's per-chunk
    // (or whole-payload) publication, copy the region out of staging, and under
    // ring staging publish the drain that frees each slot.
    __device__ __forceinline__
    static void consume(cuda::std::byte* __restrict__ const& dst,
      const size_t& bytes,
      cuda::std::byte* __restrict__ const& workspace,
      const Context& ctx,
      const EpochState& epochState,
      const int& bIdx,
      const PeerBlock& peerBlock,
      uint64_t* __restrict__ const& signalBase,
      const size_t& stagingPrefix, const size_t& globalMaxBytes = 0)
      requires (op == ConsumeOp::gather) {
      // Under the multimem allReduce the reduced shards were broadcast into every
      // replica, so the gather is a local read of this rank's own staging.
      constexpr auto localGather =
        Topology::DATAPATH == Datapath::multimem && outputLayout == DataLayout::scattered;
      size_t sourceOffset = 0;
      if constexpr (outputLayout == DataLayout::scattered) {
        sourceOffset = bytes * peerBlock.peer;
      }
      else if constexpr (outputLayout == DataLayout::transposed) {
        sourceOffset = bytes * ctx.rank;
      }
      const auto sigPrefix = (epochState.epoch % 2) * ctx.world;
      auto* __restrict__ vSignal = ctx.varOffsetSignals[ctx.rank] + (sigPrefix + peerBlock.peer);
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        auto* __restrict__ srcOffset = reinterpret_cast<size_t*>(workspace);
        if (!threadIdx.x) {
          if constexpr (outputLayout == DataLayout::transposedV) {
            const auto cv = vSignal->waitUntilAtLeast(epochState.nextEpoch);
            *srcOffset = cv.data; // obtain offset
            cuda::std::ignore = vSignal->loadAcquire();
          }
          else {
            auto* __restrict__ signal = signalBase + peerBlock.peer;
            waitUntilAtLeast(signal, epochState.nextEpoch);
          }
        }
        __syncthreads();
        if constexpr (outputLayout == DataLayout::transposedV) {
          sourceOffset = *srcOffset;
          __syncthreads(); // <- ensures everyone has read the above
        }
        const auto* __restrict__ srcBase =
          ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + sourceOffset);
        const auto* __restrict__ srcP = srcBase;
        auto* __restrict__ dstP = dst;
        superCopy<PurlinAtom>(dstP, srcP, bytes, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        markEpoch(ctx, bIdx, epochState.nextEpoch);
      }
      else if constexpr (Topology::RING) {
        constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
        const int slots = ctx.ringSlots;
        const auto windowBytes = static_cast<size_t>(slots) * chunkSize;
        // fixed ring windows replace the payload-sized staging regions
        size_t regionOffset = 0;
        if constexpr (outputLayout == DataLayout::scattered) {
          regionOffset = windowBytes * static_cast<size_t>(peerBlock.peer);
        }
        else if constexpr (outputLayout == DataLayout::transposed || outputLayout == DataLayout::transposedV) {
          regionOffset = windowBytes * static_cast<size_t>(ctx.rank);
        }
        const auto* __restrict__ srcBase =
          ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + regionOffset);
        auto* __restrict__ dstP = dst;
        const auto chunks = static_cast<int>(bytes / chunkSize);
        const auto chunkCutoff = chunkSize * chunks;
        auto flag = epochState.epoch;
        auto* __restrict__ signal = signalBase + peerBlock.peer;
        // Remote regions publish their drain to the staging owner; under the
        // multimem gather every read is of the local replica, whose producer polls
        // the local, region-indexed entry instead.
        auto* __restrict__ consumedSignal = localGather ?
          ctx.consumedSignals[ctx.rank] + peerBlock.peer :
          ctx.consumedSignals[peerBlock.peer] + ctx.rank;
        auto* __restrict__ consumedCounter = ctx.consumedCounter + peerBlock.peer * MAX_CHUNKS;
        for (int i = 0; i < chunks; ++i) {
          flag++;
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          const auto slot = i % ctx.ringSlots;
          const auto* __restrict__ srcP = srcBase + static_cast<size_t>(slot) * chunkSize;
          superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
            peerBlock.blockSetSize, peerBlock.intraIdx);
          signalConsumed(consumedCounter + slot, consumedSignal, peerBlock.blockSetSize, flag);
          dstP += chunkSize;
        }
        if (bytes > chunkCutoff) {
          flag++;
          const auto residue = bytes - chunkCutoff;
          const auto slot = chunks % ctx.ringSlots;
          dstP = dst + chunkCutoff;
          const auto* __restrict__ srcP = srcBase + static_cast<size_t>(slot) * chunkSize;
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          superCopy<PurlinAtom>(dstP, srcP, residue, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
          signalConsumed(consumedCounter + slot, consumedSignal, peerBlock.blockSetSize, flag);
        }
        const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
          outputLayout == DataLayout::transposedV ? cuda::ceil_div(globalMaxBytes, chunkSize) :
          (outputLayout == DataLayout::packedV ?
            cuda::ceil_div(ctx.vState.maxBytes, chunkSize) : flag - epochState.epoch));
        markEpoch(ctx, bIdx, nextEpoch);
      }
      else {
        auto* __restrict__ srcBase =
          ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + sourceOffset);
        auto* __restrict__ srcP = srcBase;
        auto* __restrict__ dstP = dst;
        constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
        const auto chunks = static_cast<int>(bytes / CollConfig::CHUNK_SIZE);
        const auto chunkCutoff = CollConfig::CHUNK_SIZE * chunks;
        auto flag = epochState.epoch;
        auto* __restrict__ srcOffset = reinterpret_cast<size_t*>(workspace);
        auto* __restrict__ signal = signalBase + peerBlock.peer;
        if constexpr (outputLayout == DataLayout::transposedV) {
          if (chunks > 0) {
            flag++;
            if (!threadIdx.x) {
              const auto cv = vSignal->waitUntilAtLeast(flag);
              *srcOffset = cv.data;
              cuda::std::ignore = vSignal->loadAcquire();
            }
            __syncthreads();
            sourceOffset = *srcOffset;
            srcP += sourceOffset;
            srcBase += sourceOffset;
            __syncthreads();
            superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
              peerBlock.blockSetSize, peerBlock.intraIdx);
            srcP += CollConfig::CHUNK_SIZE;
            dstP += CollConfig::CHUNK_SIZE;
          }
          for (int i = 1; i < chunks; ++i) {
            flag++;
            if (!threadIdx.x) {
              waitUntilAtLeast(signal, flag);
            }
            __syncthreads();
            superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
              peerBlock.blockSetSize, peerBlock.intraIdx);
            srcP += CollConfig::CHUNK_SIZE;
            dstP += CollConfig::CHUNK_SIZE;
          }
        }
        else {
          for (int i = 0; i < chunks; ++i) {
            flag++;
            if (!threadIdx.x) {
              waitUntilAtLeast(signal, flag);
            }
            __syncthreads();
            superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
              peerBlock.blockSetSize, peerBlock.intraIdx);
            srcP += CollConfig::CHUNK_SIZE;
            dstP += CollConfig::CHUNK_SIZE;
          }
        }
        if (bytes > chunkCutoff) {
          flag++;
          const auto residue = bytes - chunkCutoff;
          dstP = dst + (CollConfig::CHUNK_SIZE * chunks);
          srcP = srcBase + (CollConfig::CHUNK_SIZE * chunks);
          if (!threadIdx.x) {
            if constexpr (outputLayout == DataLayout::transposedV) {
              if (chunks == 0) {
                const auto cv = vSignal->waitUntilAtLeast(flag);
                *srcOffset = cv.data;
                cuda::std::ignore = vSignal->loadAcquire();
              }
              else {
                waitUntilAtLeast(signal, flag);
              }
            }
            else {
              waitUntilAtLeast(signal, flag);
            }
          }
          __syncthreads();
          if constexpr (outputLayout == DataLayout::transposedV) {
            if (chunks == 0) {
              sourceOffset = *srcOffset;
              srcP += sourceOffset;
              __syncthreads();
            }
          }
          superCopy<PurlinAtom>(dstP, srcP, residue, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        }
        const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
          outputLayout == DataLayout::transposedV ? cuda::ceil_div(globalMaxBytes, chunkSize) :
          (outputLayout == DataLayout::packedV ?
            cuda::ceil_div(ctx.vState.maxBytes, chunkSize) : flag - epochState.epoch));
        markEpoch(ctx, bIdx, nextEpoch);
      }
    }

    // Reduce consume for one reducer block: wait for every producer's per-chunk
    // (or whole-payload) publication, reduce the shard slice across the world's
    // replicas, and publish per the topology — the gather broadcast when the
    // result feeds a gather stage, the drain broadcast under ring staging when
    // the result leaves staging directly.
    template<typename Element, typename RB>
    __device__ __forceinline__
    static void consume(cuda::std::byte* __restrict__ const& dst,
      const size_t& bytes,
      Element* __restrict__ const& typedWorkspace,
      const Context& ctx,
      const RB& reduceBlocks,
      const int& reduceBIdx,
      const int& bIdx,
      const uint64_t& epoch,
      const uint64_t& nextEpoch,
      const size_t& stagingPrefix)
      requires (op == ConsumeOp::reduce) {
      constexpr auto multimem = Topology::DATAPATH == Datapath::multimem;
      constexpr auto ring = Topology::RING;
      constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        const auto [bytesRed, redStartOffset] = partition<alignmentBytes>(bytes, reduceBlocks, reduceBIdx);
        auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
        auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(workspace + PurlinAtom::RED_PIPELINE_SMEM_BYTES);
        auto* __restrict__ gatherSignals = reinterpret_cast<uint64_t**>(staging + MAX_RANKS_PER_DOMAIN);
        static_assert(sizeof(cuda::std::byte**) == sizeof(uint64_t**) && alignof(cuda::std::byte**) == alignof(uint64_t**));
        for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
          if constexpr (!multimem) {
            const auto offset = stagingPrefix + redStartOffset;
            staging[peer] = ctx.staging[peer] + (offset + (inputLayout == DataLayout::scatteredV ? ctx.vState.offset :
                inputLayout == DataLayout::scattered ? bytes * ctx.rank : 0));
          }
          gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
        }
        cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
        const ReduceTRArgs redArgs{
          .sources = staging,
          .mcSource = multimem ?
            ctx.mcStagingTR + (stagingPrefix + redStartOffset + bytes * ctx.rank) : nullptr,
          .dst = dstP,
          .bytesRed = bytesRed,
          .world = ctx.world,
        };
        const auto warpId = threadIdx.x / WARP_SIZE;
        const auto laneId = threadIdx.x % WARP_SIZE;
        waitPeerArrivals<PurlinAtom>(ctx.signals[ctx.rank], redArgs.world, nextEpoch);
        __syncthreads();
        PurlinAtom::template reduce<reduceResultOf(outputLayout)>(redArgs, typedWorkspace);
        if constexpr (outputLayout == DataLayout::scattered) {
          __syncthreads();
          // notify that chunk is done
          if (warpId == 0) {
            int shouldNotify = reduceBlocks == 1 ? 1 : 0;
            if (reduceBlocks > 1 && !laneId) {
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.redCounter};
              shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == reduceBlocks;
              if (shouldNotify) {
                s.store(0, cuda::memory_order_relaxed);
              }
            }
            __syncwarp();
            shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
            if (shouldNotify) {
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
        auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
        auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + PurlinAtom::RED_PIPELINE_SMEM_BYTES);
        auto* __restrict__ gatherSignals = signals + MAX_RANKS_PER_DOMAIN;
        static_assert(sizeof(cuda::std::byte**) == sizeof(uint64_t**) && alignof(cuda::std::byte**) == alignof(uint64_t**));
        auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(gatherSignals + MAX_RANKS_PER_DOMAIN);
        // ring drain broadcast targets; live only where reducers are the sole
        // consumers of the staged regions they read (unicast-result reductions)
        auto* __restrict__ consumed = reinterpret_cast<uint64_t**>(staging + MAX_RANKS_PER_DOMAIN);
        const int slots = ring ? static_cast<int>(ctx.ringSlots) : 0;
        for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
          // signals
          signals[peer] = ctx.signals[ctx.rank] + peer;
          gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
          if constexpr (ring && outputLayout == DataLayout::packed) {
            consumed[peer] = ctx.consumedSignals[peer] + ctx.rank;
          }
          if constexpr (!multimem) {
            // staging
            const auto offset = stagingPrefix + redStartOffset;
            const auto regionOffset = ring ?
              (inputLayout == DataLayout::packed ? size_t{0} :
                static_cast<size_t>(slots) * CHUNK_SIZE * static_cast<size_t>(ctx.rank)) :
              (inputLayout == DataLayout::scatteredV ? ctx.vState.offset :
                inputLayout == DataLayout::scattered ? bytes * ctx.rank : 0);
            staging[peer] = ctx.staging[peer] + (offset + regionOffset);
          }
        }
        __syncthreads();
        auto flag = epoch;
        // ring: the multicast alias mirrors the unicast layout, so the shard region
        // is the same fixed window and the per-chunk offset wraps by slot.
        const auto mcRegionOffset = ring ?
          static_cast<size_t>(slots) * CHUNK_SIZE * static_cast<size_t>(ctx.rank) : bytes * ctx.rank;
        auto* __restrict__ mcPtr = multimem ?
          ctx.mcStagingTR + (stagingPrefix + redStartOffset + mcRegionOffset) : nullptr;
        cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
        const auto warpId = threadIdx.x / WARP_SIZE;
        const auto laneId = threadIdx.x % WARP_SIZE;
        const auto tidS1 = (((warpId + (PurlinAtom::WARPS - 1)) % PurlinAtom::WARPS) * WARP_SIZE) + laneId;
        const auto tidS2 = PurlinAtom::WARPS == 1 ? threadIdx.x :
        (((warpId + (PurlinAtom::WARPS - 2)) % PurlinAtom::WARPS) * WARP_SIZE) + laneId;
        for (int chunk = 0; chunk < chunks; ++chunk) {
          flag++;
          auto counterIdx = chunk;
          if constexpr (ring) {
            const int slot = chunk % ctx.ringSlots;
            counterIdx = slot;
            if constexpr (outputLayout == DataLayout::scattered) {
              // the reduced result lands back in the local shard's ring window
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
          PurlinAtom::template reduce<reduceResultOf(outputLayout)>(redArgs, typedWorkspace);
          __syncthreads();
          if constexpr (outputLayout == DataLayout::scattered) {
            // notify that chunk is done
            if (warpId == 0) {
              int shouldNotify = reduceBlocks == 1 ? 1 : 0;
              if (reduceBlocks > 1 && !laneId) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.redCounter + counterIdx)};
                shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == reduceBlocks;
                if (shouldNotify) {
                  s.store(0, cuda::memory_order_relaxed);
                }
              }
              __syncwarp();
              shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
              if (shouldNotify) {
                signalPointerList(gatherSignals, ctx.world, flag, laneId);
              }
            }
          }
          else if constexpr (ring) {
            // drain notification: the producers' ring slots are free to rewrite
            if (warpId == 0) {
              int shouldNotify = reduceBlocks == 1 ? 1 : 0;
              if (reduceBlocks > 1 && !laneId) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.redCounter + counterIdx)};
                shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == reduceBlocks;
                if (shouldNotify) {
                  s.store(0, cuda::memory_order_relaxed);
                }
              }
              __syncwarp();
              shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
              if (shouldNotify) {
                signalPointerList(consumed, ctx.world, flag, laneId);
              }
            }
          }
          dstP += CHUNK_SIZE;
          if constexpr (multimem) {
            if constexpr (ring) {
              const int nextSlot = (chunk + 1) % ctx.ringSlots;
              mcPtr += nextSlot == 0 ?
                -static_cast<ptrdiff_t>((static_cast<size_t>(slots) - 1) * CHUNK_SIZE) :
                static_cast<ptrdiff_t>(CHUNK_SIZE);
            }
            else {
              mcPtr += CHUNK_SIZE;
            }
          }
          else {
            for (int i = static_cast<int>(tidS1); i < ctx.world; i += PurlinAtom::THREADS) {
              if constexpr (ring) {
                const int nextSlot = (chunk + 1) % ctx.ringSlots;
                staging[i] += nextSlot == 0 ?
                  -static_cast<ptrdiff_t>((static_cast<size_t>(slots) - 1) * CHUNK_SIZE) :
                  static_cast<ptrdiff_t>(CHUNK_SIZE);
              }
              else {
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
          if constexpr (ring) {
            counterIdx = chunks % ctx.ringSlots;
          }
          if constexpr (!multimem) {
            for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
              if constexpr (ring) {
                // the chunk loop already advanced staging to the residue's slot
                staging[i] = (staging[i] - redStartOffset) + redStartOffsetLeft;
              }
              else {
                auto* __restrict__ stagingBase = staging[i] - (chunks * CHUNK_SIZE + redStartOffset);
                staging[i] = stagingBase + (chunks * CHUNK_SIZE + redStartOffsetLeft);
              }
            }
          }
          if constexpr (ring && outputLayout == DataLayout::scattered) {
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
          PurlinAtom::template reduce<reduceResultOf(outputLayout)>(redArgs, typedWorkspace);
          __syncthreads();
          if constexpr (outputLayout == DataLayout::scattered) {
            // notify that chunk is done
            if (warpId == 0) {
              int shouldNotify = reduceBlocks == 1 ? 1 : 0;
              if (reduceBlocks > 1 && !laneId) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.redCounter + counterIdx)};
                shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == reduceBlocks;
                if (shouldNotify) {
                  s.store(0, cuda::memory_order_relaxed);
                }
              }
              __syncwarp();
              shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
              if (shouldNotify) {
                signalPointerList(gatherSignals, ctx.world, flag, laneId);
              }
            }
          }
          else if constexpr (ring) {
            // drain notification for the residue slot
            if (warpId == 0) {
              int shouldNotify = reduceBlocks == 1 ? 1 : 0;
              if (reduceBlocks > 1 && !laneId) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.redCounter + counterIdx)};
                shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == reduceBlocks;
                if (shouldNotify) {
                  s.store(0, cuda::memory_order_relaxed);
                }
              }
              __syncwarp();
              shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
              if (shouldNotify) {
                signalPointerList(consumed, ctx.world, flag, laneId);
              }
            }
          }
        }
        const auto chunkedEpoch = chunkedNextEpoch(epoch, inputLayout == DataLayout::scatteredV ?
          cuda::ceil_div(ctx.vState.maxBytes, CHUNK_SIZE) : flag - epoch);
        markEpoch(ctx, bIdx, chunkedEpoch);
      }
    }
  };

  // Latency-regime SNAC: stage and notify fuse into a single local-to-remote
  // packet transfer whose flag rides with the data; consume is the flag-guarded
  // packet read (plus the local reduction for the reduce op).
  template<typename PurlinAtom, ConsumeOp op, DataLayout inputLayout, DataLayout outputLayout>
  struct SNAC<PurlinAtom, CollectiveConfigLR, op, inputLayout, outputLayout> {
    template<typename BT = int>
    __device__ __forceinline__
    static void run(cuda::std::byte* __restrict__ const& dst,
      const cuda::std::byte* __restrict__ const& src,
      const size_t& bytes,
      cuda::std::byte* __restrict__ const& workspace, // shared
      const Context& ctx,
      const BT& blocks,
      const int& bIdx,
      const EpochState& epochState,
      const int&, // collBlocks: the latency regime sweeps every epoch itself
      const size_t* __restrict__ const& sizes = nullptr,
      const size_t* __restrict__ const& inSizes = nullptr)
      requires (op == ConsumeOp::gather) {
      const auto& nextEpoch = epochState.nextEpoch;
      const auto& senseBit = epochState.senseBit;
      const auto stagingPrefix = (senseBit * ctx.world * purlin::PACKET_BUFFER_SIZE);
      constexpr auto bufferStride = purlin::PACKET_BUFFER_SIZE;
      const auto rankOffset = ctx.rank * purlin::PACKET_BUFFER_SIZE;
      auto* __restrict__ localStaging = ctx.stagingLR[ctx.rank] + stagingPrefix;
      // shared memory state
      auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(workspace);
      auto* __restrict__ offsets = reinterpret_cast<size_t*>(staging + MAX_RANKS_PER_DOMAIN);
      auto* __restrict__ sizesP = offsets + MAX_RANKS_PER_DOMAIN;
      auto* __restrict__ inSizesP = sizesP + MAX_RANKS_PER_DOMAIN;
      auto* __restrict__ inOffsetsP = inSizesP + MAX_RANKS_PER_DOMAIN;
      static_assert(inputLayout != DataLayout::scatteredV || (PurlinAtom::COPY_SMEM_SIZE >= sizeof(void*) * 5));
      for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
        staging[peer] = ctx.stagingLR[peer] + (stagingPrefix + rankOffset);
        if constexpr (inputLayout == DataLayout::packedV) {
          sizesP[peer] = sizes[peer];
        }
        else if constexpr (inputLayout == DataLayout::scatteredV) {
          sizesP[peer] = sizes[peer];
          inSizesP[peer] = inSizes[peer];
        }
      }
      if constexpr (inputLayout == DataLayout::packedV) {
        auto* __restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte*>(sizesP + MAX_RANKS_PER_DOMAIN);
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, scanWorkspace, ctx.world);
      }
      else if constexpr (inputLayout == DataLayout::scatteredV) {
        auto* __restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte*>(inOffsetsP + MAX_RANKS_PER_DOMAIN);
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, scanWorkspace, ctx.world);
        prefixSum<PurlinAtom::THREADS>(inSizes, inOffsetsP, scanWorkspace, ctx.world);
      }
      const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
      __syncthreads();
      const auto isInPlace = inputLayout == DataLayout::packed ? src == (dst + ctx.rank * bytes) : false; // TODO
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

    template<bool partitioned = false, typename Element, typename BT = int>
    __device__ __forceinline__
    static void run(cuda::std::byte* __restrict__ const& dst,
      const cuda::std::byte* __restrict__ const& src,
      const size_t& bytes,
      Element* __restrict__ const& typedWorkspace, // shared
      const Context& ctx,
      const BT& blocks,
      const int& bIdx,
      const EpochState& epochState,
      const int&, // collBlocks: the latency regime sweeps every epoch itself
      const size_t* __restrict__ const& sizes = nullptr)
      requires (op == ConsumeOp::reduce) {
      const auto& nextEpoch = epochState.nextEpoch;
      const auto& senseBit = epochState.senseBit;
      const auto stagingPrefix = (senseBit * ctx.world * purlin::PACKET_BUFFER_SIZE);
      constexpr auto bufferStride = purlin::PACKET_BUFFER_SIZE;
      const auto rankOffset = ctx.rank * purlin::PACKET_BUFFER_SIZE;
      auto* __restrict__ base = ctx.stagingLR[ctx.rank];
      auto* __restrict__ localStaging = base + stagingPrefix;
      cuda::std::byte** staging = nullptr;
      size_t* offsets = nullptr;
      size_t* sizesP = nullptr;
      if constexpr (inputLayout == DataLayout::packed) {
        staging = ctx.stagingLR;
      }
      else {
        staging = reinterpret_cast<cuda::std::byte**>(typedWorkspace);
        offsets = reinterpret_cast<size_t*>(staging + MAX_RANKS_PER_DOMAIN);
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
        // compute displacements
        auto* __restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte*>(sizesP + MAX_RANKS_PER_DOMAIN);
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
        .mcStaging = PurlinAtom::BaseConfig::DATAPATH == Datapath::multimem && inputLayout == DataLayout::packed ?
          ctx.mcStagingLR + (stagingPrefix + rankOffset) : nullptr,
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
      static_assert(!partitioned || inputLayout == DataLayout::packed);
      PurlinAtom::template reduce<inputLayout, partitioned>(redArgs, typedWorkspace);
      __syncthreads();
      markEpoch(ctx, bIdx, nextEpoch);
      markUnusedEpochs<PurlinAtom>(ctx, blocks, blocks, nextEpoch, tid);
    }
  };
}
#endif //PURLIN_SNAC_CUH
