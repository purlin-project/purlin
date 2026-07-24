//
// Created by osayamen on 5/25/26.
//

#ifndef PURLIN_REDUCE_CUH
#define PURLIN_REDUCE_CUH
#include "base.cuh"
#include "context.cuh"
#include "partition.cuh"
namespace purlin {
  template<typename PurlinAtom, DataLayout inputLayout, typename Element, typename BT>
  __device__ __forceinline__
  static void reduceLR(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& nextEpoch,
    const uint& senseBit,
    const size_t* __restrict__ const& sizes = nullptr) {
    const auto stagingPrefix = (senseBit * ctx.world * purlin::PACKET_BUFFER_SIZE);
    constexpr auto bufferStride = purlin::PACKET_BUFFER_SIZE;
    const auto rankOffset = ctx.rank * purlin::PACKET_BUFFER_SIZE;
    auto* __restrict__ base = ctx.stagingLR[ctx.rank];
    auto* __restrict__ localStaging = base + stagingPrefix;
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(typedWorkspace);
    auto* __restrict__ offsets = reinterpret_cast<size_t*>(staging + MAX_RANKS_PER_DOMAIN);
    auto* __restrict__ sizesP = offsets + MAX_RANKS_PER_DOMAIN;
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
      const auto peerBase = ctx.stagingLR[peer];
      staging[peer] = peerBase + (stagingPrefix + rankOffset);
      if constexpr (inputLayout == DataLayout::scatteredV) {
        sizesP[peer] = sizes[peer];
      }
    }
    if constexpr (inputLayout == DataLayout::scatteredV) {
      // compute displacements
      auto* __restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte*>(sizesP + MAX_RANKS_PER_DOMAIN);
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
    };
    PurlinAtom::template reduce<inputLayout>(redArgs, typedWorkspace);
    __syncthreads();
    markEpoch(ctx, bIdx, nextEpoch);
    markUnusedEpochs<PurlinAtom>(ctx, blocks, blocks, nextEpoch, tid);
  }

  template<
    typename PurlinAtom,
    int PUT_BLOCKS,
    DataLayout inputLayout,
    DataLayout outputLayout,
    typename Element,
    typename BT
  >
  __device__ __forceinline__
  static void reduceNonChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& nextEpoch,
    const size_t& stagingPrefix, const int& collBlocks) {
    constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    if (bIdx < PUT_BLOCKS) {
      const auto globalBytes = inputLayout == DataLayout::scatteredV ? ctx.vState.totalBytes :
        (inputLayout == DataLayout::scattered ? bytes * ctx.world : bytes);
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      const auto [bytesPut, putStartOffset] = partition<PUT_BLOCKS, alignmentBytes>(globalBytes, bIdx);
      const auto* __restrict__ srcP = src + putStartOffset;
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + stagingPrefix;
      auto* __restrict__ dstP = dstBase + putStartOffset;
      PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
      __syncthreads();
      if (threadIdx.x / WARP_SIZE == 0) {
        const auto laneId = threadIdx.x % WARP_SIZE;
        int shouldNotify = PUT_BLOCKS == 1 ? 1 : 0;
        if (!threadIdx.x) {
          cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
          shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == PUT_BLOCKS;
          if (shouldNotify) {
            s.store(0, cuda::memory_order_relaxed);
          }
        }
        __syncwarp();
        shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
        if (shouldNotify) {
          signalAllPeers(ctx.signals, ctx.rank, ctx.world, nextEpoch, laneId);
          __syncwarp();
        }
      }
      const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
      markEpoch(ctx, bIdx, nextEpoch);
      markUnusedEpochs<PurlinAtom, PUT_BLOCKS>(ctx, collBlocks, nextEpoch, tid);
      return;
    }
    // reducers
    const auto reduceBIdx = bIdx - PUT_BLOCKS;
    const auto reduceBlocks = blocks - PUT_BLOCKS;
    const auto [bytesRed, redStartOffset] = partition<alignmentBytes>(bytes, reduceBlocks, reduceBIdx);
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(workspace + PurlinAtom::RED_PIPELINE_SMEM_BYTES);
    auto* __restrict__ gatherSignals = reinterpret_cast<uint64_t**>(staging + MAX_RANKS_PER_DOMAIN);
    static_assert(sizeof(cuda::std::byte**) == sizeof(uint64_t**) && alignof(cuda::std::byte**) == alignof(uint64_t**));
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
      const auto offset = stagingPrefix + redStartOffset;
      staging[peer] = ctx.staging[peer] + (offset + (inputLayout == DataLayout::scatteredV ? ctx.vState.offset :
          inputLayout == DataLayout::scattered ? bytes * ctx.rank : 0));
      gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
    }
    cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
    const ReduceTRArgs redArgs{
      .sources = staging,
      .dst = dstP,
      .bytesRed = bytesRed,
      .world = ctx.world,
    };
    const auto warpId = threadIdx.x / WARP_SIZE;
    const auto laneId = threadIdx.x % WARP_SIZE;
    waitPeerArrivals<PurlinAtom>(ctx.signals[ctx.rank], redArgs.world, nextEpoch);
    __syncthreads();
    PurlinAtom::reduce(redArgs, typedWorkspace);
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

  template<
    typename PurlinAtom,
    int PUT_BLOCKS,
    size_t CHUNK_SIZE,
    DataLayout inputLayout,
    DataLayout outputLayout,
    typename Element,
    typename BT
  >
  __device__ __forceinline__
  static void reduceChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& epoch,
    const size_t& stagingPrefix, const int& collBlocks,
    const size_t* __restrict__ const& sizes = nullptr) {
    static_assert(CHUNK_SIZE >= MIN_CHUNK_SIZE);
    constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    //chunked-throughput regime
    if (bIdx < PUT_BLOCKS) {
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      static_assert(PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES + MAX_RANKS_PER_DOMAIN * sizeof(size_t));
      auto flag = epoch;
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
      const auto preBlockSetSize = inputLayout == DataLayout::packed ? PUT_BLOCKS : PUT_BLOCKS / ctx.world;
      const auto peerBlock = inputLayout == DataLayout::scatteredV ?
      (isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l) ?
        mapWeightedPeerBlock(bIdx, PUT_BLOCKS, sizesP, workspace, ctx.world) :
        mapPeerBlock(bIdx, preBlockSetSize)) : mapPeerBlock(bIdx, preBlockSetSize);
      const auto peer = inputLayout == DataLayout::packed ? 0 : peerBlock.peer;
      const auto intraBIdx = inputLayout == DataLayout::packed ? bIdx : peerBlock.intraIdx;
      const auto blockSetSize = inputLayout == DataLayout::packed ? PUT_BLOCKS : peerBlock.blockSetSize;
      const auto putBytes = inputLayout == DataLayout::scatteredV ? sizesP[peer] : bytes;
      const auto chunks = static_cast<int>(putBytes / CHUNK_SIZE);
      const auto cutoff = CHUNK_SIZE * chunks;
      if constexpr (inputLayout == DataLayout::scatteredV) {
        prefixSum<PurlinAtom::THREADS>(sizesP, offsets, workspace, ctx.world);
        __syncthreads();
      }
      const auto [bytesPut, putStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(blockSetSize, intraBIdx);
      const auto intraOffset = inputLayout == DataLayout::scatteredV ? offsets[peer] :
      inputLayout == DataLayout::packed ? 0 : peer * bytes;
      const auto* __restrict__ srcP = src + (putStartOffset + intraOffset);
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + (stagingPrefix + intraOffset);
      auto* __restrict__ dstP = dstBase + putStartOffset;
      const auto laneId = threadIdx.x % WARP_SIZE;
      auto* __restrict__ putCounter = inputLayout == DataLayout::packed ?
      ctx.putCounter : ctx.putCounter + peer * MAX_CHUNKS;
      for (int chunk = 0; chunk < chunks; ++chunk) {
        PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        flag++;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = blockSetSize == 1 ? 1 : 0;
          if (!laneId && blockSetSize > 1) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunk)};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            if constexpr (inputLayout == DataLayout::packed) {
              signalPointerList(signals, ctx.world, flag, laneId);
            }
            else {
              if (!laneId) {
                signalOne(signals[peer], flag);
              }
            }
            __syncwarp();
          }
        }
        dstP += CHUNK_SIZE;
        srcP += CHUNK_SIZE;
      }
      if (putBytes > cutoff) {
        const auto residue = putBytes - cutoff;
        const auto [bytesPutLeft, putStartOffsetLeft] = partition<alignmentBytes>(residue, blockSetSize, intraBIdx);
        srcP = src + ((CHUNK_SIZE * chunks + putStartOffsetLeft) + intraOffset);
        dstP = dstBase + (CHUNK_SIZE * chunks + putStartOffsetLeft);
        PurlinAtom::copy(dstP, srcP, bytesPutLeft, workspace);
        __syncthreads();
        flag++;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = blockSetSize == 1 ? 1 : 0;
          if (!laneId && blockSetSize > 1) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunks)};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            if constexpr (inputLayout == DataLayout::packed) {
              signalPointerList(signals, ctx.world, flag, laneId);
            }
            else {
              if (!laneId) {
                signalOne(signals[peer], flag);
              }
            }
            __syncwarp();
          }
        }
      }
      const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
      const auto nextEpoch = inputLayout == DataLayout::scatteredV ?
      epoch + cuda::ceil_div(ctx.vState.maxBytes, CHUNK_SIZE) : flag;
      markEpoch(ctx, bIdx, nextEpoch);
      markUnusedEpochs<PurlinAtom, PUT_BLOCKS>(ctx, collBlocks, nextEpoch, tid);
      return;
    }

    // reducer blocks
    const auto chunks = static_cast<int>(bytes / CHUNK_SIZE);
    const auto cutoff = CHUNK_SIZE * chunks;
    const auto reduceBIdx = bIdx - PUT_BLOCKS;
    const auto reduceBlocks = blocks - PUT_BLOCKS;
    const auto [bytesRed, redStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(reduceBlocks, reduceBIdx);
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + PurlinAtom::RED_PIPELINE_SMEM_BYTES);
    auto* __restrict__ gatherSignals = signals + MAX_RANKS_PER_DOMAIN;
    static_assert(sizeof(cuda::std::byte**) == sizeof(uint64_t**) && alignof(cuda::std::byte**) == alignof(uint64_t**));
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(gatherSignals + MAX_RANKS_PER_DOMAIN);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
      // signals
      signals[peer] = ctx.signals[ctx.rank] + peer;
      gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
      // staging
      const auto offset = stagingPrefix + redStartOffset;
      staging[peer] = ctx.staging[peer] + (offset + (inputLayout == DataLayout::scatteredV ? ctx.vState.offset :
          inputLayout == DataLayout::scattered ? bytes * ctx.rank : 0));
    }
    __syncthreads();
    auto flag = epoch;
    cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
    const auto warpId = threadIdx.x / WARP_SIZE;
    const auto laneId = threadIdx.x % WARP_SIZE;
    const auto tidS1 = (((warpId + (PurlinAtom::WARPS - 1)) % PurlinAtom::WARPS) * WARP_SIZE) + laneId;
    const auto tidS2 = PurlinAtom::WARPS == 1 ? threadIdx.x :
    (((warpId + (PurlinAtom::WARPS - 2)) % PurlinAtom::WARPS) * WARP_SIZE) + laneId;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      flag++;
      const ReduceTRArgs redArgs{
        .sources = staging,
        .dst = dstP,
        .bytesRed = bytesRed,
        .world = ctx.world,
      };
      waitPointerList<PurlinAtom>(signals, redArgs.world, flag, tidS2);
      __syncthreads();
      PurlinAtom::reduce(redArgs, typedWorkspace);
      __syncthreads();
      if constexpr (outputLayout == DataLayout::scattered) {
        // notify that chunk is done
        if (warpId == 0) {
          int shouldNotify = reduceBlocks == 1 ? 1 : 0;
          if (reduceBlocks > 1 && !laneId) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.redCounter + chunk)};
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
      dstP += CHUNK_SIZE;
      for (int i = static_cast<int>(tidS1); i < ctx.world; i += PurlinAtom::THREADS) {
        staging[i] += CHUNK_SIZE;
      }
    }
    if (bytes > cutoff) {
      flag++;
      const auto residue = bytes - cutoff;
      const auto [bytesRedLeft, redStartOffsetLeft] = partition<alignmentBytes>(residue, reduceBlocks, reduceBIdx);
      dstP = dst + (CHUNK_SIZE * chunks + redStartOffsetLeft);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
        auto* __restrict__ stagingBase = staging[i] - (chunks * CHUNK_SIZE + redStartOffset);
        staging[i] = stagingBase + (chunks * CHUNK_SIZE + redStartOffsetLeft);
      }
      const ReduceTRArgs redArgs{
        .sources = staging,
        .dst = dstP,
        .bytesRed = bytesRedLeft,
        .world = ctx.world,
      };
      waitPointerList<PurlinAtom>(signals, redArgs.world, flag);
      __syncthreads();
      PurlinAtom::reduce(redArgs, typedWorkspace);
      __syncthreads();
      if constexpr (outputLayout == DataLayout::scattered) {
        // notify that chunk is done
        if (warpId == 0) {
          int shouldNotify = reduceBlocks == 1 ? 1 : 0;
          if (reduceBlocks > 1 && !laneId) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.redCounter + chunks)};
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
    }
    const auto nextEpoch = inputLayout == DataLayout::scatteredV ?
      epoch + cuda::ceil_div(ctx.vState.maxBytes, CHUNK_SIZE) : flag;
    markEpoch(ctx, bIdx, nextEpoch);
  }
}
#endif //PURLIN_REDUCE_CUH
