//
// Created by osayamen on 5/25/26.
//

#ifndef SUTURE_GATHER_CUH
#define SUTURE_GATHER_CUH
#include "base.cuh"
#include "context.cuh"
#include "partition.cuh"
#include "transfer.cuh"

namespace suture {
  __host__ __forceinline__
  auto getGatherRegime(const size_t& bytes, const int& world) {
    if (world == 8) {
      if (bytes <= 1024) {
        return Regime::latency;
      }
      return Regime::throughput;
    }
    if (world == 4) {
      if (bytes <= 64 * 1024) {
        return Regime::latency;
      }
      return Regime::throughput;
    }
    if (bytes <= RED_LATENCY_BOUND_THRESHOLD) {
      return Regime::latency;
    }
    return Regime::throughput;
  }

  template<typename SutureAtom, typename CollConfig, DataLayout outputLayout>
  __device__ __forceinline__
  static void gatherConsumer(cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const Context& ctx,
    const EpochState& epochState,
    const int bIdx,
    const int consumerBIdx,
    const int consumerBlocks,
    uint64_t* __restrict__ const& signalBase,
    const size_t stagingPrefix) {
    const auto blockSetSize = consumerBlocks / ctx.world;
    const auto peerBlock = mapPeerBlock(consumerBIdx, blockSetSize);
    auto* __restrict__ signal = signalBase + peerBlock.peer;
    size_t sourceOffset = 0;
    if constexpr (outputLayout == DataLayout::scattered) {
      sourceOffset = bytes * peerBlock.peer;
    }
    else if constexpr (outputLayout == DataLayout::transposed) {
      sourceOffset = bytes * ctx.rank;
    }
    const auto* __restrict__ srcBase = ctx.staging[peerBlock.peer] + (stagingPrefix + sourceOffset);
    auto* __restrict__ dstBase = dst + bytes * peerBlock.peer;
    const auto* __restrict__ srcP = srcBase;
    auto* __restrict__ dstP = dstBase;
    if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
      if (!threadIdx.x) {
        waitUntilAtLeast(signal, epochState.nextEpoch);
      }
      __syncthreads();
      superGet<SutureAtom>(dstP, srcP, bytes, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
      markEpoch(ctx, bIdx, epochState.nextEpoch);
    }
    else {
      const auto chunks = static_cast<int>(bytes / CollConfig::CHUNK_SIZE);
      const auto chunkCutoff = CollConfig::CHUNK_SIZE * chunks;
      auto flag = epochState.epoch;
      for (int i = 0; i < chunks; ++i) {
        flag++;
        if (!threadIdx.x) {
          waitUntilAtLeast(signal, flag);
        }
        __syncthreads();
        superGet<SutureAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        srcP += CollConfig::CHUNK_SIZE;
        dstP += CollConfig::CHUNK_SIZE;
      }
      if (bytes > chunkCutoff) {
        flag++;
        const auto residue = bytes - chunkCutoff;
        dstP = dstBase + (CollConfig::CHUNK_SIZE * chunks);
        srcP = srcBase + (CollConfig::CHUNK_SIZE * chunks);
        if (!threadIdx.x) {
          waitUntilAtLeast(signal, flag);
        }
        __syncthreads();
        superGet<SutureAtom>(dstP, srcP, residue, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
      }
      markEpoch(ctx, bIdx, flag);
    }
  }

  template<typename SutureAtom, DataLayout inputLayout, typename BT = int>
  __device__ __forceinline__
  static void gatherLR(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& nextEpoch,
    const uint& senseBit) {
    static_assert(inputLayout == DataLayout::packed || inputLayout == DataLayout::scattered);
    const auto stagingPrefix = (senseBit * ctx.world * suture::PACKET_BUFFER_SIZE);
    const auto isInPlace = src == (dst + ctx.rank * bytes);
    constexpr auto bufferStride = suture::PACKET_BUFFER_SIZE;
    const auto rankOffset = ctx.rank * suture::PACKET_BUFFER_SIZE;
    auto* __restrict__ localStaging = ctx.stagingLR[ctx.rank] + stagingPrefix;
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(workspace);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      staging[peer] = ctx.stagingLR[peer] + (stagingPrefix + rankOffset);
    }
    const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
    __syncthreads();
    const LRArgs gArgs{
      .src = src,
      .staging = staging,
      .localStaging = localStaging,
      .dst = dst,
      .flag = nextEpoch,
      .bufferStride = bufferStride,
      .bytes = bytes,
      .blocks = blocks,
      .tIdx = static_cast<int>(tid),
      .world = ctx.world,
      .rank = ctx.rank,
      .isInPlace = isInPlace,
    };
    fascia::gather<typename SutureAtom::BaseConfig, inputLayout>(gArgs);
    __syncthreads();
    markEpoch(ctx, bIdx, nextEpoch);
    markUnusedEpochs<SutureAtom>(ctx, blocks, blocks, nextEpoch, tid);
  }

  template<
    typename SutureAtom,
    typename CollConfig,
    DataLayout inputLayout,
    DataLayout outputLayout,
    typename BT
  >
  __device__ __forceinline__
  static void gatherNonChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const EpochState& epochState,
    const int& collBlocks) {
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    if (bIdx < CollConfig::PUT_BLOCKS) {
      const auto globalBytes = inputLayout == DataLayout::scattered ? bytes * ctx.world : bytes;
      // no chunking
      const auto [bytesPut, putStartOffset] = partition<CollConfig::PUT_BLOCKS, alignmentBytes>(globalBytes, bIdx);
      const auto* __restrict__ srcP = src + putStartOffset;
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + epochState.trStagingPrefix;
      auto* __restrict__ dstP = dstBase + putStartOffset;
      SutureAtom::put(dstP, srcP, bytesPut, workspace);
      __syncthreads();
      if (threadIdx.x / WARP_SIZE == 0) {
        const auto laneId = threadIdx.x % WARP_SIZE;
        int shouldNotify = CollConfig::PUT_BLOCKS == 1 ? 1 : 0;
        if (!threadIdx.x) {
          cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
          shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == CollConfig::PUT_BLOCKS;
          if (shouldNotify) {
            s.store(0, cuda::memory_order_relaxed);
          }
        }
        __syncwarp();
        shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
        if (shouldNotify) {
          signalAllPeers(ctx.signals, ctx.rank, ctx.world, epochState.nextEpoch, laneId);
          __syncwarp();
        }
      }
      const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
      markEpoch(ctx, bIdx, epochState.nextEpoch);
      markUnusedEpochs<SutureAtom, CollConfig::PUT_BLOCKS>(ctx, collBlocks, epochState.nextEpoch, tid);
      return;
    }
    // consumers
    const auto cBIdx = bIdx - CollConfig::PUT_BLOCKS;
    gatherConsumer<SutureAtom, CollConfig, outputLayout>(
      dst,
      bytes,
      workspace,
      ctx,
      epochState,
      bIdx,
      cBIdx,
      static_cast<int>(blocks - CollConfig::PUT_BLOCKS),
      ctx.signals[ctx.rank],
      epochState.trStagingPrefix
    );
  }

  template<
    typename SutureAtom,
    typename CollConfig,
    DataLayout inputLayout,
    DataLayout outputLayout,
    typename BT
  >
  __device__ __forceinline__
  static void gatherChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const EpochState& epochState,
    const int& collBlocks) {
    static_assert(inputLayout == DataLayout::packed || inputLayout == DataLayout::scattered);
    const auto chunks = static_cast<int>(bytes / CollConfig::CHUNK_SIZE);
    const auto cutoff = CollConfig::CHUNK_SIZE * chunks;
    static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    if (bIdx < CollConfig::PUT_BLOCKS) {
      const auto packedPeerBlock = PeerBlock{
        .peer = 0,
        .intraIdx = bIdx,
        .blockSetSize = CollConfig::PUT_BLOCKS,
      };
      const auto peerBlock = inputLayout == DataLayout::packed ? packedPeerBlock :
      mapPeerBlock(bIdx, CollConfig::PUT_BLOCKS / ctx.world);
      auto flag = epochState.epoch;
      auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::COPY_PIPELINE_SMEM_BYTES);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
        signals[i] = ctx.signals[i] + ctx.rank;
      }
      __syncthreads();
      const auto [bytesPut, putStartOffset] = partition<CollConfig::CHUNK_SIZE, alignmentBytes>
      (peerBlock.blockSetSize, peerBlock.intraIdx);
      const auto intraOffset = peerBlock.peer * bytes;
      const auto* __restrict__ srcP = src + (putStartOffset + intraOffset);
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + (epochState.trStagingPrefix + intraOffset);
      auto* __restrict__ dstP = dstBase + putStartOffset;
      const int laneId = static_cast<int>(threadIdx.x % WARP_SIZE);
      auto* __restrict__ putCounter = ctx.putCounter + peerBlock.peer * MAX_CHUNKS;
      for (int chunk = 0; chunk < chunks; ++chunk) {
        SutureAtom::put(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        flag++;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = peerBlock.blockSetSize == 1 ? 1 : 0;
          if (peerBlock.blockSetSize > 1 && !laneId) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunk)};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == peerBlock.blockSetSize;
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
                signalOne(signals[peerBlock.peer], flag);
              }
            }
            __syncwarp();
          }
        }
        dstP += CollConfig::CHUNK_SIZE;
        srcP += CollConfig::CHUNK_SIZE;
      }
      if (bytes > cutoff) {
        const auto residue = bytes - cutoff;
        const auto [bytesPutLeft, putStartOffsetLeft] = partition<alignmentBytes>
        (residue, peerBlock.blockSetSize, peerBlock.intraIdx);
        srcP = src + ((CollConfig::CHUNK_SIZE * chunks + putStartOffsetLeft) + intraOffset);
        dstP = dstBase + (CollConfig::CHUNK_SIZE * chunks + putStartOffsetLeft);
        SutureAtom::put(dstP, srcP, bytesPutLeft, workspace);
        __syncthreads();
        flag++;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = peerBlock.blockSetSize == 1 ? 1 : 0;
          if (peerBlock.blockSetSize > 1 && !laneId) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunks)};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == peerBlock.blockSetSize;
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
                signalOne(signals[peerBlock.peer], flag);
              }
            }
            __syncwarp();
          }
        }
      }
      const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
      markEpoch(ctx, bIdx, flag);
      markUnusedEpochs<SutureAtom, CollConfig::PUT_BLOCKS>(ctx, collBlocks, flag, tid);
      return;
    }
    // consumers
    const auto cBIdx = bIdx - CollConfig::PUT_BLOCKS;
    gatherConsumer<SutureAtom, CollConfig, outputLayout>(
      dst,
      bytes,
      workspace,
      ctx,
      epochState,
      bIdx,
      cBIdx,
      static_cast<int>(blocks - CollConfig::PUT_BLOCKS),
      ctx.signals[ctx.rank],
      epochState.trStagingPrefix
    );
  }
}
#endif //SUTURE_GATHER_CUH
