//
// Created by Osayamen on 4/16/26.
//

#ifndef PURLIN_COLLECTIVE_CUH
#define PURLIN_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"
#include "epoch.cuh"
#include "gather.cuh"
#include "reduce.cuh"

namespace purlin::RS {
  __device__ __host__ __forceinline__
   constexpr auto getRegime(const size_t& bytesPerRank, const int& world) {
    if (world == 8) {
      if (bytesPerRank <= 64 * 1024) {
        return Regime::latency;
      }
      return Regime::throughput;
    }
    if (bytesPerRank <= RED_LATENCY_BOUND_THRESHOLD) {
      return Regime::latency;
    }
    return Regime::throughput;
  }
}

namespace purlin::A2A {
  __host__ __device__ __forceinline__
  constexpr Regime getRegime(const size_t& bytesPerRank, const int& world) {
    switch (world) {
      case 4: {
        if (bytesPerRank <= 256 * 1024) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
        break;
      case 8: {
        if (bytesPerRank <= 4 * 1024) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
        break;
      default: {
        if (bytesPerRank <= RED_LATENCY_BOUND_THRESHOLD) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
    }
  }
}

namespace purlin {
  template<
    typename PurlinAtom,
    typename CollConfig,
    typename Element,
    typename BT = int
  >
  __device__ __forceinline__
  static void reduceScatter(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    const auto epochState = makeEpochState(ctx, bIdx);
    static_assert(PurlinAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    if constexpr (PurlinAtom::REGIME == Regime::latency) {
      reduceLR<PurlinAtom, DataLayout::scattered>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        reduceNonChunked<PurlinAtom, CollConfig::PUT_BLOCKS, DataLayout::scattered, DataLayout::packed>
          (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch,
            epochState.trStagingPrefix, blocks);
      }
      else {
        reduceChunked<PurlinAtom, CollConfig::PUT_BLOCKS, CollConfig::CHUNK_SIZE,
        DataLayout::scattered, DataLayout::packed>
        (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.epoch,
          epochState.trStagingPrefix, blocks);
      }
    }
  }

  template<
    typename PurlinAtom,
    typename CollConfig,
    typename Element,
    typename BT = int
  >
  __device__ __forceinline__
  static void reduceScatterV(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t* __restrict__ const& sizes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    const auto bytes = sizes[ctx.rank];
    const auto epochState = makeEpochState(ctx, bIdx);
    static_assert(PurlinAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    if constexpr (PurlinAtom::REGIME == Regime::latency) {
      reduceLR<PurlinAtom, DataLayout::scatteredV>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit, sizes);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        reduceNonChunked<PurlinAtom, CollConfig::PUT_BLOCKS, DataLayout::scatteredV, DataLayout::packed>
          (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch,
            epochState.trStagingPrefix, blocks);
      }
      else {
        reduceChunked<PurlinAtom, CollConfig::PUT_BLOCKS, CollConfig::CHUNK_SIZE,
        DataLayout::scatteredV, DataLayout::packed>
        (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.epoch,
          epochState.trStagingPrefix, blocks, sizes);
      }
    }
  }

  template<
    typename PurlinAtom,
    typename CollConfig,
    typename BT = int
  >
  __device__ __forceinline__
  static void allGather(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    const auto epochState = makeEpochState(ctx, bIdx);

    if constexpr (PurlinAtom::REGIME == Regime::latency) {
      gatherLR<PurlinAtom, DataLayout::packed>
      (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        gatherNonChunked<PurlinAtom, CollConfig, DataLayout::packed, DataLayout::packed>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
      }
      else {
        static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
        gatherChunked<PurlinAtom, CollConfig, DataLayout::packed, DataLayout::packed>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
      }
    }
  }

  template<
    typename PurlinAtom,
    typename CollConfig,
    typename BT = int
  >
  __device__ __forceinline__
  static void allGatherV(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t* __restrict__ const& sizes,
    cuda::std::byte* __restrict__ const& workspace,
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    const auto bytes = sizes[ctx.rank];
    const auto epochState = makeEpochState(ctx, bIdx);
    static_assert(PurlinAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    if constexpr (PurlinAtom::REGIME == Regime::latency) {
      gatherLR<PurlinAtom, DataLayout::packedV>
      (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit, sizes);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        gatherNonChunked<PurlinAtom, CollConfig, DataLayout::packedV, DataLayout::packedV>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks, sizes);
      }
      else {
        static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
        gatherChunked<PurlinAtom, CollConfig, DataLayout::packedV, DataLayout::packedV>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks, sizes);
      }
    }
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void all2all(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    const auto epochState = makeEpochState(ctx, bIdx);

    if constexpr (PurlinAtom::REGIME == Regime::latency) {
      gatherLR<PurlinAtom, DataLayout::scattered>
      (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        gatherNonChunked<PurlinAtom, CollConfig, DataLayout::scattered, DataLayout::transposed>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
      }
      else {
        static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
        gatherChunked<PurlinAtom, CollConfig, DataLayout::scattered, DataLayout::transposed>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
      }
    }
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void all2allV(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t* __restrict__ const& inSplits,
    const size_t* __restrict__ const& outSplits,
    cuda::std::byte* __restrict__ const& workspace,
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    const auto epochState = makeEpochState(ctx, bIdx);
    auto* __restrict__ maxSize = reinterpret_cast<unsigned long long*>(workspace);
    if (threadIdx.x == 0) {
      *maxSize = 0;
    }
    __syncthreads();
    const auto sigPrefix = (epochState.epoch % 2) * ctx.world;
    if (blockIdx.x == 0) {
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
        auto* __restrict__ varSigs = ctx.varLenSignals[i] + (sigPrefix + ctx.rank);
        const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> sig{*varSigs};
        LRP16 lrp{};
        lrp.pack(ctx.vState.maxBytes, epochState.nextEpoch);
        sig.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
      }
    }
    auto* __restrict__ vSigs = ctx.varLenSignals[ctx.rank] + sigPrefix;
    for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
      const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> sig{*(vSigs + i)};
      auto currentPacket = cuda::std::bit_cast<LRP16>(sig.load(cuda::memory_order_relaxed));
      auto hPA = currentPacket.flag == epochState.nextEpoch;
      while (!hPA) {
        currentPacket = cuda::std::bit_cast<LRP16>(sig.load(cuda::memory_order_relaxed));
        hPA = currentPacket.flag == epochState.nextEpoch;
      }
      atomicMax_block(maxSize, currentPacket.data);
    }
    __syncthreads();
    const auto globalMaxSize = *maxSize;
    if (A2A::getRegime(globalMaxSize, ctx.world) == Regime::latency) {
      gatherLR<PurlinAtom, DataLayout::scatteredV>(dst, src, ctx.vState.maxBytes, workspace, ctx, blocks, bIdx, epochState.nextEpoch,
        epochState.senseBit, outSplits, inSplits);
      return;
    }
    using chunkedConfig = CollectiveConfig<
        CollectiveType::chunked,
        CollConfig::PUT_BLOCKS,
        CollConfig::GATHER_BLOCKS,
        CollConfig::CHUNK_SIZE,
        CollConfig::LOCAL_PUT_BLOCKS
      >;
    gatherChunked<PurlinAtom, chunkedConfig, DataLayout::scatteredV, DataLayout::transposedV>
    (dst, src, globalMaxSize, workspace, ctx, blocks, bIdx, epochState, blocks, outSplits, inSplits);
  }

  template<typename PurlinAtom, typename CollConfig, typename Element, typename BT>
  __device__ __forceinline__
  static void allReduceDirect(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace,
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const EpochState& epochState) {
    const auto stagingPrefix = epochState.trStagingPrefix;
    if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
      reduceNonChunked<
        PurlinAtom,
        CollConfig::PUT_BLOCKS,
        DataLayout::packed,
        DataLayout::packed
      >
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, stagingPrefix, blocks);
    }
    else {
      reduceChunked<
        PurlinAtom,
        CollConfig::PUT_BLOCKS,
        CollConfig::CHUNK_SIZE,
        DataLayout::packed,
        DataLayout::packed
      >
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.epoch, stagingPrefix, blocks);
    }
  }

  template<typename PurlinAtom, typename CollConfig, typename Element, typename BT>
  __device__ __forceinline__
  static void allReduceReduceScatterAllGather(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace,
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const EpochState& epochState) {
    const auto stagingPrefix = epochState.trStagingPrefix;
    const auto localBytes = bytes / ctx.world_l;
    const auto reduceScatterBlocks = blocks - CollConfig::GATHER_BLOCKS;
    if (bIdx < reduceScatterBlocks) {
      auto* __restrict__ sDst = ctx.staging[ctx.rank] + (stagingPrefix + localBytes * ctx.rank);
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        reduceNonChunked<
          PurlinAtom,
          CollConfig::PUT_BLOCKS,
          DataLayout::scattered,
          DataLayout::scattered
        >
        (sDst, src, localBytes, typedWorkspace, ctx, reduceScatterBlocks, bIdx,
          epochState.nextEpoch, stagingPrefix, blocks);
      }
      else {
        reduceChunked<
          PurlinAtom,
          CollConfig::PUT_BLOCKS,
          CollConfig::CHUNK_SIZE,
          DataLayout::scattered,
          DataLayout::scattered
        >
        (sDst, src, localBytes, typedWorkspace, ctx, reduceScatterBlocks, bIdx,
          epochState.epoch, stagingPrefix, blocks);
      }
      return;
    }
    const auto gBIdx = bIdx - reduceScatterBlocks;
    const auto blockSetSize = CollConfig::GATHER_BLOCKS / ctx.world;
    const auto peerBlock = mapPeerBlock(gBIdx, blockSetSize);
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    gatherConsumer<PurlinAtom, CollConfig, DataLayout::scattered>(
      dst + (localBytes * peerBlock.peer),
      localBytes,
      workspace,
      ctx,
      epochState,
      bIdx,
      peerBlock,
      ctx.gatherSignals[ctx.rank],
      stagingPrefix
    );
  }

  template<
    typename PurlinAtom,
    typename CollConfig,
    World2Bypass wb = World2Bypass::unknown,
    typename Element,
    typename BT = int
  >
  __device__ __forceinline__
  static void allReduce(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    const auto epochState = makeEpochState(ctx, bIdx);
    static_assert(PurlinAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    if constexpr (PurlinAtom::REGIME == Regime::latency) {
      reduceLR<PurlinAtom, DataLayout::packed>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else if constexpr (wb == World2Bypass::yes) {
      allReduceDirect<PurlinAtom, CollConfig>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState);
    }
    else if constexpr (wb == World2Bypass::no) {
      allReduceReduceScatterAllGather<PurlinAtom, CollConfig>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState);
    }
    else {
      if (ctx.world == 2) {
        allReduceDirect<PurlinAtom, CollConfig>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState);
      }
      else {
        allReduceReduceScatterAllGather<PurlinAtom, CollConfig>
        (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState);
      }
    }
  }
}
#endif //PURLIN_COLLECTIVE_CUH
