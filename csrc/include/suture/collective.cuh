//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_COLLECTIVE_CUH
#define SUTURE_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"
#include "epoch.cuh"
#include "gather.cuh"
#include "reduce.cuh"

namespace suture {
  template<
    typename SutureAtom,
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
    static_assert(SutureAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    if constexpr (SutureAtom::REGIME == Regime::latency) {
      reduceLR<SutureAtom, DataLayout::scattered>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        reduceNonChunked<SutureAtom, CollConfig::PUT_BLOCKS, DataLayout::scattered, DataLayout::packed>
          (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch,
            epochState.trStagingPrefix, blocks);
      }
      else {
        reduceChunked<SutureAtom, CollConfig::PUT_BLOCKS, CollConfig::CHUNK_SIZE,
        DataLayout::scattered, DataLayout::packed>
        (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.epoch,
          epochState.trStagingPrefix, blocks);
      }
    }
  }

  template<
    typename SutureAtom,
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

    if constexpr (SutureAtom::REGIME == Regime::latency) {
      gatherLR<SutureAtom, DataLayout::packed>
      (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        gatherNonChunked<SutureAtom, CollConfig, DataLayout::packed, DataLayout::packed>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
      }
      else {
        static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
        gatherChunked<SutureAtom, CollConfig, DataLayout::packed, DataLayout::packed>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
      }
    }
  }

  template<typename SutureAtom, typename CollConfig, typename BT = int>
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

    if constexpr (SutureAtom::REGIME == Regime::latency) {
      gatherLR<SutureAtom, DataLayout::scattered>
      (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        gatherNonChunked<SutureAtom, CollConfig, DataLayout::scattered, DataLayout::transposed>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
      }
      else {
        static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
        gatherChunked<SutureAtom, CollConfig, DataLayout::scattered, DataLayout::transposed>
        (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
      }
    }
  }

  template<
    typename SutureAtom,
    typename CollConfig,
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
    static_assert(SutureAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    if constexpr (SutureAtom::REGIME == Regime::latency) {
      reduceLR<SutureAtom, DataLayout::packed>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      const auto stagingPrefix = epochState.trStagingPrefix;
      if (ctx.world == 2) {
        if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
          reduceNonChunked<
            SutureAtom,
            CollConfig::PUT_BLOCKS,
            DataLayout::packed,
            DataLayout::packed
          >
          (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, stagingPrefix, blocks);
        }
        else {
          reduceChunked<
            SutureAtom,
            CollConfig::PUT_BLOCKS,
            CollConfig::CHUNK_SIZE,
            DataLayout::packed,
            DataLayout::packed
          >
          (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.epoch, stagingPrefix, blocks);
        }
        return;
      }
      const auto localBytes = bytes / ctx.world_l;
      // RS+AG
      const auto reduceScatterBlocks = blocks - CollConfig::GATHER_BLOCKS;
      if (bIdx < reduceScatterBlocks) {
        auto* __restrict__ sDst = ctx.staging[ctx.rank] + (stagingPrefix + localBytes * ctx.rank);
        if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
          reduceNonChunked<
            SutureAtom,
            CollConfig::PUT_BLOCKS,
            DataLayout::scattered,
            DataLayout::scattered
          >
          (sDst, src, localBytes, typedWorkspace, ctx, reduceScatterBlocks, bIdx,
            epochState.nextEpoch, stagingPrefix, blocks);
        }
        else {
          reduceChunked<
            SutureAtom,
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
      // gather blocks
      const auto gBIdx = bIdx - reduceScatterBlocks;
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      gatherConsumer<SutureAtom, CollConfig, DataLayout::scattered>(
        dst,
        localBytes,
        workspace,
        ctx,
        epochState,
        bIdx,
        gBIdx,
        CollConfig::GATHER_BLOCKS,
        ctx.gatherSignals[ctx.rank],
        stagingPrefix
      );
    }
  }
}
#endif //SUTURE_COLLECTIVE_CUH
