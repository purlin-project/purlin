//
// Created by Osayamen on 4/16/26.
//

#ifndef PURLIN_COLLECTIVE_CUH
#define PURLIN_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"
#include "epoch.cuh"
#include "partition.cuh"
#include "snac.cuh"

namespace purlin {
  // Every collective is a naming of one SNAC: a consume op and a layout pair.
  // The config type selects the regime — CollectiveConfigLR resolves to the
  // fused latency specialization, a throughput config to the staged protocol.
  // allReduce composes two SNACs; all2allV chooses its SNAC from exchanged
  // footprints. Everything mechanical lives in snac.cuh.
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
    static_assert(PurlinAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    const auto epochState = makeEpochState(ctx, bIdx);
    SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scattered, DataLayout::packed>::run
    (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState, blocks);
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
    static_assert(PurlinAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    const auto bytes = sizes[ctx.rank];
    const auto epochState = makeEpochState(ctx, bIdx);
    SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scatteredV, DataLayout::packed>::run
    (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState, blocks, sizes);
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
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::packed, DataLayout::packed>::run
    (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
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
    static_assert(PurlinAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    const auto bytes = sizes[ctx.rank];
    const auto epochState = makeEpochState(ctx, bIdx);
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::packedV, DataLayout::packedV>::run
    (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks, sizes);
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
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::scattered, DataLayout::transposed>::run
    (dst, src, bytes, workspace, ctx, blocks, bIdx, epochState, blocks);
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
    static_assert(CollConfig::LATENCY_THRESHOLD > 0);
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    const auto epochState = makeEpochState(ctx, bIdx);
    auto* __restrict__ maxSize = reinterpret_cast<unsigned long long*>(workspace);
    auto* __restrict__ maxFootprint = maxSize + 1;
    if (threadIdx.x == 0) {
      *maxSize = 0;
      *maxFootprint = 0;
    }
    __syncthreads();
    const auto sigPrefix = (epochState.epoch % 2) * ctx.world;
    // The splits are rank-local, so both regime decisions must be made from
    // exchanged values. The packet carries this rank's staged-input footprint,
    // with the top bit flagging a split above the latency threshold; separate
    // maxima recover each decision exactly.
    constexpr auto EXCEEDS_LATENCY = 1ull << 63;
    const auto payload = static_cast<unsigned long long>(ctx.vState.totalBytes) |
      (ctx.vState.maxBytes > CollConfig::LATENCY_THRESHOLD ? EXCEEDS_LATENCY : 0ull);
    if (blockIdx.x == 0) {
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
        auto* __restrict__ varSigs = ctx.varLenSignals[i] + (sigPrefix + ctx.rank);
        varSigs->write(payload, epochState.nextEpoch);
      }
    }
    auto* __restrict__ vSigs = ctx.varLenSignals[ctx.rank] + sigPrefix;
    for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
      const auto currentPacket = vSigs[i].wait(epochState.nextEpoch);
      atomicMax_block(maxSize, currentPacket.data);
      atomicMax_block(maxFootprint, currentPacket.data & ~EXCEEDS_LATENCY);
    }
    __syncthreads();
    const auto globalMaxSize = *maxSize;
    const auto globalMaxFootprint = *maxFootprint;
    __syncthreads(); // downstream paths reuse the workspace holding the maxima
    if (!(globalMaxSize & EXCEEDS_LATENCY)) {
      SNAC<PurlinAtom, CollectiveConfigLR, ConsumeOp::gather, DataLayout::scatteredV, DataLayout::transposedV>::run
      (dst, src, ctx.vState.maxBytes, workspace, ctx, blocks, bIdx, epochState, blocks, outSplits, inSplits);
      return;
    }
    if (globalMaxFootprint > ctx.stagingTRSize) {
      using ringConfig = CollectiveConfig<
          CollectiveType::chunked,
          CollConfig::PUT_BLOCKS,
          CollConfig::GATHER_BLOCKS,
          CollConfig::CHUNK_SIZE,
          CollConfig::LOCAL_PUT_BLOCKS,
          CollConfig::LATENCY_THRESHOLD,
          StagingMode::ring
        >;
      SNAC<PurlinAtom, ringConfig, ConsumeOp::gather, DataLayout::scatteredV, DataLayout::transposedV>::run
      (dst, src, globalMaxFootprint, workspace, ctx, blocks, bIdx, epochState, blocks, outSplits, inSplits);
      return;
    }
    using chunkedConfig = CollectiveConfig<
        CollectiveType::chunked,
        CollConfig::PUT_BLOCKS,
        CollConfig::GATHER_BLOCKS,
        CollConfig::CHUNK_SIZE,
        CollConfig::LOCAL_PUT_BLOCKS,
        CollConfig::LATENCY_THRESHOLD
      >;
    SNAC<PurlinAtom, chunkedConfig, ConsumeOp::gather, DataLayout::scatteredV, DataLayout::transposedV>::run
    (dst, src, globalMaxFootprint, workspace, ctx, blocks, bIdx, epochState, blocks, outSplits, inSplits);
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
    // The direct (world-2) form reduces the full buffer per rank; the multimem
    // datapath is defined for the reduce-scatter-into-staging form only.
    static_assert(PurlinAtom::BaseConfig::DATAPATH == Datapath::unicast,
      "the multimem datapath is defined for the reduce-scatter-into-staging form only");
    SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed>::run
    (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState, blocks);
  }

  // allReduce is SNAC composed with SNAC: a reduce whose scattered result lands
  // back in staging (re-notifying through the gather signals), then a gather
  // that drains the reduced shards.
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
    // Under ring staging the shard regions are fixed windows rather than
    // localBytes-sized slices; the reduced result lands in the local window.
    constexpr auto ring = CollConfig::STAGING_MODE == StagingMode::ring;
    const auto shardStagingOffset = ring ?
      static_cast<size_t>(static_cast<int>(ctx.ringSlots)) * CollConfig::CHUNK_SIZE *
        static_cast<size_t>(ctx.rank) :
      localBytes * ctx.rank;
    if (bIdx < reduceScatterBlocks) {
      auto* __restrict__ sDst = ctx.staging[ctx.rank] + (stagingPrefix + shardStagingOffset);
      SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scattered, DataLayout::scattered>::run
      (sDst, src, localBytes, typedWorkspace, ctx, reduceScatterBlocks, bIdx, epochState, blocks);
      return;
    }
    const auto gBIdx = bIdx - reduceScatterBlocks;
    // uneven split: worlds that do not divide the gather-block count would
    // otherwise map trailing blocks to a nonexistent peer
    const auto peerBlock = mapPeerBlockUneven(static_cast<int>(gBIdx),
      CollConfig::GATHER_BLOCKS, ctx.world);
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::packed, DataLayout::scattered>::consume(
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
    bool partitioned = false,
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
      SNAC<PurlinAtom, CollectiveConfigLR, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed>::
      template run<partitioned>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState, blocks);
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
