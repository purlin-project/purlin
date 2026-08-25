//
// Created by Osayamen on 4/16/26.
//

#ifndef PURLIN_COLLECTIVE_CUH
#define PURLIN_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"
#include "snac.cuh"

namespace purlin {
  // Every collective is a naming of one SNAC: a consume op and a layout pair
  // applied to one SnacArgs. The config type selects the regime —
  // CollectiveConfigLR resolves to the fused latency specialization, a
  // throughput config to the staged protocol. allReduce composes two SNACs;
  // all2allV chooses its SNAC from exchanged footprints. Everything mechanical
  // lives in snac.cuh.
  template<typename PurlinAtom, typename CollConfig, typename Element,
    ReduceOp ro = ReduceOp::add, typename BT = int>
  __device__ __forceinline__
  static void reduceScatter(const SnacArgs<BT>& args, const Context& ctx) {
    SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scattered, DataLayout::packed, ro>::
    template run<Element>(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename Element,
    ReduceOp ro = ReduceOp::add, typename BT = int>
  __device__ __forceinline__
  static void reduceScatterV(const SnacArgs<BT>& args, const Context& ctx) {
    SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scatteredV, DataLayout::packed, ro>::
    template run<Element>(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void allGather(const SnacArgs<BT>& args, const Context& ctx) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::packed, DataLayout::scattered>::run(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void allGatherV(const SnacArgs<BT>& args, const Context& ctx) {
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::packedV, DataLayout::scatteredV>::run(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void all2all(const SnacArgs<BT>& args, const Context& ctx) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::scattered, DataLayout::transposed>::run(args, ctx);
  }

  namespace detail {
    // The all2allV regime rendezvous: the splits are rank-local, so both regime
    // decisions must be made from exchanged values. Each rank broadcasts its
    // staged-input footprint (top bit: any split above the latency threshold)
    // through the variable-length packet signals, and every block reduces the
    // exchanged maxima locally; separate maxima recover each decision exactly.
    struct A2AVRegime {
      bool exceedsLatency;
      size_t maxFootprint;
    };
    template<typename PurlinAtom, size_t latencyThreshold>
    __device__ __forceinline__
    static A2AVRegime all2allVRendezvous(cuda::std::byte *__restrict__ const&workspace,
                                         const Context &ctx,
                                         const int &bIdx) {
      const auto epochState = makeEpochState(ctx, bIdx);
      auto *__restrict__ maxSize = reinterpret_cast<unsigned long long*>(workspace);
      auto *__restrict__ maxFootprint = maxSize + 1;
      if (threadIdx.x == 0) {
        *maxSize = 0;
        *maxFootprint = 0;
      }
      __syncthreads();
      const auto sigPrefix = epochState.senseBit * ctx.world;
      constexpr auto EXCEEDS_LATENCY = 1ull << 63;
      const auto payload = static_cast<unsigned long long>(ctx.vState.totalBytes) |
        (ctx.vState.maxBytes > latencyThreshold ? EXCEEDS_LATENCY : 0ull);
      if (blockIdx.x == 0) {
        for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
          auto *__restrict__ varSigs = ctx.varLenSignals[i] + (sigPrefix + ctx.rank);
          varSigs->write(payload, epochState.nextEpoch);
        }
      }
      auto *__restrict__ vSigs = ctx.varLenSignals[ctx.rank] + sigPrefix;
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
        const auto currentPacket = vSigs[i].wait(epochState.nextEpoch);
        atomicMax_block(maxSize, currentPacket.data);
        atomicMax_block(maxFootprint, currentPacket.data & ~EXCEEDS_LATENCY);
      }
      __syncthreads();
      const auto globalMaxSize = *maxSize;
      const auto globalMaxFootprint = *maxFootprint;
      __syncthreads(); // downstream paths reuse the workspace holding the maxima
      return A2AVRegime{
        .exceedsLatency = (globalMaxSize & EXCEEDS_LATENCY) != 0ull,
        .maxFootprint = static_cast<size_t>(globalMaxFootprint)
      };
    }
    // The direct (world-2) form reduces the full buffer per rank; the multimem
    // datapath is defined for the reduce-scatter-into-staging form only.
    template<typename PurlinAtom, typename CollConfig, ReduceOp ro = ReduceOp::add,
      typename Element, typename BT>
    __device__ __forceinline__
    static void allReduceDirect(const SnacArgs<BT>& args, const Context& ctx) {
      static_assert(PurlinAtom::BaseConfig::MEMTYPE == MemType::unicast,
        "the multimem datapath is defined for the reduce-scatter-into-staging form only");
      SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed, ro>::
      template run<Element>(args, ctx);
    }
  }

  template<typename PurlinAtom, typename CollConfig, World2Bypass wb = World2Bypass::unknown,
    LRMode mode = LRMode::fullBuffer, ReduceOp ro = ReduceOp::add, typename Element, typename BT = int>
  __device__ __forceinline__
  static void allReduce(const SnacArgs<BT>& args, const Context& ctx) {
    if constexpr (regimeOf<CollConfig> == Regime::latency) {
      SNAC<PurlinAtom, CollectiveConfigLR, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed, ro>::
      template run<Element, mode>(args, ctx);
    }
    else if constexpr (wb == World2Bypass::yes) {
      detail::allReduceDirect<PurlinAtom, CollConfig, ro, Element>(args, ctx);
    }
    else if constexpr (wb == World2Bypass::no) {
      ReduceGatherSNAC<PurlinAtom, CollConfig, ro>::template run<Element>(args, ctx);
    }
    else {
      if (ctx.world == 2) {
        detail::allReduceDirect<PurlinAtom, CollConfig, ro, Element>(args, ctx);
      }
      else {
        ReduceGatherSNAC<PurlinAtom, CollConfig, ro>::template run<Element>(args, ctx);
      }
    }
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void all2allV(const SnacArgs<BT>& args, const Context& ctx) {
    static_assert(CollConfig::LATENCY_THRESHOLD > 0);
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    const auto regime = detail::all2allVRendezvous<PurlinAtom, CollConfig::LATENCY_THRESHOLD>(
      args.workspace, ctx, args.bIdx);
    // the rendezvous picked the extent; everything else rides through unchanged
    const auto withBytes = [&](const size_t& bytes) {
      return SnacArgs<BT>{
        .dst = args.dst,
        .src = args.src,
        .bytes = bytes,
        .workspace = args.workspace,
        .sizes = args.sizes,
        .inSizes = args.inSizes,
        .blocks = args.blocks,
        .collBlocks = args.collBlocks,
        .bIdx = args.bIdx,
      };
    };
    if (!regime.exceedsLatency) {
      SNAC<PurlinAtom, CollectiveConfigLR, ConsumeOp::gather, DataLayout::scatteredV, DataLayout::transposedV>::run
      (withBytes(ctx.vState.maxBytes), ctx);
      return;
    }
    if (regime.maxFootprint > ctx.stagingTRSize) {
      using cyclicConfig = CollectiveConfig<
          CollectiveType::chunked,
          CollConfig::PUT_BLOCKS,
          CollConfig::GATHER_BLOCKS,
          CollConfig::CHUNK_SIZE,
          CollConfig::LOCAL_PUT_BLOCKS,
          CollConfig::LATENCY_THRESHOLD,
          StagingMode::cyclic
        >;
      SNAC<PurlinAtom, cyclicConfig, ConsumeOp::gather, DataLayout::scatteredV, DataLayout::transposedV>::run
      (withBytes(regime.maxFootprint), ctx);
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
    (withBytes(regime.maxFootprint), ctx);
  }
}
#endif //PURLIN_COLLECTIVE_CUH
