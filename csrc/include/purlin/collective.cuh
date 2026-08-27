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
  // all2allV routes per stream inside its one SNAC. Everything mechanical
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
    SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scatteredV, DataLayout::packedV, ro>::
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

  template<typename PurlinAtom, typename CollConfig, World2Bypass wb = World2Bypass::unknown,
    LRMode mode = LRMode::fullBuffer, ReduceOp ro = ReduceOp::add, typename Element, typename BT = int>
  __device__ __forceinline__
  static void allReduce(const SnacArgs<BT>& args, const Context& ctx) {
    if constexpr (regimeOf<CollConfig> == Regime::latency) {
      SNAC<PurlinAtom, CollectiveConfigLR, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed, ro>::
      template run<Element, mode>(args, ctx);
    }
    else if constexpr (wb == World2Bypass::yes) {
      // direct allReduce
      static_assert(PurlinAtom::BaseConfig::MEMTYPE == MemType::unicast);
      SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed, ro>::
      template run<Element>(args, ctx);
    }
    else if constexpr (wb == World2Bypass::no) {
      ReduceGatherSNAC<PurlinAtom, CollConfig, ro>::template run<Element>(args, ctx);
    }
    else {
      if (ctx.world == 2) {
        SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed, ro>::
        template run<Element>(args, ctx);
      }
      else {
        ReduceGatherSNAC<PurlinAtom, CollConfig, ro>::template run<Element>(args, ctx);
      }
    }
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void all2allV(const SnacArgs<BT>& args, const Context& ctx) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    static_assert(CollConfig::PER_STREAM_THRESHOLD > 0, "all2allV runs the per-stream protocol only");
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::scatteredV, DataLayout::transposedV>::run(args, ctx);
  }
}
#endif //PURLIN_COLLECTIVE_CUH
