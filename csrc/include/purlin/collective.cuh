#ifndef PURLIN_COLLECTIVE_CUH
#define PURLIN_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"
#include "snac.cuh"

namespace purlin {
  template<typename PurlinAtom, typename CollConfig, typename Element, ReduceOp ro = ReduceOp::add, typename BT = int>
  __device__ __forceinline__
  static void reduceScatter(const SnacArgs<BT>& args, const Context& ctx) {
    SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scattered, DataLayout::packed, ro>::template run<Element>(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename Element, ReduceOp ro = ReduceOp::add, typename BT = int>
  __device__ __forceinline__
  static void reduceScatterV(const SnacArgs<BT>& args, const Context& ctx) {
    SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scatteredV, DataLayout::packedV, ro>::template run<Element>(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void allGather(const SnacArgs<BT>& args, const Context& ctx) {
    SNAC<PurlinAtom, CollConfig, ConsumeOp::copy, DataLayout::packed, DataLayout::scattered>::run(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void allGatherV(const SnacArgs<BT>& args, const Context& ctx) {
    SNAC<PurlinAtom, CollConfig, ConsumeOp::copy, DataLayout::packedV, DataLayout::scatteredV>::run(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void all2all(const SnacArgs<BT>& args, const Context& ctx) {
    SNAC<PurlinAtom, CollConfig, ConsumeOp::copy, DataLayout::scattered, DataLayout::transposed>::run(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void all2allV(const SnacArgs<BT>& args, const Context& ctx) {
    static_assert(CollConfig::PER_STREAM_THRESHOLD > 0);
    SNAC<PurlinAtom, CollConfig, ConsumeOp::copy, DataLayout::scatteredV, DataLayout::transposedV>::run(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, AllReducePath path,
    ReduceOp ro = ReduceOp::add, typename Element, typename BT = int>
  __device__ __forceinline__
  static void allReduce(const SnacArgs<BT>& args, const Context& ctx) {
    using Direct = SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed, ro>;
    using ReduceScatter = SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scattered, DataLayout::packed, ro>;
    using AllGather = SNAC<PurlinAtom, CollConfig, ConsumeOp::copy, DataLayout::packed, DataLayout::scattered, ro>;
    if constexpr (path == AllReducePath::direct) {
      Direct::template run<Element>(args, ctx);
    }
    else {
      Compose<ReduceScatter, AllGather>::template run<Element>(args, ctx);
    }
  }
}
#endif // PURLIN_COLLECTIVE_CUH
