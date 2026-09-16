#ifndef PURLIN_COLLECTIVE_CUH
#define PURLIN_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"
#include "snac.cuh"

namespace purlin {
  // Each wrapper below describes a collective by choosing a consume operation
  // and an input-to-output layout transformation.
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
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::scattered, DataLayout::transposed>::run(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, typename BT = int>
  __device__ __forceinline__
  static void all2allV(const SnacArgs<BT>& args, const Context& ctx) {
    static_assert(residencyOf<CollConfig> == Staging::zero || CollConfig::PER_STREAM_THRESHOLD > 0,
      "staged all2allV runs the per-stream protocol only");
    SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::scatteredV, DataLayout::transposedV>::run(args, ctx);
  }

  template<typename PurlinAtom, typename CollConfig, AllReducePath path,
    ReduceOp ro = ReduceOp::add, typename Element, typename BT = int>
  __device__ __forceinline__
  static void allReduce(const SnacArgs<BT>& args, const Context& ctx) {
    // The direct path reduces the whole payload on every rank, this is faster for 2 ranks.
    // The composed path is reduceScatter followed by allGather.
    using Direct = SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::packed, DataLayout::packed, ro>;
    using ReduceScatter = SNAC<PurlinAtom, CollConfig, ConsumeOp::reduce, DataLayout::scattered, DataLayout::packed, ro>;
    using AllGather = SNAC<PurlinAtom, CollConfig, ConsumeOp::gather, DataLayout::packed, DataLayout::scattered, ro>;
    if constexpr (path == AllReducePath::direct) {
      Direct::template run<Element>(args, ctx);
    }
    else {
      Compose<ReduceScatter, AllGather>::template run<Element>(args, ctx);
    }
  }
}
#endif // PURLIN_COLLECTIVE_CUH
