//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_CORTEX_CUH
#define PURLIN_CORTEX_CUH

#include "base.cuh"
template<>
  struct purlin::InplaceSum<float2, 1000> {
  __device__ __forceinline__
  void operator()(float2& lhs, const float2& rhs) const {
    lhs = __fadd2_rn(lhs, rhs);
  }
};

template<>
  struct purlin::InplaceMul<float2, 1000> {
  __device__ __forceinline__
  void operator()(float2& lhs, const float2& rhs) const {
    lhs = __fmul2_rn(lhs, rhs);
  }
};

template<typename Config_>
struct purlin::Atom<1000, Config_> {
  using BaseConfig = Config_;
  using BaseAtom = Atom<900, Config_>;
  using Config = typename BaseAtom::Config;
  static constexpr int NARCH = 1000;
  static constexpr int COPY_PIPELINE_BYTES = BaseAtom::COPY_PIPELINE_BYTES;
  static constexpr int RED_PIPELINE_BYTES = BaseAtom::RED_PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_SMEM_BYTES = BaseAtom::COPY_PIPELINE_SMEM_BYTES;
  static constexpr int RED_PIPELINE_SMEM_BYTES = BaseAtom::RED_PIPELINE_SMEM_BYTES;
  static constexpr int THREADS = BaseAtom::THREADS;
  static constexpr int WARPS = BaseAtom::WARPS;
  static constexpr int STAGE_BYTES = BaseAtom::STAGE_BYTES;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = BaseAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
  __device__ __forceinline__
  static void copy(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    BaseAtom::copy(dst, src, bytes, workspace);
  }

  template<ReduceResult result, ReduceOp ro = ReduceOp::add,
    typename RedOp = typename LoweredReduceOp<ro, NARCH>::type, typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
    if constexpr (BaseConfig::MEMTYPE == MemType::multimem) {
      static_assert(multimemReducible<NARCH, Element, ro>(),
        "the multimem datapath has no mapping for this element/op pair");
      ligament::multimemReduce<BaseConfig, Element, result, ro>(redArgs);
    }
    else {
      BaseAtom::template reduce<ReduceResult::unicast, ro, RedOp>(redArgs, typedWorkspace);
    }
  }
};
#endif //PURLIN_CORTEX_CUH
