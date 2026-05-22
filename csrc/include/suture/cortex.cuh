//
// Created by Osaya on 4/7/26.
//

#ifndef SUTURE_CORTEX_CUH
#define SUTURE_CORTEX_CUH

#include "base.cuh"
template<>
  struct suture::InplaceSum<float2, 1000> {
  __device__ __forceinline__
  void operator()(float2& lhs, const float2& rhs) const {
    lhs = __fadd2_rn(lhs, rhs);
  }
};

template<typename Config_>
struct suture::Atom<1000, Config_> {
  using BaseConfig = Config_;
  using Config = Config_;
  using BaseAtom = Atom<900, Config_>;
  static constexpr int COLL_STATE_BYTES = BaseAtom::COLL_STATE_BYTES;
  static constexpr int COPY_PIPELINE_BYTES = BaseAtom::COPY_PIPELINE_BYTES;
  static constexpr int RED_PIPELINE_BYTES = BaseAtom::RED_PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_SMEM_BYTES = BaseAtom::COPY_PIPELINE_SMEM_BYTES;
  static constexpr int RED_PIPELINE_SMEM_BYTES = BaseAtom::RED_PIPELINE_SMEM_BYTES;
  static constexpr int RED_SMEM_SIZE = BaseAtom::RED_SMEM_SIZE;
  static constexpr int COPY_SMEM_SIZE = BaseAtom::COPY_SMEM_SIZE;
  static constexpr int THREADS = Config::THREADS;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config_::GMEM_ACCESS_ALIGNMENT_BYTES;
  __device__ __forceinline__
  static void putAsync(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    BaseAtom::putAsync(dst, src, bytes, workspace);
  }
  __device__ __forceinline__
  static void put(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    BaseAtom::put(dst, src, bytes, workspace);
  }

  // latency-regime
  template<DataLayout iLayout, typename Element>
  __device__ __forceinline__
  static void reduce(const LRArgs& redArgs, Element* __restrict__ const&) {
    using RedOp = ArrayInplaceSum<1000>;
    fascia::reduce<Config_, RedOp, Element, iLayout>(redArgs);
  }

  template<typename RedOp = ArrayInplaceSum<1000>, typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
    BaseAtom::template reduce<RedOp>(redArgs, typedWorkspace);
  }

  __device__ __forceinline__
  static void fenceAlias() {
    BaseAtom::fenceAlias();
  }
};
#endif //SUTURE_CORTEX_CUH