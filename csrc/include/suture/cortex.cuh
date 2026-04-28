//
// Created by Osa on 4/7/26.
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
  using Config = Config_;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config_::GMEM_ACCESS_ALIGNMENT_BYTES;
  __device__ __forceinline__
  static void putAsync(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    Atom<900, Config>::putAsync(dst, src, bytes, workspace);
  }
  // latency-regime
  template<typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceLRArgs& redArgs, Element* __restrict__ const&) {
    using RedOp = ArrayInplaceSum<1000>;
    fascia::reduce<Config, RedOp, Element>(redArgs);
  }
};
#endif //SUTURE_CORTEX_CUH