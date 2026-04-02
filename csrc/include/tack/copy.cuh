//
// Created by azureuser on 3/27/26.
//

#ifndef TACK_COPY_CUH
#define TACK_COPY_CUH
#include <cuda/ptx>
#include <cute/arch/copy_sm80.hpp>
namespace tack {
  template <int Size>
  __device__ __forceinline__
  void cp_async_global_to_shared(void* __restrict__ const& smem_ptr, const void* __restrict__ const& gmem_ptr) {
    static_assert(Size == 4 || Size == 8 || Size == 16, "cp.async only supports Size in {4, 8, 16}");
    uint32_t sp = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
      :
      : "r"(sp), "l"(gmem_ptr), "n"(Size)
    );
  }

  template<typename Element>
  __device__ __forceinline__
  auto load(const Element* __restrict__ const& src) {
    if constexpr (MAX_ACCESS_ALIGNMENT > 16) {
      return cuda::ptx::ld(cuda::ptx::space_global, src);
    }
    else {
      return *src;
    }
  }
  template<typename Element>
  __device__ __forceinline__
  void store(Element* __restrict__ const& dst, const Element& v) {
    if constexpr (MAX_ACCESS_ALIGNMENT > 16) {
      cuda::ptx::st(cuda::ptx::space_global, dst, v);
    }
    else {
      *dst = v;
    }
  }
  template<typename Element>
  __device__ __forceinline__
  void copy(Element* __restrict__ const& dst, const Element* __restrict__ const& src) {
    if constexpr (MAX_ACCESS_ALIGNMENT > 16) {
      const auto v = cuda::ptx::ld(cuda::ptx::space_global, src);
      cuda::ptx::st(cuda::ptx::space_global, dst, v);
    }
    else {
      *dst = *src;
    }
  }
}
#endif //TACK_COPY_CUH