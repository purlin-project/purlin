//
// Created by azureuser on 3/27/26.
//

#ifndef SUTURE_COPY_CUH
#define SUTURE_COPY_CUH
#include <cuda/ptx>
namespace suture {
  template <int Size>
  __device__ __forceinline__
  void cpAsync(void* __restrict__ const& smem_ptr, const void* __restrict__ const& gmem_ptr) {
    static_assert(Size == 4 || Size == 8 || Size == 16, "cp.async only supports Size in {4, 8, 16}");
    uint32_t sp = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
      "cp.async.ca.shared.global [%0], [%1], %2;\n"
      :
      : "r"(sp), "l"(gmem_ptr), "n"(Size)
      : "memory"
    );
  }
  // cp.async.wait_group N: wait until at most N groups remain outstanding
  // N must be a compile-time constant — enforced via template parameter
  template<int N>
  __device__ __forceinline__
  void cpAsyncWait() {
    if constexpr (N == 0) {
      asm volatile("cp.async.wait_all;\n" ::: "memory");
    }
    else {
      static_assert(N >= 0, "cp.async.wait_group argument must be >= 0");
      asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
    }
  }

  // cp.async.commit_group: close the current group on this thread's ring
  __device__ __forceinline__
  void cpAsyncCommit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }

  template<typename Element>
  __device__ __forceinline__
  auto load(const Element* __restrict__ const& src) {
    if constexpr (alignof(Element) > 16) {
      return cuda::ptx::ld(cuda::ptx::space_global, src);
    }
    else {
      return *src;
    }
  }
  template<typename Element>
  __device__ __forceinline__
  void store(Element* __restrict__ const& dst, const Element& v) {
    if constexpr (alignof(Element) > 16) {
      cuda::ptx::st(cuda::ptx::space_global, dst, v);
    }
    else {
      *dst = v;
    }
  }
  template<typename Element>
  __device__ __forceinline__
  void copy(Element* __restrict__ const& dst, const Element* __restrict__ const& src) {
    if constexpr (alignof(Element) > 16) {
      const auto v = cuda::ptx::ld(cuda::ptx::space_global, src);
      cuda::ptx::st(cuda::ptx::space_global, dst, v);
    }
    else {
      *dst = *src;
    }
  }
  struct ST {
    template<typename Element>
    __device__ __forceinline__
    void operator()(Element* __restrict__ const& dst, const Element& v) const {
      suture::store(dst, v);
    }
  };
}
#endif //SUTURE_COPY_CUH