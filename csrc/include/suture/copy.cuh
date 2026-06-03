//
// Created by osay on 3/27/26.
//

#ifndef SUTURE_COPY_CUH
#define SUTURE_COPY_CUH
#include <cuda/ptx>
namespace suture {
  template<typename Element>
  __device__ __forceinline__
  auto load(const Element* __restrict__ const& src) {
    if constexpr (alignof(Element) > 16) {
      static_assert(sizeof(Element) == alignof(Element));
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
      static_assert(sizeof(Element) == alignof(Element));
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
      static_assert(sizeof(Element) == alignof(Element));
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
