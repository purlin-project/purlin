//
// Created by osay on 3/27/26.
//

#ifndef PURLIN_COPY_CUH
#define PURLIN_COPY_CUH
#include <cuda/ptx>
namespace purlin {
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
      purlin::store(dst, v);
    }
  };
}
#endif //PURLIN_COPY_CUH
