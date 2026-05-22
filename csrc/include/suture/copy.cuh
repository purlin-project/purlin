//
// Created by osay on 3/27/26.
//

#ifndef SUTURE_COPY_CUH
#define SUTURE_COPY_CUH
#include <cuda/ptx>
namespace suture {
  template<UseMulticast u, int worldUnroll>
  struct MVSConfig {
    static constexpr UseMulticast USE_MULTICAST = u;
    static constexpr int WORLD_UNROLL = worldUnroll;
  };
  // broadcast store
  template<typename Config, typename VTP, typename VT>
  __device__ __forceinline__
  void bST(VTP* __restrict__ const& sources, const VT& v,
    const size_t& offset,
    const int& worldTrips,
    const int& cutoff,
    const int& world) {
    static_assert(cuda::std::is_pointer_v<VTP>);
    for (int t = 0; t < worldTrips; ++t) {
      cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
        const auto peer = t * Config::WORLD_UNROLL + p;
        auto* __restrict__ pD = reinterpret_cast<VT*>(sources[peer]);
        pD[offset] = v;
      });
    }
    if (world > cutoff) {
      for (int peer = cutoff; peer < world; ++peer) {
        auto* __restrict__ pD = reinterpret_cast<VT*>(sources[peer]);
        pD[offset] = v;
      }
    }
  }
  // adaptable multicast vector store
  template<typename Config, int nArch, typename Element = void>
  struct MVS {
    static_assert(Config::USE_MULTICAST == UseMulticast::no ||
      (nArch >= 900 &&
        (cuda::std::is_same_v<Element, __half2> ||
          cuda::std::is_same_v<Element, __nv_bfloat162> ||
          cuda::std::is_same_v<Element, float> ||
          cuda::std::is_same_v<Element, double>)));
    template<typename VTP, typename VT>
    __device__ __forceinline__
    void operator()(VTP* __restrict__ const& sources, const VT& v,
      const size_t& offset,
      const int& worldTrips,
      const int& cutoff,
      const int& world) const {
      bST<Config>(sources, v, offset, worldTrips, cutoff, world);
    }
  };

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
