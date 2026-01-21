//
// Created by azureuser on 1/8/26.
//

#ifndef TACT_SM70_CUH
#define TACT_SM70_CUH

#include <cuda/ptx>
#include <cutlass/fast_math.h>

#include "descriptor.cuh"
namespace tact {
  struct LD<70, SrcPolicy::Default> {
    template<typename Element>
    __device__ __forceinline__
    auto operator()(const Element* __restrict__ const& addr) {
      constexpr int Alignment = sizeof(Element);
      static_assert(cutlass::is_pow2<Alignment>::value && Alignment > 0 && Alignment <= MAX_ALIGNMENT_BYTES);
      return cuda::ptx::ld(addr);
    }
  };
  struct LD<70, SrcPolicy::Stream> {
    template<typename Element>
    __device__ __forceinline__
    auto operator()(const Element* __restrict__ const& addr) {
      constexpr int Alignment = sizeof(Element);
      static_assert(cutlass::is_pow2<Alignment>::value && Alignment > 0 && Alignment <= MAX_ALIGNMENT_BYTES);
      return cuda::ptx::ld_L1_no_allocate(addr);
    }
  };
  struct LD<70, SrcPolicy::Persistent> {
    template<typename Element>
    __device__ __forceinline__
    auto operator()(const Element* __restrict__ const& addr) {
      constexpr int Alignment = sizeof(Element);
      static_assert(cutlass::is_pow2<Alignment>::value && Alignment > 0 && Alignment <= MAX_ALIGNMENT_BYTES);
      return cuda::ptx::ld_L1_evict_last(addr);
    }
  };
  struct LD<70, SrcPolicy::ReadMostly> {
    template<typename Element>
    __device__ __forceinline__
    auto operator()(const Element* __restrict__ const& addr) {
      constexpr int Alignment = sizeof(Element);
      static_assert(cutlass::is_pow2<Alignment>::value && Alignment > 0 && Alignment <= MAX_ALIGNMENT_BYTES);
      return cuda::ptx::ld_nc(addr);
    }
  };

  struct ST<70, DstPolicy::Default> {
    template<typename Element>
    __device__ __forceinline__
    void operator()(Element* __restrict__ const& addr, const Element& val) {
      constexpr int Alignment = sizeof(Element);
      static_assert(cutlass::is_pow2<Alignment>::value && Alignment > 0 && Alignment <= MAX_ALIGNMENT_BYTES);
      return cuda::ptx::st(addr, val);
    }
  };
  struct ST<70, DstPolicy::ConsumeSoon> {
    template<typename Element>
    __device__ __forceinline__
    void operator()(Element* __restrict__ const& addr, const Element& val) {
      constexpr int Alignment = sizeof(Element);
      static_assert(cutlass::is_pow2<Alignment>::value && Alignment > 0 && Alignment <= MAX_ALIGNMENT_BYTES);
      return cuda::ptx::st_L1_evict_last(addr, val);
    }
  };
  struct ST<70, DstPolicy::Stream> {
    template<typename Element>
    __device__ __forceinline__
    void operator()(Element* __restrict__ const& addr, const Element& val) {
      constexpr int Alignment = sizeof(Element);
      static_assert(cutlass::is_pow2<Alignment>::value && Alignment > 0 && Alignment <= MAX_ALIGNMENT_BYTES);
      return cuda::ptx::st_L1_evict_first(addr, val);
    }
  };
  // GMEM -> GMEM
  template<
   int threads, // we could make this dynamic
   long int size,
   Regime regime
  >
  struct TransferDescriptor<70, StateSpace::GMEM, threads, size, regime> {
    template<SrcPolicy policy, typename Element>
    __device__ __forceinline__
    static void put(Element* __restrict__ const& dst, Element* __restrict__ const& src, const size_t& nElems,
      const int& pe, const int& tid) {
      auto* __restrict__ dP = nvshmem_ptr(dst, pe);
      constexpr LD<70, policy> load{};
      if constexpr (regime == Regime::throughput) {
        // assume sufficiently sized
        // TODO unroll intelligently
        if constexpr (size == DYNAMIC_VALUE) {

        }
        else {

        }
      }
      else {
        // small message sizes
        if constexpr (size == DYNAMIC_VALUE) {
          for (size_t i = tid; i < nElems; i += threads) {
            dP[i] = load(src + i);
          }
        }
        else {
          // user specified size as compile-time constant, yay!
          constexpr auto sizePerThread = size / threads;
          using VTD = VectorTypeDescriptor<Element, ElementAlignment<Element, sizePerThread>>;
          using VT = VTD::VectorType;
          constexpr auto vectorSize = sizePerThread / VTD::VectorWidth::value;
          auto* __restrict__ vectorDest = reinterpret_cast<VT*>(dP);
          const auto* __restrict__ vectorSrc = reinterpret_cast<const VT*>(src);
          // assume here that sizePerThread is not too large otherwise there will be register spilling
          #pragma unroll
          for (int i = 0; i < vectorSize; i++) {
            const auto idx = threads * i + tid;
            vectorDest[idx] = load(vectorSrc + idx);
          }
          constexpr auto residue = size - sizePerThread * threads;
          if constexpr (residue > 0) {
            if (tid < residue) {
              const auto idx = threads * residue + tid;
              vectorDest[idx] = load(vectorSrc + idx);
            }
          }
        }
      }
    }
  };
  //TODO RMEM -> GMEM
  //TODO SMEM -> GMEM
}

#endif //TACT_SM70_CUH