//
// Created by Osayamen on 1/8/26.
//

#ifndef tack_DESCRIPTOR_CUH
#define tack_DESCRIPTOR_CUH
#include <cutlass/array.h>
#include <cutlass/fast_math.h>
#include <cute/numeric/integral_constant.hpp>
namespace tack {
#if (__CUDA_ARCH__ >= 1000) && (defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 12) && (defined(__CUDACC_VER_MINOR__) && __CUDACC_VER_MINOR__ >= 9)
  constexpr int MAX_ALIGNMENT_BYTES = 32;
#else
  constexpr int MAX_ALIGNMENT_BYTES = 16;
#endif
  constexpr int DYNAMIC_VALUE = -1;

  enum class Regime {
    latency,
    throughput
  };

  enum class StateSpace {
    RMEM,
    SMEM,
    GMEM
  };
  // put, get
  template<
    int Architecture,
    StateSpace localSpace,
    int threads = 128,
    long int size = DYNAMIC_VALUE,
    Regime regime = Regime::latency
  >
  struct TransferDescriptor {};
  enum class SrcPolicy : uint8_t {
    Default,     // normal caching
    Stream,      // avoid L1 pollution (prefer L2-only / streaming)
    Persistent,
    ReadMostly   // read-only / nc path (caller guarantees immutability)
  };

  enum class DstPolicy : uint8_t {
    Default,     // normal write-back
    Stream,      // best-effort: reduce cache pollution for streaming output
    ConsumeSoon  // best-effort: prefetch dst for imminent reads (local dst only)
  };

  template<int Architecture, SrcPolicy P>
  struct LD {};
  template<int Architecture, DstPolicy P>
  struct ST {};

  enum class RegisterLayout {
    blocked,
    striped
  };

  template<typename T, int Alignment = MAX_ALIGNMENT_BYTES>
    struct VectorTypeDescriptor {
    static_assert(Alignment % sizeof(T) == 0 && Alignment / sizeof(T) >= 1);
    using VectorWidth = std::integral_constant<int, Alignment / sizeof(T)>;
    using VectorType = cutlass::AlignedArray<T, VectorWidth::value>;
  };
  template<typename Element, int dim>
  constexpr int ElementWidth = cute::min(dim, MAX_ALIGNMENT_BYTES / sizeof(Element));
  template<typename Element, int dim>
  constexpr int ElementAlignment = (cutlass::is_pow2<ElementWidth<Element, dim>>::value ?
      ElementWidth<Element, dim> : 1) * sizeof(Element);
}
#endif //tack_DESCRIPTOR_CUH