//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_BASE_CUH
#define SUTURE_BASE_CUH
#include <cuda/utility>
#include <cutlass/array.h>

namespace suture {
  template<int Arch>
  consteval auto normalizeArch() {
    if constexpr (Arch >= 1000) {
      return 1000;
    }
    if constexpr (Arch >= 900) {
      return 900;
    }
    if constexpr (Arch >= 800) {
      return 800;
    }
    return 700; // base
  }

  enum class StateSpace {
    GMEM,
    SMEM, // TODO: RMEM, TMEM
  };

  template<int AlignmentBytes>
  requires(cuda::is_power_of_two(AlignmentBytes))
  struct AlignedType {
    using type = uint32_t;
  };

  template<>
  struct AlignedType<1> {
    using type = cuda::std::byte;
  };

  template<>
  struct AlignedType<2> {
    using type = uint16_t;
  };
}

namespace suture::fascia {
  template<
    int threads,
    int unrollFactor,
    int AlignmentBytes,
    typename R2GOp,
    typename AlignedElement,
    typename Index = size_t
  >
  __device__ __forceinline__
  void peerOp(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes) {
    constexpr int VectorWidth = AlignmentBytes / sizeof(AlignedElement);
    using VT = cutlass::AlignedArray<AlignedElement, VectorWidth>;
    const int vP = static_cast<int>(bytes / AlignmentBytes);
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    // use unrolled direct loads as pipelining is not necessary
    const auto threadElems = vP / threads;
    const auto trips = threadElems / unrollFactor;
    R2GOp op{};
    for (int i = 0; i < trips; ++i) {
      VT reginald[unrollFactor];
      Index indices[unrollFactor];
      // gmem -> rmem
      cuda::static_for<unrollFactor>([&](auto j) {
        indices[j] = (i * unrollFactor + j) * threads + threadIdx.x;
        reginald[j] = vS[indices[j]];
      });
      // rmem -> gmem operation
      cuda::static_for<unrollFactor>([&](auto j) {
        op(vD + indices[j], reginald[j]);
      });
    }
    const auto residue = vP - trips * unrollFactor * threads;
    vS += (trips * unrollFactor * threads);
    vD += (trips * unrollFactor * threads);
    for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
      const auto v = vS[i];
      op(vD + i, v);
    }
  }
}
#endif //SUTURE_BASE_CUH