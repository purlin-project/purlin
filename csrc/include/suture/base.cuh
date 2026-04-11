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

  enum StageStatus: uint32_t {
    empty = 0U,
    full = 1U
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
    typename Cfg,
    typename R2GOp,
    typename AlignedElement,
    typename Index_,
    int unrollFactor = Cfg::UNROLL_FACTOR
  >
  struct PeerOpConfig {
    static constexpr int THREADS = Cfg::THREADS;
    static constexpr int UNROLL_FACTOR = unrollFactor;
    static constexpr int ALIGNMENT_BYTES = Cfg::ALIGNMENT_BYTES;
    static constexpr int VECTOR_WIDTH = ALIGNMENT_BYTES / sizeof(AlignedElement);
    using Operation = R2GOp;
    using Element = AlignedElement;
    using IndexType = Index_;
  };

  template<typename Config>
  __device__ __forceinline__
  void peerOp(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes,
    const uint32_t tIdx = threadIdx.x) {
    using VT = cutlass::AlignedArray<typename Config::Element, Config::VECTOR_WIDTH>;
    using IndexT = Config::IndexType;
    const auto vP = static_cast<IndexT>(bytes / Config::ALIGNMENT_BYTES);
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const auto threadElems = vP / Config::THREADS;
    const auto trips = threadElems / Config::UNROLL_FACTOR;
    typename Config::Operation op{};
    for (int i = 0; i < trips; ++i) {
      VT reginald[Config::UNROLL_FACTOR];
      IndexT indices[Config::UNROLL_FACTOR];
      // gmem -> rmem
      cuda::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        indices[j] = (i * Config::UNROLL_FACTOR + j) * Config::THREADS + tIdx;
        reginald[j] = vS[indices[j]];
      });
      // rmem -> gmem operation
      cuda::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        op(vD + indices[j], reginald[j]);
      });
    }
    const auto residue = vP - trips * Config::UNROLL_FACTOR * Config::THREADS;
    vS += (trips * Config::UNROLL_FACTOR * Config::THREADS);
    vD += (trips * Config::UNROLL_FACTOR * Config::THREADS);
    for (int i = static_cast<int>(tIdx); i < residue; i += Config::THREADS) {
      const auto v = vS[i];
      op(vD + i, v);
    }
  }
}
#endif //SUTURE_BASE_CUH