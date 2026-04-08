//
// Created by azureuser on 4/7/26.
//

#ifndef SUTURE_LIGAMENT_CUH
#define SUTURE_LIGAMENT_CUH
#include "base.cuh"
namespace suture::ligament {
  // nArch is implicitly 900 in tendon
  constexpr int nArch = 900;
}

// GMEM (local) -> GMEM(remote)
template<>
struct suture::Atom<900, suture::StateSpace::GMEM> {
  static_assert(ligament::nArch == 900);
  using MaxAlignmentBytes = cuda::std::integral_constant<int, 16>;
  __device__ __forceinline__
  static void put(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& workspace,
    const size_t& bytes) {
    //assert(__isShared(workspace));
    // 1. if less than threshold, do LSU GMEM -> GMEM
  }
};
#endif //SUTURE_LIGAMENT_CUH