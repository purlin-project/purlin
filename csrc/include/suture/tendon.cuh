//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_TENDON_CUH
#define SUTURE_TENDON_CUH
#include "base.cuh"
#include "copy.cuh"
namespace suture::tendon {
  // nArch is implicitly 800 in tendon
  constexpr int nArch = 800;
}

// GMEM (local) -> GMEM(remote)
template<>
struct suture::Atom<800, suture::StateSpace::GMEM> {
  static_assert(tendon::nArch == 800);
  using MaxAlignmentBytes = cuda::std::integral_constant<int, 16>;
  template<
    int threads,
    int pipeStages = 4, // tuned default
    int stageExtent = 4,
    int unrollFactor = 2,
    int AlignmentBytes = MaxAlignmentBytes::value
  >
  __device__ __forceinline__
  static void put(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& workspace,
    const size_t& bytes) {
    //assert(__isShared(workspace));
    using AT = AlignedType<AlignmentBytes>::type;
    if (bytes < threads * AlignmentBytes * pipeStages * stageExtent) {
      // use unrolled direct loads as pipelining is not possible
      fascia::peerOp<threads, unrollFactor,AlignmentBytes, ST, AT, uint32_t>(src, dst, bytes);
    }
    else {
      constexpr int VectorWidth = AlignmentBytes / sizeof(AT);
      using VT = cutlass::AlignedArray<AT, VectorWidth, AlignmentBytes>;
      static_assert(pipeStages >= 1);
      auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
      auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
      const int stages = static_cast<int>(bytes / (threads * AlignmentBytes * stageExtent));
      cuda::static_for<pipeStages>([&vW, &vS](auto i) {
        cuda::static_for<stageExtent>([&i, &vW, &vS](auto j) {
          const int slot = ((i * stageExtent + j) * threads) + threadIdx.x;
          // async gmem -> smem
          cags<AlignmentBytes>(vW + slot, vS + slot);
        });
        cute::cp_async_fence();
      });
      VT reginald[stageExtent];
      for (int i = pipeStages; i < stages; ++i) {
        cute::cp_async_wait<pipeStages - 1>();
        const int stage_out = i - pipeStages;
        const int cs = stage_out % pipeStages;
        cuda::static_for<stageExtent>([&i, &cs, &vW, &reginald, &vS](auto j) {
          const int csW = (cs * stageExtent + j) * threads + threadIdx.x;
          const long int slot = (i * stageExtent + j) * threads + threadIdx.x;
          // smem -> rmem
          reginald[j] = vW[csW];
          // async gmem -> smem prefetch
          cags<AlignmentBytes>(vW + csW, vS + slot);
        });
        cuda::static_for<stageExtent>([&stage_out, &reginald, &vD](auto j) {
          const long int slot = (stage_out * stageExtent + j) * threads + threadIdx.x;
          // rmem -> gmem
          vD[slot] = reginald[j];
        });
        // commit async transfers from this stage
        cute::cp_async_fence();
      }
      // tail
      cuda::static_for<pipeStages>([&vW, &reginald, &vS, &vD, &stages](auto i) {
        const int stage = (stages - pipeStages) + i;
        const int cs = stage % pipeStages;
        cute::cp_async_wait<pipeStages - 1 - i>();
        cuda::static_for<stageExtent>([&i, &cs, &vW, &reginald, &vS, &stages](auto j) {
          const int csW = (cs * stageExtent + j) * threads + threadIdx.x;
          // smem -> rmem
          reginald[j] = vW[csW];
        });
        cuda::static_for<stageExtent>([&stage, &reginald, &vD](auto j) {
          const long int slot = (stage * stageExtent + j) * threads + threadIdx.x;
          // rmem -> gmem
          vD[slot] = reginald[j];
        });
      });
      // residue
      const auto cutoff = stages * static_cast<size_t>(threads * AlignmentBytes * stageExtent);
      const auto cutoffElems = cutoff / AlignmentBytes;
      const auto residue = static_cast<int>((bytes - cutoff) / AlignmentBytes); // elements not bytes
      vS += cutoffElems;
      vD += cutoffElems;
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
        suture::store(vD + i, vS[i]);
      }
    }
  }
};
#endif //SUTURE_TENDON_CUH