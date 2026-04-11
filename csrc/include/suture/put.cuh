//
// Created by Osayamen on 3/20/26.
//

#ifndef SUTURE_PUT_CUH
#define SUTURE_PUT_CUH

#include <cuda/cmath>
#include <cuda/utility>
#include <cuda/ptx>
#include <cutlass/array.h>
#include <cute/arch/copy_sm80.hpp>

#include "constants.cuh"
#include "copy.cuh"
namespace suture {
  // GMEM -> GMEM
  template<int Arch = 700>
  struct Put {
    static_assert(Arch >= 700 && Arch < 800);
    __device__ __forceinline__
    void operator()(cuda::std::byte* __restrict__ const& dst, const cuda::std::byte* __restrict__ const& src,
      const size_t& bytes /*in bytes*/) const {
      constexpr int VectorWidth = MAX_ACCESS_ALIGNMENT / sizeof(uint);
      using VT = cutlass::AlignedArray<uint, VectorWidth, MAX_ACCESS_ALIGNMENT>;
      static_assert(cuda::std::is_trivially_copyable_v<VT>);
      const auto vP = bytes / MAX_ACCESS_ALIGNMENT;
      auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
      // use unrolled direct loads as pipelining is not necessary
      const auto threadElems = vP / kThreads;
      const auto trips = threadElems / kUnrollFactor;
      for (auto i = 0; i < trips; ++i) {
        VT reginald[kUnrollFactor];
        uint indices[kUnrollFactor];
        // precompute indices
        cuda::static_for<kUnrollFactor>([&i, &indices](auto j) {
          indices[j] = (i * kUnrollFactor + j) * kThreads + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<kUnrollFactor>([&vS, &indices, &reginald](auto j) {
          reginald[j] = suture::load(vS + indices[j]);
        });
        // rmem -> gmem
        cuda::static_for<kUnrollFactor>([&vD, &indices, &reginald](auto j) {
          suture::store(vD + indices[j], reginald[j]);
        });
      }
      const auto residue = vP - trips * static_cast<size_t>(kUnrollFactor * kThreads);
      vS += (trips * static_cast<size_t>(kUnrollFactor * kThreads));
      vD += (trips * static_cast<size_t>(kUnrollFactor * kThreads));
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += kThreads) {
        copy(vD + i, vS + i);
      }
    }
  };

  template<>
  struct Put<800> {
    __device__ __forceinline__
    void operator()(cuda::std::byte* __restrict__ const& dst, const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& workspace, const size_t& bytes /*in bytes*/) const {
      if (bytes < kThreads * kAlignment * kPipeStages * kStageExtent) {
        constexpr int VectorWidth = MAX_ACCESS_ALIGNMENT / sizeof(uint);
        using VT = cutlass::AlignedArray<uint, VectorWidth, MAX_ACCESS_ALIGNMENT>;
        static_assert(cuda::std::is_trivially_copyable_v<VT>);
        const int vP = static_cast<int>(bytes / MAX_ACCESS_ALIGNMENT);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        // use unrolled direct loads as pipelining is not necessary
        const auto threadElems = vP / kThreads;
        const auto trips = threadElems / kUnrollFactor;
        for (int i = 0; i < trips; ++i) {
          VT reginald[kUnrollFactor];
          uint indices[kUnrollFactor];
          // precompute indices
          cuda::static_for<kUnrollFactor>([&i, &indices](auto j) {
            indices[j] = (i * kUnrollFactor + j) * kThreads + threadIdx.x;
          });
          // gmem -> rmem
          cuda::static_for<kUnrollFactor>([&vS, &indices, &reginald](auto j) {
            reginald[j] = suture::load(vS + indices[j]);
          });
          // rmem -> gmem
          cuda::static_for<kUnrollFactor>([&vD, &indices, &reginald](auto j) {
            suture::store(vD + indices[j], reginald[j]);
          });
        }
        const auto residue = vP - trips * kUnrollFactor * kThreads;
        vS += (trips * kUnrollFactor * kThreads);
        vD += (trips * kUnrollFactor * kThreads);
        for (int i = static_cast<int>(threadIdx.x); i < residue; i += kThreads) {
          copy(vD + i, vS + i);
        }
      }
      else {
        constexpr int VectorWidth = kAlignment / sizeof(uint);
        using VT = cutlass::AlignedArray<uint, VectorWidth, kAlignment>;
        static_assert(kPipeStages >= 1);
        auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        const int stages = static_cast<int>(bytes / (kThreads * kAlignment * kStageExtent));
        cuda::static_for<kPipeStages>([&vW, &vS](auto i) {
          cuda::static_for<kStageExtent>([&i, &vW, &vS](auto j) {
            const int slot = ((i * kStageExtent + j) * kThreads) + threadIdx.x;
            // async gmem -> smem
            cpAsync<kAlignment>(vW + slot, vS + slot);
          });
          cute::cp_async_fence();
        });
        VT reginald[kStageExtent];
        for (int i = kPipeStages; i < stages; ++i) {
          cute::cp_async_wait<kPipeStages - 1>();
          const int stage_out = i - kPipeStages;
          const int cs = stage_out % kPipeStages;
          cuda::static_for<kStageExtent>([&i, &cs, &vW, &reginald, &vS](auto j) {
            const int csW = (cs * kStageExtent + j) * kThreads + threadIdx.x;
            const long int slot = (i * kStageExtent + j) * kThreads + threadIdx.x;
            // smem -> rmem
            reginald[j] = vW[csW];
            // async gmem -> smem prefetch
            cpAsync<kAlignment>(vW + csW, vS + slot);
          });
          cuda::static_for<kStageExtent>([&stage_out, &reginald, &vD](auto j) {
            const long int slot = (stage_out * kStageExtent + j) * kThreads + threadIdx.x;
            // rmem -> gmem
            vD[slot] = reginald[j];
          });
          // commit async transfers from this stage
          cute::cp_async_fence();
        }
        // tail
        cuda::static_for<kPipeStages>([&vW, &reginald, &vS, &vD, &stages](auto i) {
          const int stage = (stages - kPipeStages) + i;
          const int cs = stage % kPipeStages;
          cute::cp_async_wait<kPipeStages - 1 - i>();
          cuda::static_for<kStageExtent>([&i, &cs, &vW, &reginald, &vS, &stages](auto j) {
            const int csW = (cs * kStageExtent + j) * kThreads + threadIdx.x;
            // smem -> rmem
            reginald[j] = vW[csW];
          });
          cuda::static_for<kStageExtent>([&stage, &reginald, &vD](auto j) {
            const long int slot = (stage * kStageExtent + j) * kThreads + threadIdx.x;
            // rmem -> gmem
            vD[slot] = reginald[j];
          });
        });
        // residue
        const auto cutoff = stages * static_cast<size_t>(kThreads * kAlignment * kStageExtent);
        const auto cutoffElems = cutoff / kAlignment;
        const auto residue = (bytes - cutoff) / kAlignment; // elements not bytes
        vS += cutoffElems;
        vD += cutoffElems;
        for (size_t i = threadIdx.x; i < residue; i += kThreads) {
          copy(vD + i, vS + i);
        }
      }
    }
  };
}
#endif //SUTURE_PUT_CUH