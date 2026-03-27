//
// Created by azureuser on 3/27/26.
//

#ifndef TACK_REDUCE_CUH
#define TACK_REDUCE_CUH
#include "copy.cuh"
#include "rvt.cuh"
namespace tack {
  enum class RedDataType {
    fp16,
    bf16,
    fp32,
    fp64
  };

  template<int Arch, RedDataType r>
  consteval auto redVectorWidth() {
    if (Arch < 900) {
      if (r == RedDataType::bf16 || r == RedDataType::fp16) {
        return 2;
      }
      return 1;
    }
    // Hopper and above
    if (r == RedDataType::bf16 || r == RedDataType::fp16) {
      return 8;
    }
    if (r == RedDataType::fp32) {
      return 4;
    }
    return 1;
  }
  using RedElement = __half;
  constexpr auto RE = RedDataType::fp16;
  constexpr int RED_ALIGNMENT = redVectorWidth<800, RE>() * sizeof(RedElement);
  template<int PutArch, int RedArch>
  struct Reduce {
    static_assert(PutArch >= 700 && PutArch < 800);
    __device__ __forceinline__
    void operator()(const cuda::std::byte* __restrict__ const& src, cuda::std::byte* __restrict__ const& dst, const size_t& bytes) const {
      using RAT = RedAddType<RedElement, RED_ALIGNMENT>::Type;
      constexpr int redVW = RED_ALIGNMENT / sizeof(RAT);
      using RedAddOp = RedAdd<RedArch, RAT, redVW>;
      constexpr RedAddOp op{};
      constexpr auto alignment = redVectorWidth<RedArch, RE>();
      using VT = cutlass::AlignedArray<RAT, redVW>;
      const int vP = static_cast<int>(bytes / RED_ALIGNMENT);
      auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
      // use unrolled direct loads as pipelining is not necessary
      const auto threadElems = vP / threads;
      const auto trips = threadElems / unrollFactor;
      for (int i = 0; i < trips; ++i) {
        VT reginald[unrollFactor];
        uint indices[unrollFactor];
        // precompute indices
        cuda::static_for<unrollFactor>([&i, &indices](auto j) {
          indices[j] = (i * unrollFactor + j) * threads + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<unrollFactor>([&vS, &indices, &reginald](auto j) {
          reginald[j] = vS[indices[j]];
        });
        // rmem -> gmem reduction
        cuda::static_for<unrollFactor>([&vD, &indices, &reginald](auto j) {
          auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + indices[j]);
          op(dstP, reginald[j]);
        });
      }
      const auto residue = vP - trips * unrollFactor * threads;
      vS += (trips * unrollFactor * threads);
      vD += (trips * unrollFactor * threads);
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
        const auto v = vS[i];
        auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + i);
        op(dstP, v);
      }
    }
  };

  template<int RedArch>
  struct Reduce<800, RedArch> {
    __device__ __forceinline__
    void operator()(const cuda::std::byte* __restrict__ const& src, cuda::std::byte* __restrict__ const& dst,
      cuda::std::byte* __restrict__ const& workspace, const size_t& bytes) const {
      using RAT = RedAddType<RedElement, RED_ALIGNMENT>::Type;
      constexpr int redVW = RED_ALIGNMENT / sizeof(RAT);
      using RedAddOp = RedAdd<800, RAT, redVW>;
      constexpr RedAddOp op{};
      if (bytes <= threads * RED_ALIGNMENT * pipeStages * stageExtent) {
        using VT = cutlass::AlignedArray<RAT, redVW>;
        const int vP = static_cast<int>(bytes / RED_ALIGNMENT);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        // use unrolled direct loads as pipelining is not necessary
        const auto threadElems = vP / threads;
        const auto trips = threadElems / unrollFactor;
        for (int i = 0; i < trips; ++i) {
          VT reginald[unrollFactor];
          uint indices[unrollFactor];
          // precompute indices
          cuda::static_for<unrollFactor>([&i, &indices](auto j) {
            indices[j] = (i * unrollFactor + j) * threads + threadIdx.x;
          });
          // gmem -> rmem
          cuda::static_for<unrollFactor>([&vS, &indices, &reginald](auto j) {
            reginald[j] = vS[indices[j]];
          });
          // rmem -> gmem reduction
          cuda::static_for<unrollFactor>([&vD, &indices, &reginald](auto j) {
            auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + indices[j]);
            op(dstP, reginald[j]);
          });
        }
        const auto residue = vP - trips * unrollFactor * threads;
        vS += (trips * unrollFactor * threads);
        vD += (trips * unrollFactor * threads);
        for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
          const auto v = vS[i];
          auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + i);
          op(dstP, v);
        }
      }
      else {
        constexpr auto copyAlignment = RED_ALIGNMENT;
        constexpr int VectorWidth = copyAlignment / sizeof(RAT);
        constexpr int nAddOps = VectorWidth / RedAddOp::VectorWidth::value;
        using VT = cutlass::AlignedArray<RAT, VectorWidth, copyAlignment>;
        using AT = cutlass::AlignedArray<RAT, RedAddOp::VectorWidth::value>;
        static_assert(pipeStages >= 1);
        auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        const int stages = static_cast<int>(bytes / (threads * copyAlignment * stageExtent));
        cuda::static_for<pipeStages>([&vW, &vS](auto i) {
          cuda::static_for<stageExtent>([&i, &vW, &vS](auto j) {
            const int slot = ((i * stageExtent + j) * threads) + threadIdx.x;
            // async gmem -> smem
            cp_async_global_to_shared<copyAlignment>(vW + slot, vS + slot);
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
            cp_async_global_to_shared<copyAlignment>(vW + csW, vS + slot);
          });
          cuda::static_for<stageExtent>([&stage_out, &reginald, &vD](auto j) {
            const long int slot = (stage_out * stageExtent + j) * threads + threadIdx.x;
            // rmem -> gmem, reduction
            const auto v = reginald[j];
            cuda::static_for<nAddOps>([&vD, &v](auto k) {
              auto* __restrict__ vDp = reinterpret_cast<RAT*>(vD + slot) + k;
              op(vDp, )
            });
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
        const auto cutoff = stages * static_cast<size_t>(threads * copyAlignment * stageExtent);
        const auto cutoffElems = cutoff / copyAlignment;
        const auto residue = (bytes - cutoff) / copyAlignment; // elements not bytes
        vS += cutoffElems;
        vD += cutoffElems;
        for (size_t i = threadIdx.x; i < residue; i += threads) {
          copy(vD + i, vS + i);
        }
      }
    }
  };
}
#endif //TACK_REDUCE_CUH