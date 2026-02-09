//
// Created by osayamen on 2/9/26.
//

#ifndef TACK_GEMM_AG_CUH
#define TACK_GEMM_AG_CUH
#include <cuda/utility>
#include <cuda/cmath>
#include <cuda/atomic>

#include "gemm_ag.cuh"
#include "tile.cuh"

namespace tack::examples
{
  constexpr int threads = 128;
  constexpr int bM = 128;
  constexpr int bN = 128;
  constexpr int bK = 64;
  constexpr int pipeStages = ARCH >= 800 ? 2 : 1;
  using Element = __half;
  using TileGEMM = tile::CollectiveMainloop<ARCH, bM, bN, bK, Element, float, threads, pipeStages>;
  template <int Size>
  __device__ __forceinline__
  void cp_async_global_to_shared(void* __restrict__ const& smem_ptr, const void* __restrict__ const& gmem_ptr) {
    static_assert(Size == 4 || Size == 8 || Size == 16,
                  "cp.async only supports Size in {4, 8, 16}");
    uint32_t sp = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
      :
      : "r"(sp), "l"(gmem_ptr), "n"(Size)
    );
  }

  template <int pipeStages = 2, int stageExtent = 2, int Alignment = 16>
  __device__ __forceinline__
  void put(void* __restrict__ const& dst, const void* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& workspace, const size_t& partition) {
    using VT = uint4;
    static_assert(cuda::is_power_of_two(pipeStages) && pipeStages >= 1 && pipeStages <= 8);
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    const int vP = static_cast<int>(partition / Alignment);
    auto* __restrict__ vD = static_cast<VT*>(dst);
    const auto* __restrict__ vS = static_cast<const VT*>(src);
    if (partition <= threads * Alignment * pipeStages * stageExtent) {
      // use direct loads as pipelining is not necessary
      #pragma unroll 2
      for (int i = static_cast<int>(threadIdx.x); i < vP; i += threads) {
        vD[i] = vS[i];
      }
    }
    else {
      const int stages = static_cast<int>(partition / (threads * Alignment * stageExtent));
      cuda::static_for<pipeStages>([&vW, &vS](auto i) {
        cuda::static_for<stageExtent>([&i, &vW, &vS](auto j) {
          const int slot = ((i * stageExtent + j) * threads) + threadIdx.x;
          // async gmem -> smem
          cp_async_global_to_shared<Alignment>(vW + slot, vS + slot);
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
          cp_async_global_to_shared<Alignment>(vW + csW, vS + slot);
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
    }
  }

  struct Args {
    const cuda::std::byte* const A; // [M / world, K]
    const cuda::std::byte* const B; // [N / world, K]
    cuda::std::byte* const C; // [M / world, N]
    uint64_t* const notification; // symmetric [ctas, world]
    uint64_t epoch;
    const uint M;
    const uint localM; // M / world
    const uint N;
    const uint localN; // N / world
    const uint K;
    const int world;
    const int rank;
  };

  // TODO standalone AllGather
  __device__ __forceinline__
  void gemmAG(const Args& args) {
    // notify all
    auto* __restrict__ nP = args.notification + blockIdx.x * args.world;
    for (uint i = threadIdx.x; i < args.world; i += threads) {
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db{*(nP + i)};
      db.fetch_add(1, cuda::std::memory_order_release);
    }
    const uint numTiles =
  }
}
#endif //TACK_GEMM_AG_CUH
