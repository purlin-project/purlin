//
// Created by osayamen on 2/23/26.
//

#ifndef TACK_AG_CUH
#define TACK_AG_CUH

#include <cuda/cmath>
#include <cuda/utility>
#include <cuda/ptx>
#include <cutlass/array.h>
#include <cute/arch/copy_sm80.hpp>

#include <nvshmem.h>

constexpr int WARP_SIZE = 32;
constexpr int threads = 512;
static_assert(threads >= 32 && threads % WARP_SIZE == 0);
constexpr int Alignment = 16;
constexpr int pipeStages = 2;
constexpr int stageExtent = 4;
constexpr int unrollFactor = 2;
#if (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 9)
constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif
struct __align__(16) AGArgs {
  cuda::std::byte* sendBuff = nullptr; // [size], symmetric
  uint64_t* const completions = nullptr; // [ctas], symmetric
  uint64_t* const arrivals = nullptr; // [ctas, world], symmetric
  uint64_t signal = 0; // epoch
  size_t size = 0; // per rank message size in bytes
  const int rank = 0;
  const int world = 1;
};

namespace tack
{
  __device__ __forceinline__
  void arrive(const AGArgs& args, const int& peer) {
    for (int i = static_cast<int>(threadIdx.x); i < args.world; i += threads) {
      auto* ma = static_cast<uint64_t*>(nvshmem_ptr(args.arrivals + (blockIdx.x * args.world + args.rank), i));
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ap{*ma};
      ap.store(args.signal, cuda::memory_order_relaxed);
    }
    cooperative_groups::invoke_one(cooperative_groups::this_thread_block(), [&args, &peer]() {
      auto* na = args.arrivals + (blockIdx.x * args.world + peer);

      // wait for notification
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> np{*na};
      auto isNotified = np.load(cuda::memory_order_relaxed) == args.signal;
      while (!isNotified) {
        isNotified = np.load(cuda::memory_order_relaxed) == args.signal;
      }
    });
    __syncthreads();
  }
  template <int Size>
  __device__ __forceinline__
  void cp_async_global_to_shared(void* __restrict__ const& smem_ptr, const void* __restrict__ const& gmem_ptr) {
    static_assert(Size == 4 || Size == 8 || Size == 16, "cp.async only supports Size in {4, 8, 16}");
    uint32_t sp = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
      :
      : "r"(sp), "l"(gmem_ptr), "n"(Size)
    );
  }

  template<typename Element>
  __device__ __forceinline__
  void copy(Element* __restrict__ const& dst, const Element* __restrict__ const& src) {
    if constexpr (MAX_ACCESS_ALIGNMENT > 16) {
      const auto v = cuda::ptx::ld(cuda::ptx::space_global, src);
      cuda::ptx::st(cuda::ptx::space_global, dst, v);
    }
    else {
      *dst = *src;
    }
  }

  // GMEM -> GMEM
  template<int Arch = 700>
  struct Put {
    static_assert(Arch >= 700 && Arch < 800);
    __device__ __forceinline__
    void operator()(cuda::std::byte* __restrict__ const& dst, const cuda::std::byte* __restrict__ const& src,
      const size_t& partition /*in bytes*/) const {
      constexpr int VectorWidth = MAX_ACCESS_ALIGNMENT / sizeof(uint);
      using VT = cutlass::AlignedArray<uint, VectorWidth, MAX_ACCESS_ALIGNMENT>;
      static_assert(cuda::std::is_trivially_copyable_v<VT>);
      const int vP = static_cast<int>(partition / MAX_ACCESS_ALIGNMENT);
      auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
      // use unrolled direct loads as pipelining is not necessary
      const auto threadElems = vP / threads;
      const auto trips = threadElems / unrollFactor;
      for (int i = 0; i < trips; ++i) {
        cuda::static_for<unrollFactor>([&i, &vD, &vS](auto j) {
          const auto idx = (i * unrollFactor + j) * threads + threadIdx.x;
          copy(vD + idx, vS + idx);
        });
      }
      const auto residue = vP - trips * unrollFactor * threads;
      vS += (trips * unrollFactor * threads);
      vD += (trips * unrollFactor * threads);
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
        copy(vD + i, vS + i);
      }
    }
  };

  template<>
  struct Put<800> {
    __device__ __forceinline__
    void operator()(cuda::std::byte* __restrict__ const& dst, const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& workspace, const size_t& partition /*in bytes*/) const {
      if (partition <= threads * Alignment * pipeStages * stageExtent) {
        constexpr int VectorWidth = MAX_ACCESS_ALIGNMENT / sizeof(uint);
        using VT = cutlass::AlignedArray<uint, VectorWidth, MAX_ACCESS_ALIGNMENT>;
        static_assert(cuda::std::is_trivially_copyable_v<VT>);
        const int vP = static_cast<int>(partition / MAX_ACCESS_ALIGNMENT);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        // use unrolled direct loads as pipelining is not necessary
        const auto threadElems = vP / threads;
        const auto trips = threadElems / unrollFactor;
        for (int i = 0; i < trips; ++i) {
          cuda::static_for<unrollFactor>([&i, &vD, &vS](auto j) {
            const auto idx = (i * unrollFactor + j) * threads + threadIdx.x;
            copy(vD + idx, vS + idx);
          });
        }
        const auto residue = vP - trips * unrollFactor * threads;
        vS += (trips * unrollFactor * threads);
        vD += (trips * unrollFactor * threads);
        for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
          copy(vD + i, vS + i);
        }
      }
      else {
        constexpr int VectorWidth = Alignment / sizeof(uint);
        using VT = cutlass::AlignedArray<uint, VectorWidth, Alignment>;
        static_assert(pipeStages >= 1);
        auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
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
        // residue
        const auto cutoff = stages * static_cast<size_t>(threads * Alignment * stageExtent);
        const auto cutoffElems = cutoff / Alignment;
        const auto residue = (partition - cutoff) / Alignment; // elements not bytes
        vS += cutoffElems;
        vD += cutoffElems;
        for (size_t i = threadIdx.x; i < residue; i += threads) {
          copy(vD + i, vS + i);
        }
      }
    }
  };
  __device__ __forceinline__
  void wait(const AGArgs& args, const int& peer) {
    __syncthreads();
    cooperative_groups::invoke_one(cooperative_groups::this_thread_block(), [&args, peer] {
      auto* mySP = args.completions + blockIdx.x;
      auto* sP = static_cast<uint64_t*>(nvshmem_ptr(mySP, peer));
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*sP};
      // notify peer
      p.store(args.signal, cuda::memory_order_release);
      // await
      auto received = p.load(cuda::memory_order_acquire) == args.signal;
      while (!received) {
        received = p.load(cuda::memory_order_acquire) == args.signal;
      }
    });
    __syncthreads();
  }
}

__launch_bounds__(threads, 1)
__global__ void ag(const __grid_constant__ AGArgs args) {
  extern __shared__ __align__(Alignment) cuda::std::byte workspace[];
  // compute indices
  // # ctas >= actualWorld
  // size % MAX_ACCESS_ALIGNMENT == 0
  const auto actualWorld = args.world - 1;
  const int numSuperBlocks = actualWorld;
  const auto blocks = gridDim.x;
  const int superBlockIdx = static_cast<int>(blockIdx.x % numSuperBlocks);
  const int intraIdx = static_cast<int>(blockIdx.x) / numSuperBlocks;
  const int superBlockSize = static_cast<int>((blocks / actualWorld) + (superBlockIdx < blocks % actualWorld));
  const auto peer = (superBlockIdx + args.rank + 1) % args.world;

  // total number of aligned elements
  const size_t scaledChunkSize = args.size / MAX_ACCESS_ALIGNMENT;
  const size_t ctaBaseChunk = scaledChunkSize / superBlockSize;
  const int residue = static_cast<int>(scaledChunkSize % superBlockSize);
  const size_t ctaChunk = ctaBaseChunk + (intraIdx < residue);
  // compute buffer offset
  const auto startOffset = (ctaBaseChunk * intraIdx + min(intraIdx, residue)) * MAX_ACCESS_ALIGNMENT;
  const auto* __restrict__ srcP = args.sendBuff + startOffset;
  auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(args.sendBuff + startOffset, peer));
  const size_t bytes = ctaChunk * MAX_ACCESS_ALIGNMENT;

  constexpr tack::Put<ARCH> put{};
  tack::arrive(args, peer);
  put(dstP, srcP, workspace, bytes);
  tack::wait(args, peer);
}
#endif //TACK_AG_CUH
