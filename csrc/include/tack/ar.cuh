//
// Created by azureuser on 3/25/26.
//

#ifndef TACK_AR_CUH
#define TACK_AR_CUH
#include <cuda/cmath>
#include <cuda/utility>
#include <cuda/ptx>
#include <cutlass/array.h>
#include <cute/arch/copy_sm80.hpp>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <nvshmem.h>
#include "constants.cuh"
#include "rvt.cuh"

struct __align__(16) ARArgs {
  cuda::std::byte* const src = nullptr;
  cuda::std::byte* const dst = nullptr;
  uint64_t* const completions = nullptr; // [world, maxSuperBlockSize], symmetric
  uint64_t* const arrivals = nullptr; // [world, maxSuperBlockSize], symmetric
  uint64_t* const senseBits = nullptr; // [world, maxSuperBlockSize], local
  const size_t ctaBaseChunk = 0;
  const cuda::fast_mod_div<int> superBlockSize_v;
  const cuda::fast_mod_div<int> world_v;
  const int chunkResidue = 0;
  const int maxSuperBlockSize = 1;
  const int rank = 0;
  const int world = 1;
};

namespace tack {
  enum class RedDataType {
    fp16,
    bf16,
    fp32,
    fp64
  };
  __device__ __forceinline__
  void arrive(const ARArgs& args, const uint64_t& senseBit, const int& peer, const int& intraIdx) {
    static_assert(threads > WARP_SIZE);
    const auto toggledBit = senseBit == 0 ? 1 : 0;
    if (threadIdx.x / WARP_SIZE == 0) {
      if (!threadIdx.x) {
        // notify peer
        auto* ma = static_cast<uint64_t*>(nvshmem_ptr(args.arrivals + (args.rank * args.maxSuperBlockSize + intraIdx), peer));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ap{*ma};
        ap.store(toggledBit, cuda::memory_order_relaxed);
      }
      __syncwarp();
    }
    else if (threadIdx.x == WARP_SIZE) {
      auto* na = args.arrivals + (peer * args.maxSuperBlockSize + intraIdx);
      // wait for notification
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> np{*na};
      auto isNotified = np.load(cuda::memory_order_relaxed) == toggledBit;
      while (!isNotified) {
        isNotified = np.load(cuda::memory_order_relaxed) == toggledBit;
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__
  void wait(const ARArgs& args, const uint64_t& senseBit, const int& peer, const int& intraIdx) {
    __syncthreads();
    static_assert(threads > WARP_SIZE);
    const size_t toggledBit = senseBit == 0 ? 1 : 0;
    if (!threadIdx.x) {
      // flip senseBit persistently for the next epoch
      args.senseBits[peer * args.maxSuperBlockSize + intraIdx] = toggledBit;
    }
    if (threadIdx.x / WARP_SIZE == 0) {
      // notify
      if (!threadIdx.x) {
        auto* sp = static_cast<uint64_t*>(nvshmem_ptr(args.completions + (args.rank * args.maxSuperBlockSize + intraIdx), peer));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*sp};
        p.store(toggledBit, cuda::memory_order_release);
      }
      __syncwarp();
    }
    else if (threadIdx.x == WARP_SIZE){
      // wait
      auto* mySP = args.completions + (peer * args.maxSuperBlockSize + intraIdx);
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*mySP};
      auto received = p.load(cuda::memory_order_acquire) == toggledBit;
      while (!received) {
        received = p.load(cuda::memory_order_acquire) == toggledBit;
      }
    }
    __syncthreads();
  }

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
      constexpr auto alignment = redVectorWidth<RedArch, RE>();

    }
  };

  template<int RedArch>
  struct Reduce<800, RedArch> {
    __device__ __forceinline__
    void operator()(const cuda::std::byte* __restrict__ const& src, cuda::std::byte* __restrict__ const& dst, const size_t& bytes) const {
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
    }
  };
}

__launch_bounds__(tack::threads, 1)
__global__ void allReduce(const __grid_constant__ ARArgs args) {
  static_assert(tack::threads > tack::WARP_SIZE && tack::threads % tack::WARP_SIZE == 0);
  extern __shared__ __align__(tack::RED_MAX_ALIGNMENT) cuda::std::byte workspace[];
  // compute indices
  // # ctas >= actualWorld
  // # ctas == superBlockSize
  // size % RED_MAX_ALIGNMENT == 0
  const int superBlockIdx = static_cast<int>(blockIdx.x) / args.superBlockSize_v;
  const int intraIdx = static_cast<int>(blockIdx.x) % args.superBlockSize_v;
  const auto peer = (superBlockIdx + args.rank + 1) % args.world_v;
  const auto senseBit = args.senseBits[peer * args.maxSuperBlockSize + intraIdx];
  // compute buffer offset
  const auto startOffset = (args.ctaBaseChunk * intraIdx + min(intraIdx, args.chunkResidue)) * tack::RED_ALIGNMENT;
  const auto* __restrict__ srcP = args.src + startOffset;
  auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(args.dst + startOffset, peer));
  // total number of aligned elements
  const size_t ctaChunk = args.ctaBaseChunk + (intraIdx < args.chunkResidue);
  const size_t bytes = ctaChunk * tack::RED_ALIGNMENT;
  constexpr tack::Reduce<ARCH, ARCH> reduce{};
  tack::arrive(args, senseBit, peer, intraIdx);
  // call reduce op
  reduce(srcP, dstP, bytes);
  tack::wait(args, senseBit, peer, intraIdx);
}
#endif //TACK_AR_CUH