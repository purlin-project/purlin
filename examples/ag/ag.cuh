//
// Created by osayamen on 2/23/26.
//

#ifndef TACK_AG_CUH
#define TACK_AG_CUH
#include <cuda/atomic>
#include <nvshmem.h>

#include "../constants.cuh"
#include "../put.cuh"

struct __align__(16) AGArgs {
  cuda::std::byte* sendBuff = nullptr; // [size], symmetric
  uint64_t* const completions = nullptr; // [ctas, world], symmetric
  uint64_t* const arrivals = nullptr; // [ctas, world], symmetric
  uint64_t* const senseBits = nullptr; // [ctas], local
  size_t size = 0; // per rank message size in bytes
  const int rank = 0;
  const int world = 1;
};

namespace tack
{
  __device__ __forceinline__
  void arrive(const AGArgs& args, const uint64_t& senseBit, const int& peer) {
    static_assert(threads > WARP_SIZE);
    const auto toggledBit = senseBit == 0 ? 1 : 0;
    if (threadIdx.x / WARP_SIZE == 0) {
      // producer
      for (int i = static_cast<int>(threadIdx.x); i < args.world; i += WARP_SIZE) {
        auto* ma = static_cast<uint64_t*>(nvshmem_ptr(args.arrivals + (blockIdx.x * args.world + args.rank), i));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ap{*ma};
        ap.store(toggledBit, cuda::memory_order_relaxed);
      }
    }
    else if (threadIdx.x == WARP_SIZE) {
      // consumer
      auto* na = args.arrivals + (blockIdx.x * args.world + peer);
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
  void wait(const AGArgs& args, const uint64_t& senseBit) {
    __syncthreads();
    static_assert(threads > WARP_SIZE);
    const size_t toggledBit = senseBit == 0 ? 1 : 0;
    if (!threadIdx.x) {
      // flip senseBit persistently for the next epoch
      args.senseBits[blockIdx.x] = toggledBit;
    }
    if (threadIdx.x / WARP_SIZE == 0) {
      auto* ourSP = args.completions + (blockIdx.x * args.world + args.rank);
      // notify
      for (int i = static_cast<int>(threadIdx.x); i < args.world; i += WARP_SIZE) {
        auto* sp = static_cast<uint64_t*>(nvshmem_ptr(ourSP, i));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*sp};
        p.store(toggledBit, cuda::memory_order_release);
      }
    }
    else {
      const auto tid = static_cast<int>(threadIdx.x - WARP_SIZE);
      constexpr auto pollers = threads - WARP_SIZE;
      // wait
      auto* mySP = args.completions + blockIdx.x * args.world;
      for (int i = tid; i < args.world; i += pollers) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*(mySP + i)};
        auto received = p.load(cuda::memory_order_acquire) == toggledBit;
        while (!received) {
          received = p.load(cuda::memory_order_acquire) == toggledBit;
        }
      }
    }
    __syncthreads();
  }
}

__launch_bounds__(threads, 1)
__global__ void ag(const __grid_constant__ AGArgs args) {
  extern __shared__ __align__(Alignment) cuda::std::byte workspace[];
  // compute indices
  // # ctas >= actualWorld
  // size % MAX_ACCESS_ALIGNMENT == 0
  const auto senseBit = args.senseBits[blockIdx.x];
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
  tack::arrive(args, senseBit, peer);
  put(dstP, srcP, workspace, bytes);
  tack::wait(args, senseBit);
}
#endif //TACK_AG_CUH
