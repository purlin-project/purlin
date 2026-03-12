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

namespace tack
{
  __device__ __forceinline__
  void arrive(const AGArgs& args, const uint64_t& senseBit, const int& peer, const int& intraIdx) {
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
  void wait(const AGArgs& args, const uint64_t& senseBit, const int& peer, const int& intraIdx) {
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
}

__launch_bounds__(threads, 1)
__global__ void ag(const __grid_constant__ AGArgs args) {
  static_assert(threads > WARP_SIZE && threads % WARP_SIZE == 0);
  extern __shared__ __align__(Alignment) cuda::std::byte workspace[];
  // compute indices
  // # ctas >= actualWorld
  // # ctas == superBlockSi
  // size % MAX_ACCESS_ALIGNMENT == 0
  const int superBlockIdx = static_cast<int>(blockIdx.x) / args.superBlockSize_v;
  const int intraIdx = static_cast<int>(blockIdx.x) % args.superBlockSize_v;
  const auto peer = (superBlockIdx + args.rank + 1) % args.world_v;
  const auto senseBit = args.senseBits[peer * args.maxSuperBlockSize + intraIdx];
  // compute buffer offset
  const auto startOffset = (args.ctaBaseChunk * intraIdx + min(intraIdx, args.chunkResidue)) * MAX_ACCESS_ALIGNMENT;
  const auto* __restrict__ srcP = args.sendBuff + startOffset;
  auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(args.sendBuff + startOffset, peer));
  // total number of aligned elements
  const size_t ctaChunk = args.ctaBaseChunk + (intraIdx < args.chunkResidue);
  const size_t bytes = ctaChunk * MAX_ACCESS_ALIGNMENT;

  constexpr tack::Put<ARCH> put{};
  tack::arrive(args, senseBit, peer, intraIdx);
  //nvshmemx_putmem_nbi_block(args.sendBuff + startOffset, srcP, bytes, peer);
  put(dstP, srcP, workspace, bytes);
  tack::wait(args, senseBit, peer, intraIdx);
}
#endif //TACK_AG_CUH
