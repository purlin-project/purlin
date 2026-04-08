//
// Created by azureuser on 3/26/26.
//

#ifndef SUTURE_SYNC_CUH
#define SUTURE_SYNC_CUH
#include <cuda/atomic>
namespace suture
{
  __device__ __forceinline__
  void arrive(uint64_t* __restrict__ const& peerMailbox, uint64_t* __restrict__ const& myMailbox, const uint64_t& payload) {
    static_assert(kThreads > WARP_SIZE);
    if (threadIdx.x / WARP_SIZE == 0) {
      if (!threadIdx.x) {
        // notify peer
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ap{*peerMailbox};
        ap.store(payload, cuda::memory_order_relaxed);
      }
      __syncwarp();
    }
    else if (threadIdx.x == WARP_SIZE) {
      // wait for notification
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> np{*myMailbox};
      auto isNotified = np.load(cuda::memory_order_relaxed) == payload;
      while (!isNotified) {
        isNotified = np.load(cuda::memory_order_relaxed) == payload;
      }
    }
    __syncthreads();
  }
  __device__ __forceinline__
  void wait(uint64_t* __restrict__ const& peerMailbox, uint64_t* __restrict__ const& myMailbox,
    const uint64_t& payload, uint8_t* __restrict__ const& senseBits) {
    __syncthreads();
    static_assert(kThreads > WARP_SIZE);
    if (!threadIdx.x) {
      // flip senseBit persistently for the next epoch
      *senseBits = static_cast<uint8_t>(payload);
    }
    if (threadIdx.x / WARP_SIZE == 0) {
      // notify
      if (!threadIdx.x) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*peerMailbox};
        p.store(payload, cuda::memory_order_release);
      }
      __syncwarp();
    }
    else if (threadIdx.x == WARP_SIZE){
      // wait
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*myMailbox};
      auto received = p.load(cuda::memory_order_acquire) == payload;
      while (!received) {
        received = p.load(cuda::memory_order_acquire) == payload;
      }
    }
    __syncthreads();
  }
}
#endif //SUTURE_SYNC_CUH