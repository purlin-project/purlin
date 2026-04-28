//
// Created by azureuser on 3/26/26.
//

#ifndef SUTURE_SYNC_CUH
#define SUTURE_SYNC_CUH
#include <cuda/atomic>
namespace suture {
  __device__ __forceinline__
  void syncRelaxed(const size_t& rOffset, const size_t& lOffset, const uint64_t& payload,
    uint64_t* __restrict__ const& syncP) {
    if (threadIdx.x / WARP_SIZE == 0) {
      if (!threadIdx.x) {
        // notify peer
        const cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ap{*(syncP + rOffset)};
        ap.store(static_cast<uint64_t>(payload), cuda::memory_order_relaxed);
      }
      __syncwarp();
    }
    else if (threadIdx.x == WARP_SIZE) {
      // wait for notification
      const cuda::atomic_ref<uint64_t, cuda::thread_scope_system> np{*(syncP + lOffset)};
      auto isNotified = np.load(cuda::memory_order_relaxed) >= payload;
      while (!isNotified) {
        isNotified = np.load(cuda::memory_order_relaxed) >= payload;
      }
    }
    __syncthreads();
  }
  __device__ __forceinline__
  void syncStrong(const size_t& rOffset, const size_t& lOffset, const uint64_t& payload,
    uint64_t* __restrict__ const& syncP) {
    if (threadIdx.x / WARP_SIZE == 0) {
      // notify
      if (!threadIdx.x) {
        const cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*(syncP + rOffset)};
        p.store(payload, cuda::memory_order_release);
      }
      __syncwarp();
    }
    else if (threadIdx.x == WARP_SIZE){
      // wait
      const cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*(syncP + lOffset)};
      auto received = p.load(cuda::memory_order_relaxed) >= payload;
      while (!received) {
        received = p.load(cuda::memory_order_relaxed) >= payload;
      }
      cuda::std::ignore = p.load(cuda::memory_order_acquire);
    }
    __syncthreads();
  }
}
#endif //SUTURE_SYNC_CUH