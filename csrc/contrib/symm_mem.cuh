#ifndef PURLIN_CONTRIB_SYMM_MEM_CUH
#define PURLIN_CONTRIB_SYMM_MEM_CUH

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <nvshmem.h>

namespace purlin {
  // Initialize NVSHMEM and select the CUDA device before use.
  struct NvshmemMemory {
    struct Allocation {
      std::vector<void*> peers;
      void* multicast = nullptr;
      void* local = nullptr;
    };

    Allocation allocate_zeroed(size_t bytes, size_t alignment, cudaStream_t stream) {
      Allocation allocation;
      const int world = nvshmem_n_pes();
      try {
        allocation.peers.resize(world);
      } catch (...) {
        fail("Could not allocate the peer pointer array");
      }
      allocation.local = nvshmem_align(alignment, bytes);
      if (allocation.local == nullptr) fail("Could not allocate symmetric memory");
      for (int peer = 0; peer < world; ++peer) {
        allocation.peers[peer] = nvshmem_ptr(allocation.local, peer);
        if (allocation.peers[peer] == nullptr) fail("A peer has no direct GPU mapping");
      }
      if (nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE) == world &&
          std::getenv("PURLIN_DISABLE_MULTIMEM") == nullptr) {
        allocation.multicast = nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, allocation.local);
      }
      check(cudaMemsetAsync(allocation.local, 0, bytes, stream));
      check(cudaStreamSynchronize(stream));
      nvshmem_barrier_all();
      return allocation;
    }

    void deallocate(Allocation& allocation) noexcept {
      nvshmem_free(allocation.local);
      allocation = {};
    }

  private:
    [[noreturn]] static void fail(const char* message) {
      std::fprintf(stderr, "Purlin NVSHMEM provider: %s\n", message);
      nvshmem_global_exit(EXIT_FAILURE);
      std::abort();
    }

    static void check(cudaError_t status) {
      if (status != cudaSuccess) fail(cudaGetErrorString(status));
    }
  };
}

#endif // PURLIN_CONTRIB_SYMM_MEM_CUH
