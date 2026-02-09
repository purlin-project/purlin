//
// Created by Osayamen on 1/8/26.
//

#ifndef tack_BOOTSTRAP_CUH
#define tack_BOOTSTRAP_CUH

#include <nvshmem.h>
namespace tack {
  __host__ __forceinline__
  void initialize() {
    nvshmem_init();
  }
  __host__ __device__ __forceinline__
  int rank() {
    return nvshmem_my_pe();
  }
  __host__ __device__ __forceinline__
  int world() {
    return nvshmem_n_pes();
  }
  __host__ __forceinline__
  void finalize() {
    nvshmem_finalize();
  }
}
#endif //tack_BOOTSTRAP_CUH
