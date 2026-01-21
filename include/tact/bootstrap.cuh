//
// Created by Osayamen on 1/8/26.
//

#ifndef TACT_BOOTSTRAP_CUH
#define TACT_BOOTSTRAP_CUH

#include <nvshmem/nvshmem.h>
namespace tact {
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
#endif //TACT_BOOTSTRAP_CUH
