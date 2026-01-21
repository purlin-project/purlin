//
// Created by azureuser on 1/8/26.
//

#ifndef TACT_MEMORY_CUH
#define TACT_MEMORY_CUH
#include <nvshmem/nvshmem.h>
namespace tact {
    __host__ __forceinline__
    auto* malloc(const size_t& size) {
      return nvshmem_malloc(size);
    }
    __host__ __forceinline__
    auto* calloc(const size_t& size) {
      return nvshmem_calloc(size, 1);
    }
    __host__ __forceinline__
    void free(void* const& p) {
      nvshmem_free(p);
    }
}
#endif //TACT_MEMORY_CUH