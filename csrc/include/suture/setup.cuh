//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_SETUP_CUH
#define SUTURE_SETUP_CUH
#include <stdexcept>
#include <string>

#include <nvshmem.h>
#include "constants.cuh"
#include "context.cuh"
#include "regime.cuh"

#if !defined(CHECK_CUDA)
#  define CHECK_CUDA(e)                                      \
do {                                                         \
    cudaError_t code = (e);                                  \
    if (code != cudaSuccess) {                               \
        fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",         \
            __FILE__, __LINE__, #e,                          \
            cudaGetErrorName(code),                          \
            cudaGetErrorString(code));                       \
        fflush(stderr);                                      \
        exit(1);                                             \
    }                                                        \
} while (0);
#endif
namespace suture {
  __host__ __forceinline__
  auto initialize(const int& rank, const int& world, cudaStream_t stream,
    const size_t& maxSB = suture::MAX_SUPER_BLOCK_SIZE_,
    const size_t& maxARSize = suture::MAX_ALL_REDUCE_SIZE_) {
    SutureContext ctx{};
    if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
      throw std::runtime_error("nvshmem is not initialized");
    }
    static_assert(suture::AR_LATENCY_BOUND_THRESHOLD % sizeof(LRP16::RT) == 0);
    using SCT = cuda::std::remove_pointer_t<decltype(ctx.sigCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.sigCounter, sizeof(SCT) * world, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.sigCounter, 0, sizeof(SCT) * world, stream));
    using ET = cuda::std::remove_pointer_t<decltype(ctx.epochs)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.epochs, sizeof(ET) * suture::MAX_NUM_CTAS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.epochs, 0, sizeof(ET) * suture::MAX_NUM_CTAS, stream));
    ctx.signals = static_cast<uint64_t*>(nvshmem_calloc(world, sizeof(uint64_t)));
    ctx.sync = static_cast<uint64_t *>(nvshmem_calloc(world * maxSB, sizeof(uint64_t)));
    ctx.staging = static_cast<cuda::std::byte*>(nvshmem_calloc(2 * world * suture::PACKET_BUFFER_SIZE, sizeof(cuda::std::byte)));
    ctx.reduceBuffer = nullptr;
    ctx.reduceBuffer = static_cast<cuda::std::byte*>(nvshmem_malloc(world * maxARSize));
    if (ctx.reduceBuffer == nullptr) {
      throw std::runtime_error("nvshmem_malloc failed");
    }
    ctx.world = cuda::fast_mod_div<int>{world};
    ctx.rank = rank;
    ctx.maxSuperBlockSize = maxSB;
    ctx.maxARSize = maxARSize;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return ctx;
  }

  __host__ __forceinline__
  void finalize(const SutureContext& ctx, cudaStream_t stream) {
    CHECK_CUDA(cudaFreeAsync(ctx.sigCounter, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.epochs, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    nvshmem_free(ctx.signals);
    nvshmem_free(ctx.sync);
    nvshmem_free(ctx.staging);
    nvshmem_free(ctx.reduceBuffer);
  }
}
#endif //SUTURE_SETUP_CUH