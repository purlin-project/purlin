//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_SETUP_CUH
#define SUTURE_SETUP_CUH
#include <stdexcept>
#include <string>

#include <nvshmem.h>
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
  auto initialize(const int& rank, const int& world, cudaStream_t stream, const size_t& maxSB = suture::MAX_SUPER_BLOCK_SIZE_,
    const size_t& maxARSize = suture::MAX_ALL_REDUCE_SIZE_) {
    SutureContext ctx{};
    // we assume single-node
    if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
      throw std::runtime_error("nvshmem is not initialized");
    }

    static_assert(suture::AR_LATENCY_BOUND_THRESHOLD % sizeof(LRP16::RT) == 0);
    using FPT = cuda::std::remove_pointer_t<decltype(ctx.flagPutSense)>;
    using FRT = cuda::std::remove_pointer_t<decltype(ctx.flagRedSense)>;
    static_assert(cuda::std::is_same_v<FPT, FRT>);
    const auto flagSize = sizeof(FPT) * 2 * world * FLAG_BUFFER_SIZE;
    CHECK_CUDA(cudaMallocAsync(&ctx.flagPutSense,flagSize, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.flagPutSense, 0, flagSize, stream));
    CHECK_CUDA(cudaMallocAsync(&ctx.flagRedSense, flagSize, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.flagRedSense, 0, flagSize, stream));
    using SCT = cuda::std::remove_pointer_t<decltype(ctx.sigCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.sigCounter, sizeof(SCT), stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.sigCounter, 0, sizeof(SCT), stream));

    ctx.signals = static_cast<uint32_t*>(nvshmem_calloc(world, sizeof(uint32_t)));
    ctx.sync0 = static_cast<uint64_t *>(nvshmem_calloc(world * maxSB, sizeof(uint64_t)));
    ctx.sync1 = static_cast<uint64_t*>(nvshmem_calloc(world * maxSB, sizeof(uint64_t)));
    ctx.senseBitsTR = static_cast<uint8_t *>(nvshmem_calloc(world * maxSB, sizeof(uint8_t)));
    ctx.senseBitsLR = static_cast<uint8_t *>(nvshmem_calloc(world * maxSB, sizeof(uint8_t)));
    ctx.staging = static_cast<cuda::std::byte*>(nvshmem_calloc(2 * world * suture::PACKET_BUFFER_SIZE, sizeof(cuda::std::byte)));
    ctx.reduceBuffer = static_cast<cuda::std::byte*>(nvshmem_malloc(world * maxARSize));
    ctx.world = cuda::fast_mod_div<int>{world};
    ctx.rank = rank;
    return ctx;
  }

  __host__ __forceinline__
  void finalize(const SutureContext& ctx, cudaStream_t stream) {
    CHECK_CUDA(cudaFreeAsync(ctx.flagPutSense, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.flagRedSense, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.sigCounter, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    nvshmem_free(ctx.signals);
    nvshmem_free(ctx.sync0);
    nvshmem_free(ctx.sync1);
    nvshmem_free(ctx.senseBitsTR);
    nvshmem_free(ctx.senseBitsLR);
    nvshmem_free(ctx.staging);
    nvshmem_free(ctx.reduceBuffer);
  }
}
#endif //SUTURE_SETUP_CUH