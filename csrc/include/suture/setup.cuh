//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_SETUP_CUH
#define SUTURE_SETUP_CUH
#include <stdexcept>
#include <string>
#include <vector>

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
  auto initialize(const int& rank, const int& world, cudaStream_t stream) {
    if (RED_PUT_BLOCKS % world != 0) {
      throw std::runtime_error("RED_PUT_BLOCKS: " +
        std::to_string(RED_PUT_BLOCKS) + " should be a multiple of world: " + std::to_string(world));
    }
    Context ctx{};
    if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
      throw std::runtime_error("nvshmem is not initialized");
    }
    static_assert(suture::RED_LATENCY_BOUND_THRESHOLD % sizeof(LRP16::RT) == 0);
    using ET = cuda::std::remove_pointer_t<decltype(ctx.epochs)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.epochs, sizeof(ET) * suture::MAX_NUM_CTAS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.epochs, 0, sizeof(ET) * suture::MAX_NUM_CTAS, stream));
    using PCT = cuda::std::remove_pointer_t<decltype(ctx.putCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.putCounter, sizeof(PCT) * world, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.putCounter, 0, sizeof(PCT) * world, stream));
    using GT = cuda::std::remove_pointer_t<decltype(ctx.groupSense)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.groupSense, sizeof(GT) * world, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.groupSense, 0, sizeof(GT) * world, stream));
    void* signals = nullptr;
    std::vector<uint64_t*> signalsV(world);
    {
      const auto signalsPtrBytes = sizeof(decltype(signalsV)::value_type) * signalsV.size();
      const auto* base = static_cast<uint64_t*>(nvshmem_calloc(world, sizeof(uint64_t)));
      CHECK_CUDA(cudaMallocAsync(&signals, signalsPtrBytes, stream));
      for (int i = 0; i < world; ++i) {
        signalsV[i] = static_cast<uint64_t*>(nvshmem_ptr(base, i));
      }
      CHECK_CUDA(cudaMemcpyAsync(signals, signalsV.data(), signalsPtrBytes, cudaMemcpyHostToDevice, stream));
      ctx.signals = static_cast<uint64_t**>(signals);
    }

    void* stagingTR = nullptr;
    std::vector<cuda::std::byte*> stagingTRV(world);
    {
      const auto stagingPtrBytes = sizeof(decltype(stagingTRV)::value_type) * stagingTRV.size();
      const auto* base = static_cast<cuda::std::byte*>(nvshmem_malloc(2 * STAGING_BUFFER_SIZE_));
      CHECK_CUDA(cudaMallocAsync(&stagingTR, stagingPtrBytes, stream));
      for (int i = 0; i < world; ++i) {
        stagingTRV[i] = static_cast<cuda::std::byte*>(nvshmem_ptr(base, i));
      }
      CHECK_CUDA(cudaMemcpyAsync(stagingTR, stagingTRV.data(), stagingPtrBytes, cudaMemcpyHostToDevice, stream));
      ctx.staging = static_cast<cuda::std::byte**>(stagingTR);
    }

    void* staging = nullptr;
    std::vector<cuda::std::byte*> stagingV(world);
    {
      const auto stagingPtrBytes = sizeof(decltype(stagingV)::value_type) * stagingV.size();
      CHECK_CUDA(cudaMallocAsync(&staging, stagingPtrBytes, stream));
      const auto* base = static_cast<cuda::std::byte*>(nvshmem_calloc(2 * world * suture::PACKET_BUFFER_SIZE, sizeof(cuda::std::byte)));
      for (int i = 0; i < world; ++i) {
        stagingV[i] = static_cast<cuda::std::byte*>(nvshmem_ptr(base, i));
      }
      CHECK_CUDA(cudaMemcpyAsync(staging, stagingV.data(), stagingPtrBytes, cudaMemcpyHostToDevice, stream));
      ctx.stagingLR = static_cast<cuda::std::byte**>(staging);
    }
    ctx.world = cuda::fast_mod_div<int, true>{world};
    ctx.rank = rank;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return ctx;
  }

  __host__ __forceinline__
  void finalize(const Context& ctx, cudaStream_t stream) {
    CHECK_CUDA(cudaFreeAsync(ctx.epochs, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.putCounter, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.groupSense, stream));
    std::array<void*, 4> heaps{};
    static_assert(sizeof(decltype(ctx.signals + ctx.rank)) == sizeof(void*));
    CHECK_CUDA(cudaMemcpyAsync(heaps.data(), ctx.signals + ctx.rank, sizeof(void*),
      cudaMemcpyDeviceToHost, stream));
    static_assert(sizeof(decltype(ctx.stagingLR + ctx.rank)) == sizeof(void*));
    CHECK_CUDA(cudaMemcpyAsync(heaps.data() + 2, ctx.stagingLR + ctx.rank, sizeof(void*),
      cudaMemcpyDeviceToHost, stream));
    static_assert(sizeof(decltype(ctx.staging + ctx.rank)) == sizeof(void*));
    CHECK_CUDA(cudaMemcpyAsync(heaps.data() + 3, ctx.staging + ctx.rank, sizeof(void*),
      cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (auto const& heap : heaps) {
      nvshmem_free(heap);
    }
  }
}
#endif //SUTURE_SETUP_CUH