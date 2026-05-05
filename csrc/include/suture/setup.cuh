//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_SETUP_CUH
#define SUTURE_SETUP_CUH
#include <stdexcept>
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
  auto initialize(const int& rank, const int& world, cudaStream_t stream,
    const size_t& maxSB = suture::MAX_SUPER_BLOCK_SIZE_,
    const size_t& maxARSize = suture::STAGING_BUFFER_SIZE_) {
    Context ctx{};
    if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
      throw std::runtime_error("nvshmem is not initialized");
    }
    static_assert(suture::RED_LATENCY_BOUND_THRESHOLD % sizeof(LRP16::RT) == 0);
    using ET = cuda::std::remove_pointer_t<decltype(ctx.epochs)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.epochs, sizeof(ET) * suture::MAX_NUM_CTAS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.epochs, 0, sizeof(ET) * suture::MAX_NUM_CTAS, stream));
    CHECK_CUDA(cudaMallocAsync(&ctx.putCounter, sizeof(uint32_t), stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.putCounter, 0, sizeof(uint32_t), stream));
    CHECK_CUDA(cudaMallocAsync(&ctx.groupSense, sizeof(uint32_t), stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.groupSense, 0, sizeof(uint32_t), stream));
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

    void* sync = nullptr;
    std::vector<uint64_t*> syncV(world);
    {
      const auto syncPtrBytes = sizeof(decltype(syncV)::value_type) * syncV.size();
      const auto* base = static_cast<uint64_t*>(nvshmem_calloc(world * maxSB, sizeof(uint64_t)));
      CHECK_CUDA(cudaMallocAsync(&sync, syncPtrBytes, stream));
      for (int i = 0; i < world; ++i) {
        syncV[i] = static_cast<uint64_t*>(nvshmem_ptr(base, i));
      }
      CHECK_CUDA(cudaMemcpyAsync(sync, syncV.data(), syncPtrBytes, cudaMemcpyHostToDevice, stream));
      ctx.sync = static_cast<uint64_t**>(sync);
    }

    void* stagingTR = nullptr;
    std::vector<cuda::std::byte*> stagingTRV(world);
    {
      const auto stagingPtrBytes = sizeof(decltype(stagingTRV)::value_type) * stagingTRV.size();
      const auto* base = static_cast<cuda::std::byte*>(nvshmem_malloc(2 * maxARSize));
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
    ctx.maxSuperBlockSize = maxSB;
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
    static_assert(sizeof(decltype(ctx.sync + ctx.rank)) == sizeof(void*));
    CHECK_CUDA(cudaMemcpyAsync(heaps.data() + 1, ctx.sync + ctx.rank, sizeof(void*),
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