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
#include "packet.cuh"

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
    constexpr auto leastChunkSize = cute::min(AG_CHUNK_SIZE, RED_CHUNK_SIZE, RS_CHUNK_SIZE);
    constexpr auto maxChunks = suture::STAGING_BUFFER_SIZE_ / leastChunkSize;
    static_assert(suture::RED_LATENCY_BOUND_THRESHOLD % sizeof(LRP16::RT) == 0);
    using ET = cuda::std::remove_pointer_t<decltype(ctx.epochs)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.epochs, sizeof(ET) * suture::MAX_NUM_CTAS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.epochs, 0, sizeof(ET) * suture::MAX_NUM_CTAS, stream));
    using PCT = cuda::std::remove_pointer_t<decltype(ctx.putCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.putCounter, sizeof(PCT) * world * maxChunks, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.putCounter, 0, sizeof(PCT) * world * maxChunks, stream));
    using RCT = cuda::std::remove_pointer_t<decltype(ctx.redCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.redCounter, sizeof(RCT) * maxChunks, stream));
    CHECK_CUDA(cudaMemsetAsync(&ctx.redCounter, 0, sizeof(RCT) * maxChunks, stream));
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

    void* gatherSignals = nullptr;
    std::vector<uint64_t*> gatherSignalsV(world);
    {
      const auto signalsPtrBytes = sizeof(decltype(gatherSignalsV)::value_type) * gatherSignalsV.size();
      const auto* base = static_cast<uint64_t*>(nvshmem_calloc(world, sizeof(uint64_t)));
      CHECK_CUDA(cudaMallocAsync(&gatherSignals, signalsPtrBytes, stream));
      for (int i = 0; i < world; ++i) {
        gatherSignalsV[i] = static_cast<uint64_t*>(nvshmem_ptr(base, i));
      }
      CHECK_CUDA(cudaMemcpyAsync(gatherSignals, gatherSignalsV.data(), signalsPtrBytes, cudaMemcpyHostToDevice, stream));
      ctx.gatherSignals = static_cast<uint64_t**>(gatherSignals);
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
    ctx.world_l = cuda::fast_mod_div<size_t, true>{static_cast<size_t>(world)};
    ctx.rank = rank;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return ctx;
  }

  __host__ __forceinline__
  void finalize(const Context& ctx, cudaStream_t stream) {
    CHECK_CUDA(cudaFreeAsync(ctx.epochs, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.putCounter, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.redCounter, stream));
    std::array<void*, 4> heaps{};
    static_assert(sizeof(decltype(ctx.signals + ctx.rank)) == sizeof(void*));
    CHECK_CUDA(cudaMemcpyAsync(heaps.data(), ctx.signals + ctx.rank, sizeof(void*),
      cudaMemcpyDeviceToHost, stream));
    static_assert(sizeof(decltype(ctx.gatherSignals + ctx.rank)) == sizeof(void*));
    CHECK_CUDA(cudaMemcpyAsync(heaps.data() + 1, ctx.gatherSignals + ctx.rank, sizeof(void*),
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