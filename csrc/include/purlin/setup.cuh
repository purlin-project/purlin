//
// Created by Osayamen on 4/16/26.
//

#ifndef PURLIN_SETUP_CUH
#define PURLIN_SETUP_CUH
#include <stdexcept>
#include "constants.cuh"
#include "context.cuh"

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
namespace purlin {
  __host__ __forceinline__
  auto initialize(const int& rank, const int& world, const WorkspaceMemory& w, cudaStream_t stream) {
    Context ctx{};
    if (world <= 1 || world > MAX_RANKS_PER_DOMAIN) {
      const auto errmsg = "world: " + std::to_string(world) + " is invalid";
      throw std::runtime_error(errmsg);
    }
    using ET = cuda::std::remove_pointer_t<decltype(ctx.epochs)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.epochs, sizeof(ET) * purlin::MAX_NUM_CTAS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.epochs, 0, sizeof(ET) * purlin::MAX_NUM_CTAS, stream));
    using PCT = cuda::std::remove_pointer_t<decltype(ctx.putCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.putCounter, sizeof(PCT) * world * MAX_CHUNKS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.putCounter, 0, sizeof(PCT) * world * MAX_CHUNKS, stream));
    using RCT = cuda::std::remove_pointer_t<decltype(ctx.redCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.redCounter, sizeof(RCT) * MAX_CHUNKS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.redCounter, 0, sizeof(RCT) * MAX_CHUNKS, stream));

    ctx.staging = w.stagingTR;
    ctx.stagingLR = w.stagingLR;
    ctx.signals = w.signals;
    ctx.gatherSignals = w.gatherSignals;
    ctx.world = cuda::fast_mod_div<int, true>{world};
    ctx.actualWorld = cuda::fast_mod_div<int>{(world - 1)};
    ctx.world_l = cuda::fast_mod_div<size_t, true>{static_cast<size_t>(world)};
    ctx.rank = rank;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return ctx;
  }
  // for Python bindings
  __host__ __forceinline__
  auto initialize(const int& rank, const int& world,
    cuda::std::byte** const& stagingLR,
    cuda::std::byte** const& stagingTR,
    uint64_t** signals,
    uint64_t** gatherSignals,
    cudaStream_t stream) {
    const WorkspaceMemory w{
      .stagingLR = stagingLR,
      .stagingTR = stagingTR,
      .signals = signals,
      .gatherSignals = gatherSignals
    };
    return initialize(rank, world, w, stream);
  }

  template<typename T>
  __host__ __forceinline__
  auto splitPointerTable(T** const& base, const size_t& offset, const int& world, cudaStream_t stream) {
    void* mem = nullptr;
    std::vector<T*> p(world);
    std::vector<T*> q(world);
    const auto pB = sizeof(typename decltype(p)::value_type) * p.size();
    CHECK_CUDA(cudaMallocAsync(&mem, pB, stream));
    CHECK_CUDA(cudaMemcpyAsync(p.data(), base, pB, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < world; ++i) {
      q[i] = p[i] + offset;
    }
    CHECK_CUDA(cudaMemcpyAsync(mem, q.data(), pB, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return static_cast<T**>(mem);
  }

  __host__ __forceinline__
  void finalize(const Context& ctx, cudaStream_t stream) {
    CHECK_CUDA(cudaFreeAsync(ctx.epochs, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.putCounter, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.redCounter, stream));
  }
}
#endif //PURLIN_SETUP_CUH