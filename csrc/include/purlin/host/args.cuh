//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ARGS_CUH
#define PURLIN_ARGS_CUH
#include <mutex>
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
  static constexpr auto MAX_LR_BLOCKS = 64;
  static constexpr auto SMEM_ALIGNMENT = 128;
  struct Args {
    const cuda::std::byte* const src;
    cuda::std::byte* const dst;
    const size_t bytes = 0;
    const cuda::fast_mod_div<long int> blocks;
  };
  template<int threads>
  __host__ __forceinline__
  auto getLRBlocks(const size_t& bytes) {
    return static_cast<int>(cuda::std::min(cuda::ceil_div(bytes, threads * sizeof(LRP::RT)), static_cast<size_t>(MAX_LR_BLOCKS)));
  }

  // Consumer-block count for the staged throughput collectives: enough blocks to
  // keep the reduce pipeline fed, capped by maxBlocks, with a flat split when the
  // payload is too small to pipeline.
  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getTRBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks,
    const int& world) {
    auto blocksNeeded = cuda::std::min(bytes / PurlinAtom::RED_PIPELINE_BYTES,
      bytes / (world * PurlinAtom::STAGE_BYTES));
    blocksNeeded = cuda::std::min(blocksNeeded, static_cast<size_t>(maxBlocks));
    if (blocksNeeded < 1) {
      // non-pipelined path
      return putBlocks + static_cast<int>(cuda::std::min(cuda::ceil_div(bytes / world,
        static_cast<size_t>(PurlinAtom::THREADS) * PurlinAtom::BaseConfig::ALIGNMENT_BYTES),
        static_cast<size_t>(maxBlocks)));
    }
    return putBlocks + static_cast<int>(blocksNeeded);
  }

  template <auto Kernel, int smem>
  __host__ __forceinline__
  void ensureOptIn() {
    static std::once_flag flag;
    std::call_once(flag, [] {
      CHECK_CUDA(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    });
  }
}
#endif //PURLIN_ARGS_CUH
