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
  struct Args {
    const cuda::std::byte* const src;
    cuda::std::byte* const dst;
    const size_t bytes;
    const cuda::fast_mod_div<long int> blocks;
  };
  template<int threads>
  __host__ __forceinline__
  auto getLRBlocks(const size_t& bytes) {
    return static_cast<int>(cute::min(cuda::ceil_div(bytes, threads * sizeof(LRP16::RT)), MAX_LR_BLOCKS));
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
