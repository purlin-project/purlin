//
// Created by Osayamen on 3/20/26.
//

#ifndef SUTURE_CONSTANTS_CUH
#define SUTURE_CONSTANTS_CUH
namespace suture {
  constexpr int WARP_SIZE = 32;
#if (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 9)
  constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
  constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif

  constexpr int kAlignment = 16;

  constexpr int kThreads = 256;
  constexpr int kPipeStages = 4;
  constexpr int kStageExtent = 4;
  constexpr int kUnrollFactor = 2;

  constexpr auto AG_SUPER_BLOCK_THRESHOLD = 2UL * 1024UL * 1024UL;
  constexpr auto P2P_SUPER_BLOCK_THRESHOLD = 1UL * 1024UL * 1024UL;
  constexpr auto AR_SUPER_BLOCK_THRESHOLD = 1UL * 1024UL * 1024UL;
  constexpr auto AR_LATENCY_BOUND_THRESHOLD = 128UL * 1024UL;
}
#endif //SUTURE_CONSTANTS_CUH