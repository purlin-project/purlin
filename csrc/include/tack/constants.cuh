//
// Created by Osayamen on 3/20/26.
//

#ifndef TACK_CONSTANTS_CUH
#define TACK_CONSTANTS_CUH
namespace tack {
  constexpr int WARP_SIZE = 32;
#if (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 9)
  constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
  constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif

  constexpr int threads = 64;
  constexpr int Alignment = 16;
  constexpr int pipeStages = 4;
  constexpr int stageExtent = 32;
  constexpr int unrollFactor = 2;

  constexpr auto AG_SUPER_BLOCK_THRESHOLD = 2UL * 1024UL * 1024UL;
  constexpr auto P2P_SUPER_BLOCK_THRESHOLD = 1UL * 1024UL * 1024UL;
}
#endif //TACK_CONSTANTS_CUH