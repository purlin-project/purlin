//
// Created by azureuser on 3/9/26.
//

#ifndef TACK_CONSTANTS_CUH
#define TACK_CONSTANTS_CUH
constexpr int WARP_SIZE = 32;
#if (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 9)
constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif

constexpr int threads = 256;
constexpr int Alignment = 16;
constexpr int pipeStages = 4;
constexpr int stageExtent = 4;
constexpr int unrollFactor = 2;
#endif //TACK_CONSTANTS_CUH