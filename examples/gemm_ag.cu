//
// Created by Osayamen on 1/8/26.
//

#include <cstdio>
#include <nvshmem.h>
#include "gemm_ag.cuh"

// Our implementations:
// 1. Standalone AG + GEMM
// 2. Fused GEMM + AG (tack)
// Baselines:
// 1. NCCL AG + cuBLAS GEMM
// 2. cuBLASMp
// 3. Triton-Dist
// 4. Fused GEMM + AG (NVSHMEM)
int main() {

}