//
// Created by Osayamen on 1/8/26.
//

#include <cstdio>
#include <nvshmem.h>
#include <nccl.h>
#include <mpi.h>
#include "gemm_ag.cuh"

// Our implementations:
// 1. Standalone AG + GEMM
// 2. Fused GEMM + AG (tack)
// Baselines:
// 1. NCCL AG + cuBLAS GEMM
// 2. cuBLASMp
// 3. Triton-Dist
// 4. Fused GEMM + AG (NVSHMEM)

__host__ __forceinline__
void kickStart(const int& M, const int& N, const int& K) {

}
int main() {

}