//
// Created by Osayamen on 1/8/26.
//

#ifndef suture_DEBUG_CUH
#define suture_DEBUG_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define NCCL_CHECK(call)                                   \
    do                                                     \
    {                                                      \
        ncclResult_t status = call;                        \
        if (status != ncclSuccess)                         \
        {                                                  \
            fprintf(stderr, "NCCL error at %s:%d : %d\n",  \
            __FILE__, __LINE__, status);                   \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

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
#endif //suture_DEBUG_CUH
