#ifndef PURLIN_SUPPORT_BENCHMARK_CHECKS_CUH
#define PURLIN_SUPPORT_BENCHMARK_CHECKS_CUH

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#if !defined(CHECK_CUDA)
#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    const cudaError_t status_ = (call);                                          \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d for %s: %s (%s)\n",           \
        __FILE__, __LINE__, #call, cudaGetErrorName(status_),                    \
        cudaGetErrorString(status_));                                            \
      std::fflush(stderr);                                                       \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)
#endif

#if !defined(NCCL_CHECK)
#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    const ncclResult_t status_ = (call);                                         \
    if (status_ != ncclSuccess) {                                                \
      std::fprintf(stderr, "NCCL error at %s:%d for %s: %s\n",                \
        __FILE__, __LINE__, #call, ncclGetErrorString(status_));                  \
      std::fflush(stderr);                                                       \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)
#endif

#if !defined(MPI_CHECK)
#define MPI_CHECK(call)                                                         \
  do {                                                                          \
    const int status_ = (call);                                                  \
    if (status_ != MPI_SUCCESS) {                                                \
      char message_[MPI_MAX_ERROR_STRING]{};                                     \
      int length_ = 0;                                                           \
      MPI_Error_string(status_, message_, &length_);                             \
      std::fprintf(stderr, "MPI error at %s:%d for %s: %.*s\n",               \
        __FILE__, __LINE__, #call, length_, message_);                           \
      std::fflush(stderr);                                                       \
      MPI_Abort(MPI_COMM_WORLD, status_);                                        \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)
#endif

#endif
