//
// Created by azureuser on 5/1/26.
//
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda/std/cstddef>

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

struct Args {
  cuda::std::byte* p[72];
};
__global__ void kernel(Args args) {
  if (threadIdx.x < 72) {
    atomicAdd(reinterpret_cast<int*>(args.p[threadIdx.x]), static_cast<int>(threadIdx.x));
  }
}

int main() {
  int* p;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaMalloc(&p, 72 * sizeof(int)));
  CHECK_CUDA(cudaMemset(p, 0, 72 * sizeof(int)));
  Args args{};
  for (int i = 0; i < 72; ++i) {
    args.p[i] = reinterpret_cast<cuda::std::byte*>(p + i);
  }
  kernel<<<1, 128>>>(args);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto* v = static_cast<int*>(std::malloc(72 * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(v, p, 72 * sizeof(int), cudaMemcpyDefault));
  CHECK_CUDA(cudaFree(p));
  for (int i = 0; i < 72; ++i) {
    printf("%d\n", v[i]);
  }
  std::free(v);
}