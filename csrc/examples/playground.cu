//
// Created by azureuser on 5/1/26.
//
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda/std/cstddef>
#include <cuda/barrier>

#include <suture/host/telemetry.cuh>

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

__global__ void kernel() {
  extern __shared__ cuda::std::byte workspace[];
  auto* __restrict__ b = reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>(workspace);
  if (!threadIdx.x) {
    init(b, blockDim.x);
  }
  auto& v = *b;
  const auto p = cuda::device::barrier_native_handle(v);
  v.arrive_and_wait();
  if (!threadIdx.x) {
    printf("bar is %p, w is %p\n", p, reinterpret_cast<uint64_t*>(b));
  }
  __syncthreads();
}

__host__ __forceinline__
void foo(const size_t& bytes) {
  const suture::SutureRange range{"suture::foo", nvtx3::payload{static_cast<uint64_t>(bytes)}};
  kernel<<<1, 128, 64>>>();
  CHECK_CUDA(cudaDeviceSynchronize());
}
int main() {
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
  foo(1024);
}