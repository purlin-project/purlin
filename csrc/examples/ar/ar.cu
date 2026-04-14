//
// Created by Osayamen on 3/30/26.
//
// non-atomic allredu
#include <cstdio>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cuda/barrier>
#include <cutlass/array.h>

#include "../debug.cuh"

template<int a>
  struct foo {
  static void work(const int& x, const int* __restrict__ const& y) {
    printf("inside foo<a>: %d, %p\n", x, y);
  }
};

__device__ __forceinline__
void bar(cuda::barrier<cuda::thread_scope_block> (&bars)[4]) {
  static_assert(sizeof(bars) == 4 * sizeof(cuda::barrier<cuda::thread_scope_block>));
  bars[0].arrive_and_wait();
  bars[3].arrive_and_wait();
}

__global__ void fun() {
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bars[4];
  if (threadIdx.x < 4) {
    init(bars + threadIdx.x, 1);
  }
  __syncthreads();
  if (!threadIdx.x) {
    bar(bars);
    printf("Done!\n");
  }
}

int main() {
  fun<<<1,32>>>();
  CHECK_CUDA(cudaDeviceSynchronize());
}