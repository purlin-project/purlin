//
// Created by Osayamen on 3/30/26.
//
#include <unistd.h>

#include <cuda/atomic>
#include <cuda/std/tuple>

#include "../debug.cuh"
#define SLEEP_TIME 500
__global__ void poll_kernel(uint64_t* __restrict__ flags, uint64_t* __restrict__ epochs) {
  if (blockIdx.x == 0) {
    __nanosleep(SLEEP_TIME);
    if (!threadIdx.x) {
      const auto epoch = epochs[blockIdx.x];
      const auto nextEpoch = epoch + 1;
      cuda::atomic_ref f{*flags};
      f.store(nextEpoch, cuda::memory_order_release);
      epochs[blockIdx.x] = nextEpoch;
    }
    __syncthreads();
  }
  else {
    // poll
    if (!threadIdx.x) {
      const auto epoch = epochs[blockIdx.x];
      cuda::atomic_ref f{*flags};
      auto payload = f.load(cuda::memory_order_relaxed);
      auto isDone = payload > epoch;
      while (!isDone) {
        payload = f.load(cuda::memory_order_relaxed);
        isDone = payload > epoch;
      }
      epochs[blockIdx.x] = payload;
      cuda::std::ignore = f.load(cuda::memory_order_acquire);
    }
    __syncthreads();
  }
}

__global__ void poll_kernel1(uint64_t* __restrict__ flags, uint64_t* __restrict__ epochs) {
  if (blockIdx.x == 0) {
    __nanosleep(SLEEP_TIME);
    if (!threadIdx.x) {
      const auto epoch = epochs[blockIdx.x];
      const auto nextEpoch = epoch + 1;
      cuda::atomic_ref f{*flags};
      f.store(nextEpoch, cuda::memory_order_release);
      epochs[blockIdx.x] = nextEpoch;
    }
    __syncthreads();
  }
  else {
    // poll
    if (!threadIdx.x) {
      const auto epoch = epochs[blockIdx.x];
      cuda::atomic_ref f{*flags};
      auto payload = f.load(cuda::memory_order_acquire);
      auto isDone = payload > epoch;
      while (!isDone) {
        payload = f.load(cuda::memory_order_acquire);
        isDone = payload > epoch;
      }
      epochs[blockIdx.x] = payload;
    }
    __syncthreads();
  }
}

__global__ void poll_kernel2(uint64_t* __restrict__ flags, uint64_t* __restrict__ epochs) {
  if (blockIdx.x == 0) {
    const auto epoch = epochs[blockIdx.x];
    const auto nextEpoch = epoch + 1;
    __nanosleep(SLEEP_TIME);
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      cuda::atomic_ref f{*(flags + i)};
      f.store(nextEpoch, cuda::memory_order_release);
    }
    __syncthreads();
    if (!threadIdx.x) {
      epochs[blockIdx.x] = nextEpoch;
    }
  }
  else {
    // poll
    if (!threadIdx.x) {
      const auto epoch = epochs[blockIdx.x];
      cuda::atomic_ref f{*(flags + blockIdx.x)};
      auto payload = f.load(cuda::memory_order_relaxed);
      auto isDone = payload > epoch;
      while (!isDone) {
        payload = f.load(cuda::memory_order_relaxed);
        isDone = payload > epoch;
      }
      cuda::std::ignore = f.load(cuda::memory_order_acquire);
      epochs[blockIdx.x] = payload;
    }
    __syncthreads();
  }
}

__global__ void poll_kernel3(uint64_t* __restrict__ flags, uint64_t* __restrict__ epochs) {
  if (blockIdx.x == 0) {
    const auto epoch = epochs[blockIdx.x];
    const auto nextEpoch = epoch + 1;
    __nanosleep(SLEEP_TIME);
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      cuda::atomic_ref f{*(flags + i)};
      f.store(nextEpoch, cuda::memory_order_release);
    }
    __syncthreads();
    if (!threadIdx.x) {
      epochs[blockIdx.x] = nextEpoch;
    }
  }
  else {
    if (!threadIdx.x) {
      const auto epoch = epochs[blockIdx.x];
      cuda::atomic_ref f{*(flags + blockIdx.x)};
      auto payload = f.load(cuda::memory_order_acquire);
      auto isDone = payload > epoch;
      while (!isDone) {
        payload = f.load(cuda::memory_order_acquire);
        isDone = payload > epoch;
      }
      epochs[blockIdx.x] = payload;
    }
    __barrier_sync_count(0, 256);
  }
}

template<typename Kernel>
__host__
float measure(Kernel& kernel,
  uint64_t* __restrict__ flags, uint64_t* __restrict__ epochs,
  const int& blocks, const int& threads,
  cudaStream_t stream,
  const int& runs, const int& graph_launches,
  cudaEvent_t start, cudaEvent_t stop) {
  const int total_launches = graph_launches * runs;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;

  // capture kernel launches
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < runs; ++i) {
    kernel<<<blocks, threads, 0, stream>>>(flags, epochs);
  }
  CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

  CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // warmup once
  CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < graph_launches; ++i) {
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

  CHECK_CUDA(cudaGraphExecDestroy(graphExec));
  CHECK_CUDA(cudaGraphDestroy(graph));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  sleep(1);

  // per-iteration time (each launch is one iteration)
  return total_ms / static_cast<float>(total_launches);
}

__host__
void drive() {
  constexpr auto minCTAs = 1;
  constexpr auto maxCTAs = 512;
  CHECK_CUDA(cudaSetDevice(0));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  uint64_t* flags = nullptr;
  uint64_t* epochs = nullptr;
  CHECK_CUDA(cudaMallocAsync(&flags, sizeof(uint64_t) * maxCTAs, stream));
  CHECK_CUDA(cudaMallocAsync(&epochs, sizeof(uint64_t) * maxCTAs, stream));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0)); // Get properties for current rank
  printf("blocks,kernel0,kernel1,kernel2,kernel3,GPUName,threads,runs,graph_launches\n");
  fflush(stdout);
  for (int blocks = minCTAs; blocks <= maxCTAs; blocks *= 2) {
    constexpr auto runs = 1024;
    constexpr auto threads = 256;
    constexpr auto graph_launches = 8;

    float t0 = 0.0f;
    float t1 = 0.0f;
    float t2 = 0.0f;
    float t3 = 0.0f;

    t0 = measure(poll_kernel, flags, epochs, blocks, threads, stream, runs, graph_launches, start, stop);
    t1 = measure(poll_kernel1, flags, epochs, blocks, threads, stream, runs, graph_launches, start, stop);
    t2 = measure(poll_kernel2, flags, epochs, blocks, threads, stream, runs, graph_launches, start, stop);
    t3 = measure(poll_kernel3, flags, epochs, blocks, threads, stream, runs, graph_launches, start, stop);
    uint64_t epoch = 0;
    CHECK_CUDA(cudaMemcpyAsync(&epoch, epochs, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

    printf("%d,%lf, %lf, %lf, %lf, %s, %d, %d, %d, %ld\n",
        blocks, t0, t1, t2, t3, prop.name, threads, runs, graph_launches, epoch);
  }
  CHECK_CUDA(cudaFreeAsync(flags, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
}

int main() {
  drive();
}