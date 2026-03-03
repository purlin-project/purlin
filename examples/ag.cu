//
// Created by osayamen on 2/18/26.
//
#include <random>
#include <string>
#include <vector>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>

#include "ag.cuh"
#include "common.cuh"
#include "debug.cuh"

struct Options {
  size_t minBytes = 128;
  size_t maxBytes = 128 * 1024 * 1024;
  int warmup = 128;
  int runs = 256;
  int graph_launches = 2;
};

struct Times {
  double t_ms;
  double ep;
};

struct AGGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;

  // must stay alive as long as we launch the exec because kernelParams points to them.
  std::vector<AGArgs> nodeArgs;
  std::vector<std::array<void*, 1>> kernelParams;
};

static __host__ __forceinline__
AGGraph BuildAGGraph(const uint& blocks, const size_t& kernelSharedSize, const AGArgs& baseArgs, const int& iters)
{
  AGGraph g{};
  CHECK_CUDA(cudaGraphCreate(&g.graph, 0));

  g.nodeArgs.resize(iters);
  g.kernelParams.resize(iters);

  cudaGraphNode_t prev = nullptr;

  for (int i = 0; i < iters; ++i) {
    g.nodeArgs[i] = baseArgs;
    g.nodeArgs[i].signal = baseArgs.signal + static_cast<uint64_t>(i);

    g.kernelParams[i][0] = static_cast<void*>(&g.nodeArgs[i]);

    cudaKernelNodeParams kparams{};
    kparams.func = (void*)ag;
    kparams.gridDim = dim3(blocks, 1, 1);
    kparams.blockDim = dim3(threads, 1, 1);
    kparams.sharedMemBytes = kernelSharedSize;
    kparams.kernelParams = static_cast<void**>(g.kernelParams[i].data());
    kparams.extra = nullptr;

    cudaGraphNode_t node = nullptr;
    if (prev) {
      CHECK_CUDA(cudaGraphAddKernelNode(&node, g.graph, &prev, 1, &kparams));
    } else {
      CHECK_CUDA(cudaGraphAddKernelNode(&node, g.graph, nullptr, 0, &kparams));
    }
    prev = node;
  }

  CHECK_CUDA(cudaGraphInstantiate(&g.exec, g.graph, nullptr, nullptr, 0));
  return g;
}

static void DestroyAGGraph(AGGraph& g) {
  if (g.exec)  CHECK_CUDA(cudaGraphExecDestroy(g.exec));
  if (g.graph) CHECK_CUDA(cudaGraphDestroy(g.graph));
  g.exec = nullptr;
  g.graph = nullptr;
  g.nodeArgs.clear();
  g.kernelParams.clear();
}

__host__
void agHost(const Options& opts) {
  cuda::std::byte* rcvBuff = nullptr; // [world, size], symmetric
  uint64_t* completions = nullptr; // [ctas], symmetric
  uint64_t* arrivals = nullptr; // [ctas, world], symmetric

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("world,localBytes,globalBytes,threads,pipeStages,stageExtent,unrollFactor,"
           "SMs,blocks,error(%%),warmup,runs,graph_launches,tack(ms),tack(GB/s)\n");
    if (world <= 1) {
      printf("pass\n");
      return;
    }
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  auto kernel = ag;
  constexpr auto kernelSharedSize = threads * Alignment * pipeStages * stageExtent;
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  if (kernelSharedSize > maxSharedMemory) {
    const auto errmsg = std::string("Required shared memory ").append(std::to_string(kernelSharedSize))
    .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
    throw std::runtime_error(errmsg);
  }
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSharedSize));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, kernelSharedSize));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  const auto actualWorld = world - 1;
  const auto blocksUpper = min(cuda::ceil_div(opts.maxBytes, threads * Alignment) * actualWorld, static_cast<size_t>(num_sms * bps));
  completions = static_cast<uint64_t*>(nvshmem_calloc(blocksUpper, sizeof(uint64_t)));
  arrivals = static_cast<uint64_t*>(nvshmem_calloc(blocksUpper * world, sizeof(uint64_t)));
  rcvBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxBytes * world));
  auto* refBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxBytes * world));
  if (rcvBuff == nullptr || !cuda::is_aligned(rcvBuff, MAX_ACCESS_ALIGNMENT)) {
    throw std::runtime_error("rcvBuff is invalid");
  }
  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.minCTAs = blocksUpper;
  config.maxCTAs = blocksUpper;
  NCCL_CHECK(ncclCommInitRankConfig(&comm, world, id, rank, &config));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::random_device rd;
  AGArgs args{
    .sendBuff = nullptr,
    .completions = completions,
    .arrivals = arrivals,
    .signal = 1,
    .size = opts.minBytes,
    .rank = rank,
    .world = world
  };
  auto agk = [&](const auto& blocks, AGArgs& kArgs, const int& runs) {
    for (int i = 0; i < runs; ++i) {
      ag<<<blocks, threads, kernelSharedSize, stream>>>(kArgs);
      kArgs.signal += 1;
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  for (size_t bytes = opts.minBytes; bytes <= opts.maxBytes; bytes *= 2) {
    // fill buffer with random values
    const auto seed = rd();
    static_assert(MAX_ACCESS_ALIGNMENT % sizeof(float) == 0);
    const auto elems = bytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(rcvBuff) + (rank * elems);
    randUniform<ARCH>(tS, elems, seed, -1.f, 1.f, stream);
    auto* tSr = reinterpret_cast<float*>(refBuff) + (rank * elems);
    randUniform<ARCH>(tSr, elems, seed, -1.f, 1.f, stream);
    args.size = bytes;
    args.sendBuff = rcvBuff + (rank * bytes);
    const auto blocks = static_cast<uint>(min(cuda::ceil_div(bytes, threads * Alignment) * actualWorld,
      static_cast<size_t>(num_sms * bps)));
    // correctness run
    agk(blocks, args, 1);
    auto* sB = refBuff + (rank * bytes);
    ncclAllGather(sB, refBuff, bytes, ncclUint8, comm, stream);
    auto ag_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(rcvBuff), {1, static_cast<matx::index_t>(elems * world)});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff), {1, static_cast<matx::index_t>(elems * world)});
    // bitwise check
    (ag_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);

    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      // --- benchmark tack via CUDA Graph ---
      // Build warmup graph (opts.warmup kernel nodes) using current args as the base.
      AGGraph warmG = BuildAGGraph(blocks, kernelSharedSize, args, opts.warmup);

      // Warmup: launch warmup graph once
      CHECK_CUDA(cudaGraphLaunch(warmG.exec, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // Advance signal to reflect the kernels executed in warmup graph
      args.signal += static_cast<uint64_t>(opts.warmup);

      // Build benchmark graph (opts.runs kernel nodes) using updated base signal.
      AGGraph benchG = BuildAGGraph(blocks, kernelSharedSize, args, opts.runs);

      // Time N launches of the *graph*
      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int j = 0; j < opts.graph_launches; ++j) {
        CHECK_CUDA(cudaGraphLaunch(benchG.exec, stream));
      }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

      // Average per graph launch, and per kernel-iteration (to match your old output meaning)
      const float avg_graph_ms = total_ms / static_cast<float>(opts.graph_launches);
      const float avg_iter_ms  = avg_graph_ms / static_cast<float>(opts.runs);

      t_ms = avg_iter_ms;

      // Advance signal to reflect all kernels executed in benchmark
      args.signal += static_cast<uint64_t>(opts.runs) * static_cast<uint64_t>(opts.graph_launches);

      // Cleanup graphs
      DestroyAGGraph(warmG);
      DestroyAGGraph(benchG);
    }
    else {
      // benchmark tack without graphs
      agk(blocks, args, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      agk(blocks, args, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }

    times.ep = 1.0 - (static_cast<double>(ag_matches()) / static_cast<double>(tR.TotalSize()));
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = (world * static_cast<double>(bytes)) / 1e9;
      const auto tack_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %lu, %d, %d, %d, %d, %d, %d, %lf, %d, %d, %d, %lf, %lf\n",
        world, bytes, world * bytes, threads, pipeStages, stageExtent, unrollFactor,
        num_sms, blocks, times.ep, opts.warmup, opts.runs,opts.graph_launches, times.t_ms, tack_algBW);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_free(arrivals);
  nvshmem_free(completions);
  nvshmem_free(rcvBuff);
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}
// ./ag <minBytes> <maxBytes> <warmup> <runs>
int main(const int argc, char** argv) {
  Options opts{};
  if (argc > 1) opts.minBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxBytes = parseSize(argv[2]);
  if (argc > 3) opts.warmup = std::stoi(argv[3]);
  if (argc > 4) opts.runs = std::stoi(argv[4]);
  if (argc > 5) opts.graph_launches = std::stoi(argv[5]);
  if (!cuda::is_power_of_two(opts.minBytes) || !cuda::is_power_of_two(opts.maxBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minBytes % MAX_ACCESS_ALIGNMENT != 0 || opts.maxBytes % MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  agHost(opts);
}
