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
#include "../common.cuh"
#include "../debug.cuh"

struct Options {
  size_t minBytes = 128;
  size_t maxBytes = 128 * 1024 * 1024;
  int warmup = 128;
  int runs = 256;
  int graph_launches = 8;
  int maxSuperBlockSize = 8; // try 16 and 32
};

__host__
void agHost(const Options& opts) {
  cuda::std::byte* rcvBuff = nullptr; // [world, size], symmetric
  uint64_t* completions = nullptr; // [ctas], symmetric
  uint64_t* arrivals = nullptr; // [ctas, world], symmetric
  uint64_t* senseBits = nullptr; // [ctas], local

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("world,localBytes,globalBytes,threads,pipeStages,stageExtent,unrollFactor,"
           "totalSMsOnGPU,superBlockSize,blocks,error(%%),warmup,runs,graph_launches,tack(ms),tack(GB/s)\n");
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
  const auto requestedCTAs = opts.maxSuperBlockSize * actualWorld;
  const auto availableCTAs = bps * num_sms;
  const auto superBlockSize0 = requestedCTAs > availableCTAs ?
  (cuda::round_down(availableCTAs, actualWorld) / actualWorld) : opts.maxSuperBlockSize;
  const auto signalLength = world * opts.maxSuperBlockSize;
  completions = static_cast<uint64_t*>(nvshmem_calloc(signalLength, sizeof(uint64_t)));
  arrivals = static_cast<uint64_t*>(nvshmem_calloc(signalLength, sizeof(uint64_t)));
  CHECK_CUDA(cudaMallocAsync(&senseBits, sizeof(uint64_t) * signalLength, stream));
  CHECK_CUDA(cudaMemsetAsync(senseBits, 0, sizeof(uint64_t) * signalLength, stream));
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
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::random_device rd;
  auto agk = [&](const auto& blocks, const AGArgs& kArgs, const int& runs) {
    for (int i = 0; i < runs; ++i) {
      ag<<<blocks, threads, kernelSharedSize, stream>>>(kArgs);
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const cuda::fast_mod_div<int> world_v{world};
  for (size_t bytes = opts.minBytes; bytes <= opts.maxBytes; bytes *= 2) {
    // fill buffer with random values
    const auto seed = rd();
    static_assert(MAX_ACCESS_ALIGNMENT % sizeof(float) == 0);
    const auto elems = bytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(rcvBuff) + (rank * elems);
    randUniform<ARCH>(tS, elems, seed, -1.f, 1.f, stream);
    auto* tSr = reinterpret_cast<float*>(refBuff) + (rank * elems);
    randUniform<ARCH>(tSr, elems, seed, -1.f, 1.f, stream);
    const auto superBlockSize = static_cast<int>(min(cuda::ceil_div(bytes, threads * MAX_ACCESS_ALIGNMENT),
      static_cast<size_t>(superBlockSize0)));
    const size_t scaledChunkSize = bytes / MAX_ACCESS_ALIGNMENT;
    const cuda::fast_mod_div<int> superBlockSize_v{superBlockSize};
    AGArgs args{
      .sendBuff = rcvBuff + (rank * bytes),
      .completions = completions,
      .arrivals = arrivals,
      .senseBits = senseBits,
      .ctaBaseChunk = scaledChunkSize / superBlockSize,
      .superBlockSize_v = superBlockSize_v,
      .world_v = world_v,
      .chunkResidue = static_cast<int>(scaledChunkSize % superBlockSize),
      .maxSuperBlockSize = opts.maxSuperBlockSize,
      .rank = rank,
      .world = world
    };
    const auto blocks = superBlockSize * actualWorld;
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
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      agk(blocks, args, opts.runs);
      CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

      CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // warmup
      CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // time total launches = opts.runs * opts.graph_launches
      const int total_launches = opts.runs * opts.graph_launches;

      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < opts.graph_launches; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

      // per-iteration time (each launch is one iteration)
      t_ms = total_ms / static_cast<float>(total_launches);

      CHECK_CUDA(cudaGraphExecDestroy(graphExec));
      CHECK_CUDA(cudaGraphDestroy(graph));
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
      printf("%d, %lu, %lu, %d, %d, %d, %d, %d, %d, %d, %lf, %d, %d, %d, %lf, %lf\n",
        world, bytes, world * bytes, threads, pipeStages, stageExtent, unrollFactor,
        num_sms, superBlockSize, blocks, times.ep, opts.warmup, opts.runs,opts.graph_launches, times.t_ms, tack_algBW);
    }
  }
  CHECK_CUDA(cudaFreeAsync(senseBits, stream));
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
// ./ag <minBytes> <maxBytes> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  Options opts{};
  if (argc > 1) opts.minBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxBytes = parseSize(argv[2]);
  if (argc > 3) opts.graph_launches = std::stoi(argv[3]);
  if (argc > 4) opts.runs = std::stoi(argv[4]);
  if (argc > 5) opts.warmup = std::stoi(argv[5]);
  if (argc > 6) opts.maxSuperBlockSize = std::stoi(argv[6]);
  if (!cuda::is_power_of_two(opts.minBytes) || !cuda::is_power_of_two(opts.maxBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minBytes % MAX_ACCESS_ALIGNMENT != 0 || opts.maxBytes % MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  agHost(opts);
}
