//
// Created by Osayamen on 3/9/26.
//
#include <random>

#include <matx.h>
#include <mpi.h>
#include <cuda/cmath>
#include <nvshmem.h>

#include "../common.cuh"
#include "../constants.cuh"
#include "../debug.cuh"

#include "p2p.cuh"

struct Options {
  size_t minLocalBytes = 128;
  size_t maxLocalBytes = 128 * 1024 * 1024;
  int warmup = 128;
  int runs = 256;
  int graph_launches = 8;
  int blocks = 32;
};

__host__
void p2pHost(const Options& opts) {
  cuda::std::byte* srcBuf = nullptr; // local
  cuda::std::byte* dstBuf = nullptr; // symmetric
  nvshmem_init();

  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

  if (world != 2) {
    if (rank == 0) {
      printf("Two processes required\n");
    }
    return;
  }
  if (rank == 0) {
    printf("bytes,tack(ms),tack(GB/s),error(%%),threads,pipeStages,stageExtent,unrollFactor,blocksPerSM,"
           "SMsOnGPU,blocks,warmup,runs,graph_launches\n");
    fflush(stdout);
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  CHECK_CUDA(cudaMallocAsync(&srcBuf, opts.maxLocalBytes, stream));
  auto kernel = p2pK;
  dstBuf = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxLocalBytes));
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
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::random_device rd;
  auto pk = [&kernelSharedSize, &stream](const auto& blocks, const P2PArgs& kArgs, const int& runs = 1) {
    for (int i = 0; i < runs; ++i) {
      p2pK<<<blocks, threads, kernelSharedSize, stream>>>(kArgs);
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const auto peer = rank == 0 ? 1 : 0;
  auto* translatedBuf = static_cast<cuda::std::byte*>(nvshmem_ptr(dstBuf, peer));
  CHECK_CUDA(cudaPeekAtLastError());
  //auto* translatedBuf = dstBuf;
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    uint seed;
    if (rank == 0) {
      seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // fill buffer with random values
    const auto mySeed = seed + rank;
    static_assert(MAX_ACCESS_ALIGNMENT % sizeof(float) == 0);
    const auto elems = bytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(srcBuf);
    randUniform<ARCH>(tS, elems, mySeed, -1.f, 1.f, stream);
    const size_t scaledChunkSize = bytes / MAX_ACCESS_ALIGNMENT;
    const P2PArgs args{
      .srcBuf = srcBuf,
      .dstBuf = translatedBuf,
      .ctaBaseChunk = scaledChunkSize / opts.blocks,
      .chunkResidue = static_cast<uint>(scaledChunkSize % opts.blocks),
      .rank = rank,
      .peer = peer
    };
    nvshmemx_sync_all_on_stream(stream); // ensures the buffer is available
    pk(opts.blocks, args);
    CHECK_CUDA(cudaPeekAtLastError());
    nvshmemx_barrier_all_on_stream(stream); // ensures we have received the peer's payload
    // check correctness
    const auto expectedSeed = seed + peer;
    randUniform<ARCH>(tS, elems, expectedSeed, -1.f, 1.f, stream);
    auto p2p_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(dstBuf), {1, static_cast<matx::index_t>(elems)});
    auto tRef = matx::make_tensor<float>(tS, {1, static_cast<matx::index_t>(elems)});
    // bitwise check
    (p2p_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);
    nvshmemx_sync_all_on_stream(stream); // ensures we complete the correctness checks before subsequent transfers
    // benchmark tack p2p
    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      pk(opts.blocks, args, opts.runs);
      CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

      CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // warmup once
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
      pk(opts.blocks, args, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      pk(opts.blocks, args, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }
    times.ep = 1.0 - (static_cast<double>(p2p_matches()) / static_cast<double>(tR.TotalSize()));
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = static_cast<double>(bytes) / 1e9;
      const auto tack_algBW = gb / (times.t_ms * 1e-3);
      printf("%lu,%lf, %lf, %lf, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
        bytes,times.t_ms, tack_algBW, times.ep,threads, pipeStages, stageExtent, unrollFactor, bps,
        num_sms, opts.blocks, opts.graph_launches > 0 ? opts.runs : opts.warmup, opts.runs, opts.graph_launches);
    }
  }
  // 7) Synchronize / cleanup
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
}

// ./p2p <minBytes> <maxBytes> <blocks> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  Options opts{};
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.blocks = std::stoi(argv[3]);
  if (argc > 4) opts.graph_launches = std::stoi(argv[4]);
  if (argc > 5) opts.runs = std::stoi(argv[5]);
  if (argc > 6) opts.warmup = std::stoi(argv[6]);
  if (opts.blocks <= 0) {
    throw std::invalid_argument("blocks must be greater than zero");
  }
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % MAX_ACCESS_ALIGNMENT != 0 || opts.maxLocalBytes % MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  p2pHost(opts);
}