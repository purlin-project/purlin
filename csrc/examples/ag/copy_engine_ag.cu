//
// Created by osayamen on 3/3/26.
//
#include <random>
#include <string>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>
#include <nvshmem.h>

#include "../common.cuh"
#include "../debug.cuh"

// baseline AG using the copy engine
struct Options {
  size_t minLocalBytes = 128;
  size_t maxLocalBytes = 128 * 1024 * 1024;
  int warmup = 128;
  int runs = 256;
  int graph_launches = 2;
};

__host__
void agHost(const Options& opts) {
  cuda::std::byte* rcvBuff = nullptr; // [world, size], symmetric

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("world,localBytes,globalBytes,error(%%),warmup,runs,graph_launches,ce(ms),ce(GB/s)\n");
    if (world <= 1) {
      printf("pass\n");
      return;
    }
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  rcvBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxLocalBytes * world));
  auto* refBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxLocalBytes * world));
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
  auto agk = [&](cuda::std::byte* __restrict__ const& rb, const size_t& bytes, const int& runs) {
    const auto* src = rb + (rank * bytes);
    for (int i = 0; i < runs; ++i) {
      nvshmemx_sync_all_on_stream(stream); // arrival rendezvous before initiating the call
      for (int j = 1; j < world; ++j) {
        const int peer = (rank + j) % world;
        auto* dst = nvshmem_ptr(src, peer);
        CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
      }
      nvshmemx_sync_all_on_stream(stream); // ensures collective is complete and the result is available after this call
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    // fill buffer with random values
    const auto seed = rd();
    const auto elems = bytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(rcvBuff) + (rank * elems);
    randUniform<ARCH>(tS, elems, seed, -1.f, 1.f, stream);
    auto* tSr = reinterpret_cast<float*>(refBuff) + (rank * elems);
    randUniform<ARCH>(tSr, elems, seed, -1.f, 1.f, stream);
    // correctness run
    agk(rcvBuff, bytes, 1);
    auto* sB = refBuff + (rank * bytes);
    ncclAllGather(sB, refBuff, bytes, ncclUint8, comm, stream);
    auto ag_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(rcvBuff), {1, static_cast<matx::index_t>(elems * world)});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff), {1, static_cast<matx::index_t>(elems * world)});
    // bitwise check
    (ag_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);

    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      // benchmark with graphs
      // benchmark with graphs: capture one graph that performs opts.runs "iterations"
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // 1) Capture the exact work of one benchmark run (opts.runs iterations)
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      agk(rcvBuff, bytes, opts.runs);
      CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

      // Instantiate
      CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

      // 2) Warmup
      // Option A: warm up by launching the graph once
      CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // 3) Time N graph launches
      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < opts.graph_launches; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

      const float avg_graph_ms = total_ms / static_cast<float>(opts.graph_launches);
      t_ms = avg_graph_ms / static_cast<float>(opts.runs); // per-iteration time (matches your old output)

      // 4) Cleanup
      CHECK_CUDA(cudaGraphExecDestroy(graphExec));
      CHECK_CUDA(cudaGraphDestroy(graph));
    }
    else {
      // benchmark without graphs
      agk(rcvBuff, bytes, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      agk(rcvBuff, bytes, opts.runs);
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
      printf("%d, %lu, %lu, %lf, %d, %d, %d, %lf, %lf\n",
        world, bytes, world * bytes, times.ep, opts.warmup, opts.runs,opts.graph_launches, times.t_ms, tack_algBW);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_free(rcvBuff);
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}
// ./ce_ag <minBytes> <maxBytes> <warmup> <runs> <graph_launches>
int main(const int argc, char** argv) {
  Options opts{};
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.warmup = std::stoi(argv[3]);
  if (argc > 4) opts.runs = std::stoi(argv[4]);
  if (argc > 5) opts.graph_launches = std::stoi(argv[5]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  agHost(opts);
}
