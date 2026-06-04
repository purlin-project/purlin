//
// Created by osayamen on 5/31/26.
//
#include <random>
#include <string>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>

#include <purlin/core.cuh>
#include <purlin/host/all2all.cuh>

#include "util.cuh"

__host__ __forceinline__
void all2allReference(const cuda::std::byte* src,
  cuda::std::byte* dst,
  const size_t bytes,
  const int rank,
  const int world,
  ncclComm_t comm,
  cudaStream_t stream) {
  CHECK_CUDA(cudaMemcpyAsync(dst + rank * bytes, src + rank * bytes, bytes, cudaMemcpyDeviceToDevice, stream));
  NCCL_CHECK(ncclGroupStart());
  for (int peer = 0; peer < world; ++peer) {
    if (peer == rank) {
      continue;
    }
    NCCL_CHECK(ncclSend(src + peer * bytes, bytes, ncclUint8, peer, comm, stream));
    NCCL_CHECK(ncclRecv(dst + peer * bytes, bytes, ncclUint8, peer, comm, stream));
  }
  NCCL_CHECK(ncclGroupEnd());
}

__host__
void a2aHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  cuda::std::byte* dstBuff = nullptr;
  cuda::std::byte* refBuff = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (world <= 1) {
    printf("Requires at least two processes!\n");
  }
  if (rank == 0) {
    printf("world,localBytes,globalBytes,purlin(ms),purlin(GB/s),error(%%),GPUName,warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  auto ctx = purlin::initialize(rank, world, stream);

  CHECK_CUDA(cudaMallocAsync(&srcBuff, opts.maxLocalBytes * world, stream));
  CHECK_CUDA(cudaMallocAsync(&dstBuff, opts.maxLocalBytes * world, stream));
  CHECK_CUDA(cudaMallocAsync(&refBuff, opts.maxLocalBytes * world, stream));
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
  matx::cudaExecutor exec{stream};
  Times times{};
  for (size_t localBytes = opts.minLocalBytes; localBytes <= opts.maxLocalBytes; localBytes *= 2) {
    // fill buffer with random values
    const auto seed = rd();
    const auto elems = (localBytes * world) / sizeof(float);
    auto* tS = reinterpret_cast<float*>(srcBuff);
    randUniform<ARCH>(tS, elems, seed, -1.f, 1.f, stream);
    // correctness run
    purlin::all2all(srcBuff, dstBuff, localBytes, ctx, stream);
    all2allReference(srcBuff, refBuff, localBytes, rank, world, comm, stream);
    auto a2a_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(dstBuff), {1, static_cast<matx::index_t>(elems)});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff), {1, static_cast<matx::index_t>(elems)});
    // bitwise check
    (a2a_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);

    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < opts.runs; ++i) {
        purlin::all2all(srcBuff, dstBuff, localBytes, ctx, stream);
      }
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
      // benchmark purlin without graphs
      for (int i = 0; i < opts.warmup; ++i) {
        purlin::all2all(srcBuff, dstBuff, localBytes, ctx, stream);
      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      for (int i = 0; i < opts.runs; ++i) {
        purlin::all2all(srcBuff, dstBuff, localBytes, ctx, stream);
      }
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }

    times.ep = (1.0 - (static_cast<double>(a2a_matches()) / static_cast<double>(tR.TotalSize()))) * 100.0;
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = (world * static_cast<double>(localBytes)) / 1e9;
      const auto purlin_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %lu, %lf, %lf, %lf, %s, %d, %d, %d\n",
        world, localBytes, world * localBytes,times.t_ms, purlin_algBW, times.ep,
        prop.name, opts.graph_launches > 0 ? opts.runs : opts.warmup, opts.runs,opts.graph_launches);
    }
  }
  CHECK_CUDA(cudaFreeAsync(srcBuff, stream));
  CHECK_CUDA(cudaFreeAsync(dstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  purlin::finalize(ctx, stream);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}
// NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_BOOTSTRAP=MPI mpirun -n <world> ./a2a <minLocalBytes> <maxLocalBytes> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.graph_launches = 8;
  opts.warmup = 128;
  opts.runs = 128;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.graph_launches = std::stoi(argv[3]);
  if (argc > 4) opts.runs = std::stoi(argv[4]);
  if (argc > 5) opts.warmup = std::stoi(argv[5]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0 || opts.maxLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(purlin::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  a2aHost(opts);
}