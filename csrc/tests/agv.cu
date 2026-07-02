//
// Created by osayamen on 6/28/26.
//

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda/cmath>

#include <mpi.h>
#include <matx.h>
#include <nccl.h>
#include <nvshmem.h>

#include <purlin/core.cuh>
#include <purlin/host/allGather.cuh>

#include "util.cuh"

__host__ __forceinline__
auto makeSizes(const size_t bytes, const int world) {
  std::vector<size_t> sizes(world, bytes);
  if (bytes >= static_cast<size_t>(128 * world)) {
    const auto step = size_t{128};
    for (int i = 1; i < world; ++i) {
      sizes[i] -= step;
    }
  }
  return sizes;
}

__host__ __forceinline__
auto makeOffsets(const std::vector<size_t>& sizes) {
  std::vector<size_t> offsets(sizes.size());
  size_t offset = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    offsets[i] = offset;
    offset += sizes[i];
  }
  return offsets;
}

__host__ __forceinline__
auto makeVState(const std::vector<size_t>& sizes, const std::vector<size_t>& offsets, const int rank) {
  return purlin::VState{
    .maxBytes = *std::max_element(sizes.begin(), sizes.end()),
    .totalBytes = std::accumulate(sizes.begin(), sizes.end(), size_t{0}),
    .offset = offsets[rank],
    .bytes = sizes[rank]
  };
}

__host__
void agvHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  cuda::std::byte* dstBuff = nullptr;
  cuda::std::byte* refBuff = nullptr;
  size_t* devSizes = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (world <= 1) {
    printf("Requires at least two processes!\n");
    return;
  }
  if (rank == 0) {
    printf("world,localBytes,maxBytes,totalBytes,purlin(ms),purlin(GB/s),nccl(ms),nccl(GB/s),error_vs_nccl(%%),"
           "GPUName,warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId));

  const auto workspace = makeWorkspace(world, stream);
  auto ctx = purlin::initialize(rank, world, workspace, stream);

  const auto maxSizes = makeSizes(opts.maxLocalBytes, world);
  const auto maxBytes = *std::ranges::max_element(maxSizes.begin(), maxSizes.end());
  const auto maxTotalBytes = std::accumulate(maxSizes.begin(), maxSizes.end(), size_t{0});
  if (maxBytes > ctx.stagingTRSize) {
    throw std::runtime_error("Total allGatherV bytes exceeds staging limit");
  }

  CHECK_CUDA(cudaMallocAsync(&srcBuff, maxBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&dstBuff, maxTotalBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&refBuff, maxTotalBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&devSizes, sizeof(size_t) * world, stream));

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
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    const auto sizes = makeSizes(bytes, world);
    const auto offsets = makeOffsets(sizes);
    ctx.vState = makeVState(sizes, offsets, rank);

    CHECK_CUDA(cudaMemcpyAsync(devSizes, sizes.data(), sizeof(size_t) * world, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(dstBuff, 0, maxTotalBytes, stream));
    CHECK_CUDA(cudaMemsetAsync(refBuff, 0, maxTotalBytes, stream));

    const auto seed = rd();
    static_assert(purlin::MAX_ACCESS_ALIGNMENT % sizeof(float) == 0);
    const auto elems = ctx.vState.bytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(srcBuff);
    randUniform<ARCH>(tS, elems, seed, -1.f, 1.f, stream);

    purlin::allGatherV(srcBuff, dstBuff, devSizes, ctx, stream);

    auto ncclAllGatherV = [&] {
      NCCL_CHECK(ncclGroupStart());
      for (int root = 0; root < world; ++root) {
        if (sizes[root] == 0) {
          continue;
        }
        const auto offset = offsets[root];
        const void* sendBuff = rank == root ? static_cast<const void*>(srcBuff) :
          static_cast<const void*>(refBuff + offset);
        void* recvBuff = refBuff + offset;
        NCCL_CHECK(ncclBroadcast(sendBuff, recvBuff, sizes[root], ncclUint8, root, comm, stream));
      }
      NCCL_CHECK(ncclGroupEnd());
    };

    ncclAllGatherV();

    auto matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(dstBuff),
      {1, static_cast<matx::index_t>(ctx.vState.totalBytes / sizeof(float))});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff),
      {1, static_cast<matx::index_t>(ctx.vState.totalBytes / sizeof(float))});
    (matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < opts.runs; ++i) {
        purlin::allGatherV(srcBuff, dstBuff, devSizes, ctx, stream);
      }
      CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

      CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      const int totalLaunches = opts.runs * opts.graph_launches;
      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < opts.graph_launches; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      t_ms = total_ms / static_cast<float>(totalLaunches);

      CHECK_CUDA(cudaGraphExecDestroy(graphExec));
      CHECK_CUDA(cudaGraphDestroy(graph));
    }
    else {
      for (int i = 0; i < opts.warmup; ++i) {
        purlin::allGatherV(srcBuff, dstBuff, devSizes, ctx, stream);
      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < opts.runs; ++i) {
        purlin::allGatherV(srcBuff, dstBuff, devSizes, ctx, stream);
      }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }

    float nccl_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < opts.runs; ++i) {
        ncclAllGatherV();
      }
      CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

      CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      const int totalLaunches = opts.runs * opts.graph_launches;
      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < opts.graph_launches; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      nccl_ms = total_ms / static_cast<float>(totalLaunches);

      CHECK_CUDA(cudaGraphExecDestroy(graphExec));
      CHECK_CUDA(cudaGraphDestroy(graph));
    }
    else {
      for (int i = 0; i < opts.warmup; ++i) {
        ncclAllGatherV();
      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < opts.runs; ++i) {
        ncclAllGatherV();
      }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&nccl_ms, start, stop));
      nccl_ms /= static_cast<float>(opts.runs);
    }

    double metrics[3] = {
      static_cast<double>(t_ms),
      static_cast<double>(nccl_ms),
      (1.0 - (static_cast<double>(matches()) /
        static_cast<double>(ctx.vState.totalBytes / sizeof(float)))) * 100.0
    };
    MPI_Allreduce(MPI_IN_PLACE, metrics, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = static_cast<double>(ctx.vState.totalBytes) / 1e9;
      const auto purlinAlgBW = gb / (metrics[0] * 1e-3);
      const auto ncclAlgBW = gb / (metrics[1] * 1e-3);
      printf("%d, %lu, %lu, %lu, %lf, %lf, %lf, %lf, %lf, %s, %d, %d, %d\n",
        world, ctx.vState.bytes, ctx.vState.maxBytes, ctx.vState.totalBytes, metrics[0], purlinAlgBW,
        metrics[1], ncclAlgBW, metrics[2], prop.name, opts.graph_launches > 0 ? opts.runs : opts.warmup, opts.runs,
        opts.graph_launches);
    }
  }

  CHECK_CUDA(cudaFreeAsync(srcBuff, stream));
  CHECK_CUDA(cudaFreeAsync(dstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  CHECK_CUDA(cudaFreeAsync(devSizes, stream));
  purlin::finalize(ctx, stream);
  destroyWorkspace(workspace, rank, stream);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}

// NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none mpirun -n <world> ./testAGV <minLocalBytes> <maxLocalBytes> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.runs = 128;
  opts.warmup = 128;
  opts.graph_launches = 8;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.graph_launches = std::stoi(argv[3]);
  if (argc > 4) opts.runs = std::stoi(argv[4]);
  if (argc > 5) opts.warmup = std::stoi(argv[5]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0 ||
    opts.maxLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " +
      std::to_string(purlin::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  agvHost(opts);
}
