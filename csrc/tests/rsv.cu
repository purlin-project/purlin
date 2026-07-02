//
// Created by osayamen on 5/31/26.
//
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nvshmem.h>

#include <purlin/core.cuh>
#include <purlin/host/reduceScatter.cuh>

#include "util.cuh"

using DataType = __half;

template<typename Element>
__global__ void rsvReferenceKernel(const Element* const* __restrict__ sources,
  Element* __restrict__ dstBuff, const int world, const size_t elems) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= elems) {
    return;
  }
  using AccumType = cuda::std::common_type_t<Element, float>;
  auto accumulator = static_cast<AccumType>(0.f);
  for (int i = 0; i < world; ++i) {
    constexpr Converter<AccumType, Element> loadConv{};
    accumulator += loadConv(sources[i][tid]);
  }
  constexpr Converter<Element, AccumType> storeConv{};
  dstBuff[tid] = storeConv(accumulator);
}

__host__ __forceinline__
auto makeSizes(const size_t bytes, const int world) {
  std::vector<size_t> sizes(world, bytes);
  const auto step = bytes >= static_cast<size_t>(128 * world) ? size_t{128} : size_t{0};
  for (int i = 1; i < world; ++i) {
    sizes[0] -= step;
    sizes[i] += step;
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
void rsvHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  cuda::std::byte* dstBuff = nullptr;
  DataType* refBuff = nullptr;
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
    printf("world,localBytes,maxBytes,totalBytes,datatype,purlin(ms),purlin(GB/s),error_vs_oracle(%%),"
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
  const auto maxTotalBytes = std::accumulate(maxSizes.begin(), maxSizes.end(), size_t{0});
  if (maxTotalBytes > ctx.stagingTRSize) {
    throw std::runtime_error("Total reduceScatterV bytes exceeds staging limit");
  }

  CHECK_CUDA(cudaMallocAsync(&srcBuff, maxTotalBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&dstBuff, maxSizes[rank], stream));
  CHECK_CUDA(cudaMallocAsync(&refBuff, maxSizes[rank], stream));
  CHECK_CUDA(cudaMallocAsync(&devSizes, sizeof(size_t) * world, stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<cuda::std::byte*> dataBuffs(world, nullptr);
  for (int i = 0; i < world; ++i) {
    if (i != rank) {
      CHECK_CUDA(cudaMallocAsync(&dataBuffs[i], maxSizes[rank], stream));
    }
  }
  void* devBs = nullptr;
  CHECK_CUDA(cudaMallocAsync(&devBs, sizeof(cuda::std::byte*) * world, stream));

  std::random_device rd;
  matx::cudaExecutor exec{stream};
  Times times{};
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    const auto sizes = makeSizes(bytes, world);
    const auto offsets = makeOffsets(sizes);
    ctx.vState = makeVState(sizes, offsets, rank);
    CHECK_CUDA(cudaMemcpyAsync(devSizes, sizes.data(), sizeof(size_t) * world, cudaMemcpyHostToDevice, stream));

    uint seed;
    if (rank == 0) {
      seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const auto localElems = ctx.vState.bytes / sizeof(DataType);
    for (int i = 0; i < world; ++i) {
      if (i != rank) {
        const auto theirSeed = (i + 1) * (seed + rank * 42);
        auto* cB = reinterpret_cast<DataType*>(dataBuffs[i]);
        randUniform<ARCH>(cB, localElems, theirSeed, -1.f, 1.f, stream);
      }
      {
        const auto theirSeed = (rank + 1) * (seed + i * 42);
        auto* cB = reinterpret_cast<DataType*>(srcBuff + offsets[i]);
        randUniform<ARCH>(cB, sizes[i] / sizeof(DataType), theirSeed, -1.f, 1.f, stream);
      }
    }
    dataBuffs[rank] = srcBuff + offsets[rank];
    CHECK_CUDA(cudaMemcpyAsync(devBs, dataBuffs.data(), sizeof(cuda::std::byte*) * world, cudaMemcpyHostToDevice, stream));

    constexpr uint rkThreads = 512;
    const auto rkBlocks = cuda::ceil_div(localElems, rkThreads);
    rsvReferenceKernel<<<rkBlocks, rkThreads, 0, stream>>>
      (static_cast<const DataType* const*>(devBs), refBuff, world, localElems);

    purlin::reduceScatterV<DataType>(srcBuff, dstBuff, devSizes, ctx, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    using MRE = MXE<DataType>;
    auto tR = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(dstBuff), {1, static_cast<matx::index_t>(localElems)});
    auto tRef = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(refBuff), {1, static_cast<matx::index_t>(localElems)});
    auto matches = matx::make_tensor<long int>({});
    (matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);

    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < opts.runs; ++i) {
        purlin::reduceScatterV<DataType>(srcBuff, dstBuff, devSizes, ctx, stream);
      }
      purlin::reduceScatterV<DataType>(srcBuff, dstBuff, devSizes, ctx, stream);
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
        purlin::reduceScatterV<DataType>(srcBuff, dstBuff, devSizes, ctx, stream);
      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      for (int i = 0; i < opts.runs; ++i) {
        purlin::reduceScatterV<DataType>(srcBuff, dstBuff, devSizes, ctx, stream);
      }
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }

    times.ep = (1.0 - static_cast<double>(matches()) / static_cast<double>(tR.TotalSize())) * 100.0;
    times.t_ms = t_ms;
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = static_cast<double>(ctx.vState.totalBytes) / 1e9;
      const auto purlin_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %lu, %lu, %s, %lf, %lf, %lf, %s, %d, %d, %d\n",
        world, ctx.vState.bytes, ctx.vState.maxBytes, ctx.vState.totalBytes, element_string<DataType>(),
        times.t_ms, purlin_algBW, times.ep, prop.name, opts.graph_launches > 0 ? opts.runs : opts.warmup,
        opts.runs, opts.graph_launches);
    }
  }
  for (int i = 0; i < world; ++i) {
    if (i != rank) {
      CHECK_CUDA(cudaFreeAsync(dataBuffs[i], stream));
    }
  }
  CHECK_CUDA(cudaFreeAsync(srcBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  CHECK_CUDA(cudaFreeAsync(dstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(devSizes, stream));
  CHECK_CUDA(cudaFreeAsync(devBs, stream));
  purlin::finalize(ctx, stream);
  destroyWorkspace(workspace, rank, stream);
  nvshmem_finalize();
}

// NVSHMEM_BOOTSTRAP=MPI mpirun -n <world> ./testRSV <minLocalBytes> <maxLocalBytes> <graph_launches> <runs> <warmup>
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
  if (opts.minLocalBytes % 32 != 0 || opts.maxLocalBytes % 32 != 0) {
    throw std::invalid_argument("Size must be a multiple of 32 bytes");
  }
  rsvHost(opts);
}
