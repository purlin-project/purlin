//
// Created by osayamen on 5/31/26.
//
#include <random>
#include <string>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nvshmem.h>

#include <purlin/core.cuh>
#include <purlin/host/reduceScatter.cuh>

#include "util.cuh"

using DataType = __half;

// oracular deterministic, reference kernel.
template<typename Element>
__global__ void rk(const Element* const* __restrict__ sources, Element* __restrict__ dstBuff,
  const int world, const size_t elems) {
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

__host__
void rsHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  cuda::std::byte* dstBuff = nullptr;
  DataType* refBuff = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (world <= 1) {
    printf("Requires at least two processes!\n");
    return;
  }
  if (rank == 0) {
    printf("world,localBytes,globalBytes,datatype,purlin(ms),purlin(GB/s),error_vs_oracle(%%)"
           "GPUName,warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  const auto workspace = makeWorkspace(world, stream);
  auto ctx = purlin::initialize(rank, world, workspace, stream);

  CHECK_CUDA(cudaMallocAsync(&srcBuff, world * opts.maxLocalBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&dstBuff, opts.maxLocalBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&refBuff, opts.maxLocalBytes, stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<cuda::std::byte*> dataBuffs(world, nullptr);
  for (int i = 0 ; i < world; ++i) {
    if (i != rank) {
      CHECK_CUDA(cudaMallocAsync(&dataBuffs[i], opts.maxLocalBytes, stream));
    }
  }
  void* devBs = nullptr;
  CHECK_CUDA(cudaMallocAsync(&devBs, sizeof(cuda::std::byte*) * world, stream));
  std::random_device rd;
  matx::cudaExecutor exec{stream};
  Times times{};
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    // fill buffer with random values
    uint seed;
    if (rank == 0) {
      seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const auto elems = bytes / sizeof(DataType);
    for (int i = 0; i < world; ++i) {
      if (i != rank) {
        const auto theirSeed = (i + 1) * (seed + rank * 42);
        auto* cB = reinterpret_cast<DataType*>(dataBuffs[i]);
        randUniform<ARCH>(cB, elems, theirSeed, -1.f, 1.f, stream);
      }
      {
        const auto theirSeed = (rank + 1) * (seed + i * 42);
        auto* cB = reinterpret_cast<DataType*>(srcBuff + bytes * i);
        randUniform<ARCH>(cB, elems, theirSeed, -1.f, 1.f, stream);
      }
    }
    dataBuffs[rank] = srcBuff + rank * bytes;
    CHECK_CUDA(cudaMemcpyAsync(devBs, dataBuffs.data(), sizeof(cuda::std::byte*) * world, cudaMemcpyHostToDevice, stream));

    constexpr uint rkThreads = 512;
    const auto rkBlocks = cuda::ceil_div(elems, rkThreads);
    rk<<<rkBlocks, rkThreads, 0, stream>>>(static_cast<const DataType* const*>(devBs), refBuff, world, elems);
    // correctness run
    purlin::reduceScatter<DataType>(srcBuff, dstBuff, bytes, ctx, stream);
    auto ar_matches0 = matx::make_tensor<long int>({});
    using MRE = MXE<DataType>;
    auto tR = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(dstBuff), {1, static_cast<matx::index_t>(elems)});
    auto tO = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(refBuff), {1, static_cast<matx::index_t>(elems)});
    // bitwise correctness check against oracle
    auto ar_matches1 = matx::make_tensor<long int>({});
    (ar_matches1 = matx::sum(matx::isclose(tR, tO, 0, 0))).run(exec);
    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < opts.runs; ++i) {
        purlin::reduceScatter<DataType>(srcBuff, dstBuff, bytes, ctx, stream);
      }
      purlin::reduceScatter<DataType>(srcBuff, dstBuff, bytes, ctx, stream);
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
        purlin::reduceScatter<DataType>(srcBuff, dstBuff, bytes, ctx, stream);
      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      for (int i = 0; i < opts.runs; ++i) {
        purlin::reduceScatter<DataType>(srcBuff, dstBuff, bytes, ctx, stream);
      }
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }
    times.ep = (1.0 - static_cast<double>(ar_matches1()) / static_cast<double>(tR.TotalSize())) * 100.0;
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = (world * static_cast<double>(bytes)) / 1e9;
      const auto purlin_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %lu, %s, %lf, %lf, %lf, %s, %d, %d, %d\n",
        world, bytes, world * bytes, element_string<DataType>(), times.t_ms, purlin_algBW, times.ep, prop.name,
        opts.graph_launches > 0 ? opts.runs : opts.warmup,
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
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  purlin::finalize(ctx, stream);
  destroyWorkspace(workspace, rank, stream);
  nvshmem_finalize();
}

//NVSHMEM_BOOTSTRAP=MPI mpirun -n <world> ./rs <minLocalBytes> <maxLocalBytes> <graph_launches> <runs> <warmup>
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
  if (opts.minLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0 || opts.maxLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(purlin::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  rsHost(opts);
}