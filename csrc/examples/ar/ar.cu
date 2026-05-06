//
// Created by Osayamen on 3/30/26.
//
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>

#include <suture/suture.cuh>

#include "../common.cuh"
#include "../debug.cuh"

#ifndef AR_THREADS
#define AR_THREADS 128
#endif
#ifndef AR_UNROLL_FACTOR
#define AR_UNROLL_FACTOR 2
#endif
#ifndef AR_ALIGNMENT
#define AR_ALIGNMENT 16
#endif
#ifndef AR_PIPE_STAGES
#define AR_PIPE_STAGES 4
#endif
#ifndef AR_ELEMENTS_PER_THREAD
#define AR_ELEMENTS_PER_THREAD 16
#endif
#ifndef AR_WORLD_UNROLL
#define AR_WORLD_UNROLL 2
#endif

constexpr auto threads = 256;
constexpr auto unrollFactor = 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 8;
constexpr auto elementsPerThread = 2;
constexpr auto worldUnroll = 2;

constexpr auto nArch = suture::normalizeArch<ARCH>();
using SutureConfig = suture::Configuration<
    nArch,
    threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor,
    suture::AUTO,
    worldUnroll
>;

struct Args {
  const cuda::std::byte* const src;
  cuda::std::byte* const dst;
  const size_t bytes;
  const cuda::fast_mod_div<long int> blocks;
};

using DataType = __half;
constexpr auto NE = ncclFloat16;

template<typename SutureAtom, typename Element>
__launch_bounds__(SutureAtom::THREADS, 1)
__global__ void allReduce(const __grid_constant__ Args kArgs,
  const __grid_constant__ suture::Context ctx) {
  extern __shared__ __align__(SutureAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
  auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
  suture::allReduce<SutureAtom>(kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx, kArgs.blocks);
}

// AllReduce reference kernel, not an optimal implementation
template<typename Element>
__global__ void rk(const Element* const* __restrict__ sources, Element* __restrict__ dstBuff, const int world, const size_t elems) {
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
void arHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  DataType* dstBuff = nullptr;
  cuda::std::byte* refBuff = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (world <= 1) {
    printf("Requires at least two processes!\n");
    return;
  }
  if (rank == 0) {
    printf("world,bytes,datatype,suture(ms),suture(GB/s),error_vs_oracle(%%),error_vs_nccl(%%),"
           "nArch,GPUName,threads,pipeStages,stageExtent,unrollFactor,worldUnroll,"
           "SMsOnGPU,superBlockSize,blocks,chunkSize(MiB),warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  auto ctx = suture::initialize(rank, world, stream);
  using SutureAtom = suture::Atom<nArch, SutureConfig>;
  auto kernel = allReduce<SutureAtom, DataType>;
  const auto kernelSharedSize = opts.maxLocalBytes > suture::RED_LATENCY_BOUND_THRESHOLD ?
  SutureAtom::SMEM_SIZE : 0;
  if (opts.maxLocalBytes > suture::RED_LATENCY_BOUND_THRESHOLD) {
    int maxSharedMemory = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
    if (kernelSharedSize > maxSharedMemory) {
      const auto errmsg = std::string("Required shared memory ").append(std::to_string(kernelSharedSize))
      .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
      throw std::runtime_error(errmsg);
    }
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSharedSize));
  }
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, SutureAtom::THREADS, kernelSharedSize));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  constexpr auto maxActualSBSize = 32;
  opts.maxSuperBlockSize = opts.maxSuperBlockSize <= 0 ? maxActualSBSize : min(opts.maxSuperBlockSize, maxActualSBSize);
  const auto requestedCTAs = opts.maxSuperBlockSize * world;
  const auto availableCTAs = bps * num_sms;
  const auto superBlockSize0 = requestedCTAs > availableCTAs ?
  (cuda::round_down(availableCTAs, world) / world) : opts.maxSuperBlockSize;

  CHECK_CUDA(cudaMallocAsync(&dstBuff, opts.maxLocalBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&refBuff, opts.maxLocalBytes, stream));
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

  std::vector<cuda::std::byte*> dataBuffs(world, nullptr);
  for (auto & dataBuff : dataBuffs) {
    CHECK_CUDA(cudaMallocAsync(&dataBuff, opts.maxLocalBytes, stream));
  }
  srcBuff = dataBuffs[rank];
  void* devBs = nullptr;
  CHECK_CUDA(cudaMallocAsync(&devBs, sizeof(cuda::std::byte*) * world, stream));
  CHECK_CUDA(cudaMemcpyAsync(devBs, dataBuffs.data(), sizeof(cuda::std::byte*) * world, cudaMemcpyHostToDevice, stream));

  std::random_device rd;
  auto ark = [&](const auto& blocks, const Args& kArgs, const suture::Context& kCtx, const bool isLR, const int& runs) {
    for (int i = 0; i < runs; ++i) {
      allReduce<SutureAtom, DataType><<<blocks, SutureAtom::THREADS, isLR ? 0 : kernelSharedSize, stream>>>(kArgs, kCtx);
    }
  };
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
      const auto theirSeed = seed + i * 42;
      auto* cB = reinterpret_cast<DataType*>(dataBuffs[i]);
      randUniform<ARCH>(cB, elems, theirSeed, -1.f, 1.f, stream);
    }
    CHECK_CUDA(cudaMemcpyAsync(refBuff, srcBuff, bytes, cudaMemcpyDeviceToDevice, stream));
    const auto isLR = bytes <= suture::RED_LATENCY_BOUND_THRESHOLD;
    const auto dataAlignment = isLR ? sizeof(suture::LRP16::RT) :
    suture::MAX_ACCESS_ALIGNMENT;
    auto superBlockSize = static_cast<int>(min(cuda::ceil_div(bytes, SutureAtom::THREADS * dataAlignment),
      static_cast<size_t>(superBlockSize0)));
    const auto reduceBlocks = superBlockSize * (world - 1);
    const auto blocks = isLR ? superBlockSize * world : reduceBlocks + suture::RED_PUT_BLOCKS;
    if (blocks < 1) {
      throw std::runtime_error("Blocks must be >= 1");
    }
    const Args kArgs{
      .src = srcBuff,
      .dst = srcBuff,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ctx.setSuperBlockSize(superBlockSize);
    constexpr uint rkThreads = 512;
    const auto rkBlocks = cuda::ceil_div(elems, rkThreads);
    // Compute the oracle before the in-place AllReduce overwrites dataBuffs[rank].
    rk<<<rkBlocks, rkThreads, 0, stream>>>(static_cast<const DataType* const*>(devBs), dstBuff, world, elems);
    // correctness run
    ark(blocks, kArgs, ctx, isLR, 1);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    NCCL_CHECK(ncclAllReduce(refBuff, refBuff, elems, NE, ncclSum, comm, stream));
    auto ar_matches0 = matx::make_tensor<long int>({});
    using MRE = MXE<DataType>;
    auto tR = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(srcBuff), {1, static_cast<matx::index_t>(elems)});
    auto tRef = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(refBuff), {1, static_cast<matx::index_t>(elems)});
    // correctness check against nccl
    (ar_matches0 = matx::sum(matx::isclose(tR, tRef, opts.rtol, opts.atol))).run(exec);
    auto tO = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(dstBuff), {1, static_cast<matx::index_t>(elems)});
    // correctness check against oracle
    auto ar_matches1 = matx::make_tensor<long int>({});
    (ar_matches1 = matx::sum(matx::isclose(tR, tO, opts.rtol, opts.atol))).run(exec);
    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      ark(blocks, kArgs, ctx, isLR, opts.runs);
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
      // benchmark suture without graphs
      ark(blocks, kArgs, ctx, isLR, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      ark(blocks, kArgs, ctx, isLR, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }
    times.ep = (1.0 - static_cast<double>(ar_matches0()) / static_cast<double>(tR.TotalSize())) * 100.0;
    times.oracle_ep = (1.0 - static_cast<double>(ar_matches1()) / static_cast<double>(tR.TotalSize())) * 100.0;
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = (static_cast<double>(bytes)) / 1e9;
      const auto suture_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %s, %lf, %lf, %lf, %lf, %d, %s, %d, %d, %d, %d, %d, %d, %d, %d, %lu, %d, %d, %d\n",
        world, bytes, element_string<DataType>(), times.t_ms, suture_algBW, times.oracle_ep, times.ep,
        nArch, prop.name, threads, pipeStages, elementsPerThread, unrollFactor, SutureConfig::WORLD_UNROLL,
        num_sms, superBlockSize, blocks, isLR ? 0 : suture::RED_CHUNK_SIZE / (1024UL * 1024),  opts.graph_launches > 0 ? opts.runs : opts.warmup,
        opts.runs, opts.graph_launches);
    }
  }
  for (auto & dataBuff : dataBuffs) {
    CHECK_CUDA(cudaFreeAsync(dataBuff, stream));
  }
  suture::finalize(ctx, stream);
  CHECK_CUDA(cudaFreeAsync(dstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}

// ./ag <minLocalBytes> <maxLocalBytes> <graph_launches> <maxSuperBlockSize> <runs> <warmup> <rtol> <atol>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.maxSuperBlockSize = -1; // auto-tuned
  opts.rtol = 0; // bitwise
  opts.atol = 0; // bitwise
  opts.runs = 128;
  opts.warmup = 128;
  opts.graph_launches = 8;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.maxSuperBlockSize = std::stoi(argv[3]);
  if (argc > 4) opts.graph_launches = std::stoi(argv[4]);
  if (argc > 5) opts.runs = std::stoi(argv[5]);
  if (argc > 6) opts.warmup = std::stoi(argv[6]);
  if (argc > 7) opts.rtol = std::stof(argv[7]);
  if (argc > 8) opts.atol = std::stof(argv[8]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % suture::MAX_ACCESS_ALIGNMENT != 0 || opts.maxLocalBytes % suture::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(suture::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  arHost(opts);
}
