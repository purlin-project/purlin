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
constexpr auto threads = 128;
constexpr auto unrollFactor = 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 4;
constexpr auto elementsPerThread = 8;
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
};

using DataType = __half;
constexpr auto NE = ncclFloat16;

template<typename SutureAtom, typename Element>
__global__ void allReduceKernel(const __grid_constant__ Args kArgs,
  const __grid_constant__ suture::SutureContext ctx) {
  extern __shared__ __align__(SutureAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
  auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
  suture::allReduce<SutureAtom>(kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx);
}

// AllReduce reference kernel, not an optimal implementation
template<typename Element>
__global__ void rk(Element** __restrict__ bufs, const int rank, const int world, const size_t elems) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= elems) {
    return;
  }
  auto* __restrict__ result = bufs[rank];
  using AccumType = cuda::std::common_type_t<Element, float>;
  auto accumulator = static_cast<AccumType>(0.f);
  for (int i = 0; i < world; ++i) {
    constexpr Converter<AccumType, Element> loadConv{};
    accumulator += loadConv(bufs[i][tid]);
  }
  constexpr Converter<Element, AccumType> storeConv{};
  result[tid] = storeConv(accumulator);
}

__host__
void arHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  cuda::std::byte* rcvBuff = nullptr;
  cuda::std::byte* refBuff = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("world,bytes,datatype,suture(ms),suture(GB/s),error_vs_oracle(%%),error_vs_nccl(%%),"
           "nArch,GPUName,threads,pipeStages,stageExtent,unrollFactor,worldUnroll,"
           "SMsOnGPU,superBlockSize,blocks,warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  auto ctx = suture::initialize(rank, world, stream);
  using SutureAtom = suture::Atom<nArch, SutureConfig>;
  auto kernel = allReduceKernel<SutureAtom, DataType>;
  const auto kernelSharedSize = opts.maxLocalBytes > suture::AR_LATENCY_BOUND_THRESHOLD ?
  SutureAtom::SMEM_SIZE : 0;
  if (opts.maxLocalBytes > suture::AR_LATENCY_BOUND_THRESHOLD) {
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
  const auto actualWorld = world - 1;
  const auto maxActualSBSize = getSBZ<nArch, suture::AG_SUPER_BLOCK_THRESHOLD>(world, opts.maxLocalBytes);
  opts.maxSuperBlockSize = opts.maxSuperBlockSize <= 0 ? maxActualSBSize : min(opts.maxSuperBlockSize, maxActualSBSize);
  const auto requestedCTAs = opts.maxSuperBlockSize * actualWorld;
  const auto availableCTAs = bps * num_sms;
  const auto superBlockSize0 = requestedCTAs > availableCTAs ?
  (cuda::round_down(availableCTAs, actualWorld) / actualWorld) : opts.maxSuperBlockSize;

  CHECK_CUDA(cudaMallocAsync(&rcvBuff, opts.maxLocalBytes, stream));
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
  auto agk = [&](const auto& blocks, const Args& kArgs, const suture::SutureContext& kCtx, const int& runs) {
    for (int i = 0; i < runs; ++i) {
      allReduceKernel<SutureAtom, DataType><<<blocks, SutureAtom::THREADS, kernelSharedSize, stream>>>(kArgs, kCtx);
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const cuda::fast_mod_div<int> world_v{world};
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
    auto superBlockSize = static_cast<int>(min(cuda::ceil_div(bytes, SutureAtom::THREADS * suture::MAX_ACCESS_ALIGNMENT),
      static_cast<size_t>(superBlockSize0)));
    if (world < 8 && superBlockSize > 16) {
      // A100
      superBlockSize = bytes < suture::AR_SUPER_BLOCK_THRESHOLD ? 16 : superBlockSize;
    }
    const cuda::fast_mod_div<int> superBlockSize_v{superBlockSize};
    const Args kArgs{
      .src = srcBuff,
      .dst = rcvBuff,
      .bytes = bytes
    };
    ctx.setSuperBlockSize(superBlockSize);
    const auto blocks = superBlockSize * actualWorld;
    // correctness run
    agk(blocks, kArgs, ctx, 1);
    ncclAllReduce(refBuff, refBuff, elems, NE, ncclSum, comm, stream);
    auto ar_matches0 = matx::make_tensor<long int>({});
    using MRE = MXE<DataType>;
    auto tR = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(rcvBuff), {1, static_cast<matx::index_t>(elems)});
    auto tRef = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(refBuff), {1, static_cast<matx::index_t>(elems)});
    // correctness check against nccl
    (ar_matches0 = matx::sum(matx::isclose(tR, tRef, opts.rtol, opts.atol))).run(exec);
    constexpr uint rkThreads = 512;
    const auto rkBlocks = cuda::ceil_div(elems, rkThreads);
    rk<<<rkBlocks, rkThreads, 0, stream>>>(static_cast<DataType**>(devBs), rank, world, elems);
    auto tO = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(srcBuff), {1, static_cast<matx::index_t>(elems)});
    // correctness check against oracle
    auto ar_matches1 = matx::make_tensor<long int>({});
    (ar_matches1 = matx::sum(matx::isclose(tR, tO, opts.rtol, opts.atol))).run(exec);
    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      agk(blocks, kArgs, ctx, opts.runs);
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
      agk(blocks, kArgs, ctx, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      agk(blocks, kArgs, ctx, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }
    times.ep = (1.0 - (static_cast<double>(ar_matches0()) / static_cast<double>(tR.TotalSize()))) * 100.0;
    times.oracle_ep = (1.0 - (static_cast<double>(ar_matches1()) / static_cast<double>(tR.TotalSize()))) * 100.0;
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = (static_cast<double>(bytes)) / 1e9;
      const auto suture_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %s, %lf, %lf, %lf, %lf, %d, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
        world, bytes, element_string<DataType>(), times.t_ms, suture_algBW, times.oracle_ep, times.ep,
        nArch, prop.name, threads, pipeStages, elementsPerThread, unrollFactor, SutureConfig::WORLD_UNROLL,
        num_sms, superBlockSize, blocks,  opts.graph_launches > 0 ? opts.runs : opts.warmup,
        opts.runs, opts.graph_launches);
    }
  }
  for (auto & dataBuff : dataBuffs) {
    CHECK_CUDA(cudaFreeAsync(dataBuff, stream));
  }
  CHECK_CUDA(cudaFreeAsync(rcvBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}

// ./ag <minLocalBytes> <maxLocalBytes> <graph_launches> <maxSuperBlockSize> <runs> <warmup> <rtol> <atol>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.maxSuperBlockSize = -1;
  opts.graph_launches = 16;
  opts.rtol = 2e-2;
  opts.atol = 2e-3;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.graph_launches = std::stoi(argv[3]);
  if (argc > 4) opts.maxSuperBlockSize = std::stoi(argv[4]);
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