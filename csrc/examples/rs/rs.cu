//
// Created by osayamen on 5/9/26.
//
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>

#include <purlin/core.cuh>

#include <util.cuh>

constexpr auto threads = 256; // A100: 256;
constexpr auto unrollFactor = 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 8; //A100: 8;
constexpr auto elementsPerThread = 2; // A100: 2;
constexpr auto worldUnroll = 2;
constexpr auto nArch = purlin::normalizeArch<ARCH>();
using TRConfig = purlin::Configuration<
    purlin::Regime::throughput,
    threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor,
    worldUnroll
>;

using LRConfig = purlin::Configuration<
  purlin::Regime::latency,
  512, /*threads*/
  alignment,
  purlin::UNUSED,
  purlin::UNUSED,
  unrollFactor,
  worldUnroll
>;

struct Args {
  const cuda::std::byte* const src;
  cuda::std::byte* const dst;
  const size_t bytes;
  const cuda::fast_mod_div<long int> blocks;
};

using DataType = __half;
// 2MiB -> 4MiB <= globalBytes <= 16MiB
// 4MiB -> 32MiB <= globalBytes <= 128MiB
constexpr size_t CHUNK_SIZE = 4 * 1024 * 1024;
constexpr int CHUNKED_PUT_BLOCKS = 16;
constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
template<typename PurlinAtom, typename Element, typename CollConfig>
__launch_bounds__(PurlinAtom::THREADS, 1)
__global__ void reduceScatter(const __grid_constant__ Args kArgs, const __grid_constant__ purlin::Context ctx) {
  extern __shared__ __align__(PurlinAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
  auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
  purlin::reduceScatter<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx, kArgs.blocks);
}

// reference kernel, not an optimal implementation
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
  if (world > purlin::MAX_RANKS_PER_DOMAIN) {
    if (rank == 0) {
      printf("Requires at most %d processes, which typically fits a single-node!\n",
        purlin::MAX_RANKS_PER_DOMAIN);
    }
    return;
  }
  if (CHUNKED_PUT_BLOCKS % world != 0) {
    throw std::runtime_error("put blocks: " + std::to_string(CHUNKED_PUT_BLOCKS) + " must be a multiple of world");
  }
  if (rank == 0) {
    printf("world,localBytes,globalBytes,datatype,purlin(ms),purlin(GB/s),error_vs_oracle(%%),"
           "nArch,GPUName,threads,pipeStages,stageExtent,unrollFactor,worldUnroll,"
           "SMsOnGPU,putBlocks,reduceBlocks,blocks,chunkSize(MiB),warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  const auto workspace = makeWorkspace(world, stream);
  auto ctx = purlin::initialize(rank, world, workspace, stream);
  using PurlinAtomLR = purlin::Atom<nArch, LRConfig>;
  using PurlinAtomTR = purlin::Atom<nArch, TRConfig>;
  using nonChunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::nonChunked,
    NON_CHUNKED_PUT_BLOCKS,
    purlin::UNUSED,
    CHUNK_SIZE
  >;
  using chunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::chunked,
    CHUNKED_PUT_BLOCKS,
    purlin::UNUSED,
    CHUNK_SIZE
  >;
  constexpr auto kSTR = cuda::std::max(PurlinAtomTR::COPY_SMEM_SIZE, PurlinAtomTR::RED_SMEM_SIZE);
  constexpr auto kSLR = PurlinAtomLR::RED_SMEM_SIZE;
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  auto kernelTRNonChunked = reduceScatter<PurlinAtomTR, DataType, nonChunkedConfig>;
  auto kernelTRChunked = reduceScatter<PurlinAtomTR, DataType, chunkedConfig>;
  auto kernelLR = reduceScatter<PurlinAtomLR, DataType, purlin::CollectiveConfigLR>;
  {
    if (kSTR > maxSharedMemory) {
      const auto errmsg = std::string("Required shared memory ").append(std::to_string(kSTR))
      .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
      throw std::runtime_error(errmsg);
    }
    CHECK_CUDA(cudaFuncSetAttribute(kernelTRNonChunked, cudaFuncAttributeMaxDynamicSharedMemorySize, kSTR));
    CHECK_CUDA(cudaFuncSetAttribute(kernelTRChunked, cudaFuncAttributeMaxDynamicSharedMemorySize, kSTR));
  }
  {
    if (kSLR > maxSharedMemory) {
      const auto errmsg = std::string("Required shared memory ").append(std::to_string(kSLR))
      .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
      throw std::runtime_error(errmsg);
    }
    CHECK_CUDA(cudaFuncSetAttribute(kernelLR, cudaFuncAttributeMaxDynamicSharedMemorySize, kSLR));
  }

  const auto CTAsUpperLR = cuda::std::min(64U, cuda::std::bit_floor(static_cast<uint32_t>(num_sms)));

  CHECK_CUDA(cudaMallocAsync(&srcBuff, world * opts.maxLocalBytes, stream));
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
  for (int i = 0 ; i < world; ++i) {
    if (i != rank) {
      CHECK_CUDA(cudaMallocAsync(&dataBuffs[i], opts.maxLocalBytes, stream));
    }
  }
  void* devBs = nullptr;
  CHECK_CUDA(cudaMallocAsync(&devBs, sizeof(cuda::std::byte*) * world, stream));
  std::random_device rd;
  auto rsk = [&](const auto& blocks, const Args& kArgs, const purlin::Context& kCtx, const bool isLR, const int& runs) {
    if (isLR) {
      for (int i = 0; i < runs; ++i) {
        reduceScatter<PurlinAtomLR, DataType, purlin::CollectiveConfigLR>
        <<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>(kArgs, kCtx);
      }
    }
    else {
      if (kArgs.bytes <= CHUNK_SIZE) {
        for (int i = 0; i < runs; ++i) {
          reduceScatter<PurlinAtomTR, DataType, nonChunkedConfig>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, kCtx);
        }
      }
      else {
        for (int i = 0; i < runs; ++i) {
          reduceScatter<PurlinAtomTR, DataType, chunkedConfig>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, kCtx);
        }
      }
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

    const auto isLR = purlin::getRedRegime(bytes, world) == purlin::Regime::latency;
    const auto putBlocks = bytes <= CHUNK_SIZE ? nonChunkedConfig::PUT_BLOCKS : chunkedConfig::PUT_BLOCKS;
    const auto maxReduceBlocks = cuda::std::min(static_cast<uint32_t>(opts.maxReduceBlocks),
    cuda::std::bit_floor(static_cast<uint32_t>(num_sms - putBlocks)));
    size_t blocks = 0;
    if (isLR) {
      blocks = cuda::std::min(cuda::ceil_div(bytes, PurlinAtomLR::THREADS*sizeof(purlin::LRP16::RT)),
        static_cast<size_t>(CTAsUpperLR));
    }
    else {
      auto blocksNeeded = cuda::std::min(bytes / PurlinAtomTR::RED_PIPELINE_BYTES,
        bytes / (world * PurlinAtomTR::STAGE_BYTES));
      blocksNeeded = static_cast<int>(min(blocksNeeded,static_cast<size_t>(maxReduceBlocks)));
      blocks = putBlocks + blocksNeeded;
      if (blocksNeeded < 1) {
        // non-pipelined path
        blocks = putBlocks + cuda::std::min(cuda::ceil_div(bytes / world,
          static_cast<size_t>(PurlinAtomTR::THREADS*PurlinAtomTR::BaseConfig::ALIGNMENT_BYTES)),
          static_cast<size_t>(maxReduceBlocks));
      }
    }
    const Args kArgs{
      .src = srcBuff,
      .dst = dstBuff,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{static_cast<long int>(blocks)}
    };
    constexpr uint rkThreads = 512;
    const auto rkBlocks = cuda::ceil_div(elems, rkThreads);
    rk<<<rkBlocks, rkThreads, 0, stream>>>(static_cast<const DataType* const*>(devBs), refBuff, world, elems);
    // correctness run
    rsk(blocks, kArgs, ctx, isLR, 1);
    using MRE = cuda::std::conditional_t<sizeof(DataType) == 1, uint8_t, MXE<DataType>>;
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
      rsk(blocks, kArgs, ctx, isLR, opts.runs);
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
      rsk(blocks, kArgs, ctx, isLR, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      rsk(blocks, kArgs, ctx, isLR, opts.runs);
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
      printf("%d, %lu, %lu, %s, %lf, %lf, %lf, %d, %s, %d, %s, %s, %s, %d, %d, %s, %s, %d, %s, %d, %d, %d\n",
        world, bytes, world * bytes, element_string<DataType>(), times.t_ms, purlin_algBW, times.ep,
        nArch, prop.name,
        isLR ? PurlinAtomLR::THREADS : PurlinAtomTR::THREADS,
        isLR ? "N/A" : std::to_string(pipeStages).c_str(),
        isLR ? "N/A" : std::to_string(elementsPerThread).c_str(),
        isLR ? "N/A" : std::to_string(unrollFactor).c_str(),
        worldUnroll,
        num_sms,
        isLR ? "N/A" : std::to_string(putBlocks).c_str(),
        isLR ? "N/A" : std::to_string(blocks - putBlocks).c_str(),
        static_cast<int>(blocks),
        isLR ? "N/A" : std::to_string(CHUNK_SIZE / (1024UL * 1024)).c_str(),
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
  CHECK_CUDA(cudaFreeAsync(dstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  purlin::finalize(ctx, stream);
  destroyWorkspace(workspace, rank, stream);
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}

//NVSHMEM_BOOTSTRAP=MPI mpirun -n <world> ./rs <minLocalBytes> <maxLocalBytes> <maxReduceBlocks> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.maxReduceBlocks = 32; // auto-tuned
  opts.runs = 128;
  opts.warmup = 128;
  opts.graph_launches = 8;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.maxReduceBlocks = std::stoi(argv[3]);
  if (argc > 4) opts.graph_launches = std::stoi(argv[4]);
  if (argc > 5) opts.runs = std::stoi(argv[5]);
  if (argc > 6) opts.warmup = std::stoi(argv[6]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0 || opts.maxLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(purlin::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  rsHost(opts);
}
