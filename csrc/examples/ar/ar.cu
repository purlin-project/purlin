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
//{128,4,4}
constexpr auto threads = 256; // A100: 256;
constexpr auto unrollFactor = 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 8; //A100: 8;
constexpr auto elementsPerThread = 1; // A100: 2;
constexpr auto worldUnroll = 8;
constexpr auto nArch = suture::normalizeArch<ARCH>();
using TRConfig = suture::Configuration<
    nArch,
    suture::Regime::throughput,
    threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor,
    worldUnroll
>;

using LRConfig = suture::Configuration<
  nArch,
  suture::Regime::latency,
  512, /*threads*/
  alignment,
  suture::UNUSED,
  suture::UNUSED,
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
constexpr auto NE = ncclFloat16;
// 2MiB -> 4MiB <= bytes <= 16MiB,
// 4MiB -> 32MiB <= bytes <= 128MiB
// 8MiB -> 256 MiB <=  bytes
constexpr size_t CHUNK_SIZE = 4 * 1024 * 1024;
constexpr int GATHER_BLOCKS = 16;

template<typename SutureAtom, typename Element, typename CollConfig>
__launch_bounds__(SutureAtom::THREADS, 1)
__global__ void allReduce(const __grid_constant__ Args kArgs, const __grid_constant__ suture::Context ctx) {
  extern __shared__ __align__(SutureAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
  auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
  suture::allReduce<SutureAtom, CollConfig>(kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx, kArgs.blocks);
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
  if (world > suture::MAX_RANKS_PER_DOMAIN) {
    if (rank == 0) {
      printf("Requires at most %d processes, which typically fits a single-node!\n",
        suture::MAX_RANKS_PER_DOMAIN);
    }
    return;
  }
  if (suture::CHUNKED_PUT_BLOCKS % world != 0) {
    throw std::runtime_error("put blocks: " + std::to_string(suture::CHUNKED_PUT_BLOCKS) + " must be a multiple of world");
  }
  if (GATHER_BLOCKS % world != 0) {
    throw std::runtime_error("gather blocks: " + std::to_string(GATHER_BLOCKS) + " must be a multiple of world");
  }
  if (rank == 0) {
    printf("world,bytes,datatype,suture(ms),suture(GB/s),error_vs_oracle(%%),error_vs_nccl(%%),"
           "nArch,GPUName,threads,pipeStages,stageExtent,unrollFactor,worldUnroll,"
           "SMsOnGPU,putBlocks,reduceBlocks,gatherBlocks,blocks,chunkSize(MiB),warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  auto ctx = suture::initialize(rank, world, stream);
  using SutureAtomLR = suture::Atom<nArch, LRConfig>;
  using SutureAtomTR = suture::Atom<nArch, TRConfig>;
  using nonChunkedConfig = suture::CollectiveConfig<
    suture::CollectiveType::nonChunked,
    16,
    GATHER_BLOCKS,
    CHUNK_SIZE
  >;
  using chunkedConfig = suture::CollectiveConfig<
    suture::CollectiveType::chunked,
    suture::CHUNKED_PUT_BLOCKS,
    GATHER_BLOCKS,
    CHUNK_SIZE
  >;
  constexpr auto kSTR = cute::max(SutureAtomTR::COPY_SMEM_SIZE, SutureAtomTR::RED_SMEM_SIZE);
  constexpr auto kSLR = SutureAtomLR::RED_SMEM_SIZE;
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  auto kernelTRNonChunked = allReduce<SutureAtomTR, DataType, nonChunkedConfig>;
  auto kernelTRChunked = allReduce<SutureAtomTR, DataType, chunkedConfig>;
  auto kernelLR = allReduce<SutureAtomLR, DataType, suture::CollectiveConfigLR>;
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
  const auto CTAsUpperLR = cute::min(64, cuda::std::bit_floor(static_cast<uint32_t>(num_sms)));

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
    if (isLR) {
      for (int i = 0; i < runs; ++i) {
        allReduce<SutureAtomLR, DataType, suture::CollectiveConfigLR>
        <<<blocks, SutureAtomLR::THREADS, kSLR, stream>>>(kArgs, kCtx);
      }
    }
    else {
      const auto bytesCheck = world == 2 ? kArgs.bytes : kArgs.bytes / world;
      if (bytesCheck <= CHUNK_SIZE) {
        for (int i = 0; i < runs; ++i) {
          allReduce<SutureAtomTR, DataType, nonChunkedConfig>
          <<<blocks, SutureAtomTR::THREADS, kSTR, stream>>>(kArgs, kCtx);
        }
      }
      else {
        for (int i = 0; i < runs; ++i) {
          allReduce<SutureAtomTR, DataType, chunkedConfig>
          <<<blocks, SutureAtomTR::THREADS, kSTR, stream>>>(kArgs, kCtx);
        }
      }
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const auto maxCTAs = cute::min(suture::MAX_NUM_CTAS, num_sms);
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    const auto putBlocks = bytes <= CHUNK_SIZE ? nonChunkedConfig::PUT_BLOCKS : chunkedConfig::PUT_BLOCKS;
    const auto transferBlocks = putBlocks + (world == 2 ? 0 : GATHER_BLOCKS);
    const auto maxReduceBlocks = cute::min(opts.maxReduceBlocks,
    cuda::std::bit_floor(static_cast<uint32_t>(num_sms - transferBlocks)));
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
    const auto isLR = suture::getRedRegime(bytes, world) == suture::Regime::latency;
    size_t blocks = 0;
    if (isLR) {
      blocks = cute::min(cuda::ceil_div(bytes, SutureAtomLR::THREADS*sizeof(suture::LRP16::RT)), CTAsUpperLR);
    }
    else {
      auto blocksNeeded = cute::min(bytes / SutureAtomTR::RED_PIPELINE_BYTES,
        bytes / (world * SutureAtomTR::STAGE_BYTES));
      blocksNeeded = static_cast<int>(min(blocksNeeded,static_cast<size_t>(maxReduceBlocks)));
      blocks = transferBlocks + blocksNeeded;
      if (blocksNeeded < 1) {
        // non-pipelined path
        blocks = transferBlocks + cute::min(cuda::ceil_div(bytes / world,
          SutureAtomLR::THREADS*sizeof(SutureAtomTR::BaseConfig::ALIGNMENT_BYTES)), maxReduceBlocks);
      }
    }
    if (blocks < 1) {
      throw std::runtime_error("Blocks must be >= 1");
    }
    if (blocks > maxCTAs) {
      throw std::runtime_error("Blocks must be <= " + std::to_string(maxCTAs));
    }
    const Args kArgs{
      .src = srcBuff,
      .dst = srcBuff,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{static_cast<long int>(blocks)}
    };
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
    // bitwise correctness check against nccl
    (ar_matches0 = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);
    auto tO = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(dstBuff), {1, static_cast<matx::index_t>(elems)});
    // bitwise correctness check against oracle
    auto ar_matches1 = matx::make_tensor<long int>({});
    (ar_matches1 = matx::sum(matx::isclose(tR, tO, 0, 0))).run(exec);
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
      printf("%d, %lu, %s, %lf, %lf, %lf, %lf, %d, %s, %d, %s, %s, %s, %d, %d, %s, %s, %s, %d, %s, %d, %d, %d\n",
        world, bytes, element_string<DataType>(), times.t_ms, suture_algBW, times.oracle_ep, times.ep,
        nArch, prop.name,
        isLR ? SutureAtomLR::THREADS : SutureAtomTR::THREADS,
        isLR ? "N/A" : std::to_string(pipeStages).c_str(),
        isLR ? "N/A" : std::to_string(elementsPerThread).c_str(),
        isLR ? "N/A" : std::to_string(unrollFactor).c_str(),
        worldUnroll,
        num_sms,
        isLR ? "N/A" : std::to_string(putBlocks).c_str(),
        isLR ? "N/A" : std::to_string(blocks - transferBlocks).c_str(),
        isLR || world == 2 ? "N/A" : std::to_string(GATHER_BLOCKS).c_str(),
        static_cast<int>(blocks),
        isLR ? "N/A" : std::to_string(CHUNK_SIZE / (1024UL * 1024)).c_str(),
        opts.graph_launches > 0 ? opts.runs : opts.warmup,
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

// ./ar <minLocalBytes> <maxLocalBytes> <maxReduceBlocks> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.maxReduceBlocks = 32;
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
  if (opts.minLocalBytes % suture::MAX_ACCESS_ALIGNMENT != 0 || opts.maxLocalBytes % suture::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(suture::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  if (opts.maxLocalBytes > suture::STAGING_BUFFER_SIZE_) {
    // TODO: add staging multiplexing
    throw std::invalid_argument("maxLocalBytes: " + std::to_string(opts.maxLocalBytes) +
      " exceeds staging buffer capacity: " + std::to_string(suture::STAGING_BUFFER_SIZE_));
  }
  arHost(opts);
}
