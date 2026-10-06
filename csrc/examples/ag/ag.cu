//
// Created by osayamen on 2/18/26.
//
#include <string>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>

#include <purlin/core.cuh>
#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <contrib/symm_mem.cuh>

struct Options : bench::Options {
  int maxSuperBlockSize = 32;
};

constexpr auto threads = 128;
constexpr auto unrollFactor = 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 8;
constexpr auto elementsPerThread = 8;

constexpr auto nArch = purlin::normalizeArch<ARCH>();
constexpr auto worldUnroll = 2;
using TRConfig = purlin::Configuration<
    threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor
>;
constexpr auto t128Lower = 128 * 1024;
constexpr auto t128Higher = 1024 * 1024;
using TR128Config = purlin::Configuration<
        128, /*threads*/
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor
>;

using LRConfig = purlin::Configuration<
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

__host__ __forceinline__
  constexpr auto getRegime(const size_t& bytesPerRank, const int& world) {
  switch (world) {
    case 4: {
      if (bytesPerRank <= 128 * 1024) {
        return purlin::Regime::latency;
      }
      return purlin::Regime::throughput;
    }
      break;
    case 8: {
      if (bytesPerRank <= 128 * 1024) {
        return purlin::Regime::latency;
      }
      return purlin::Regime::throughput;
    }
      break;
    default: {
      if (bytesPerRank <= 512 * 1024) {
        return purlin::Regime::latency;
      }
      return purlin::Regime::throughput;
    }
  }
}

// 2MiB -> 4MiB <= globalBytes <= 16MiB
// 4MiB -> 32MiB <= globalBytes <= 128MiB
//constexpr size_t CHUNK_SIZE = 8 * 1024 * 1024;
constexpr size_t CHUNK_SIZE = 4 * 1024 * 1024;
constexpr auto PUT_BLOCKS = 32; // 16 or 32
template<typename PurlinAtom, typename CollConfig>
__launch_bounds__(PurlinAtom::THREADS, 1)
__global__ void allGather(const __grid_constant__ Args kArgs, const __grid_constant__ purlin::Context ctx) {
  extern __shared__ __align__(bench::sharedMemoryAlignment) cuda::std::byte workspace[];
  const purlin::SnacArgs<cuda::fast_mod_div<long int>> args{
    .dst = kArgs.dst,
    .src = kArgs.src,
    .bytes = kArgs.bytes,
    .workspace = workspace,
    .blocks = kArgs.blocks,
    .collBlocks = static_cast<int>(kArgs.blocks),
  };
  purlin::allGather<PurlinAtom, CollConfig>(args, ctx);
}

__host__
void agHost(Options& opts) {
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
    printf("world,localBytes,globalBytes,purlin(ms),purlin(GB/s),error(%%),nArch,GPUName,threads,"
           "pipeStages,stageExtent,unrollFactor,worldUnroll,"
           "SMsOnGPU,putBlocks,consumerBlocks,blocks,chunkSize(MiB),warmup,runs,graph_launches\n");
  }
  if (world > purlin::MAX_RANKS_PER_DOMAIN) {
    throw std::runtime_error(std::to_string(world) + "exceeds max allowed of " +
      std::to_string(purlin::MAX_RANKS_PER_DOMAIN) + "ranks");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  auto managed = purlin::initialize(rank, world, stream, purlin::NvshmemMemory{});
  auto& ctx = managed.context();
  using PurlinAtomLR = purlin::Atom<nArch, LRConfig>;
  using PurlinAtomTR = purlin::Atom<nArch, TRConfig>;
  using PurlinAtomTR128 = purlin::Atom<nArch, TR128Config>;
  using nonChunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::nonChunked,
    PUT_BLOCKS,
    purlin::UNUSED,
    CHUNK_SIZE
  >;
  using chunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::chunked,
    PUT_BLOCKS,
    purlin::UNUSED,
    CHUNK_SIZE
  >;
  constexpr auto kSTR = purlin::copySmemBytes<PurlinAtomTR>();
  constexpr auto kSTR128 = purlin::copySmemBytes<PurlinAtomTR128>();
  constexpr auto kSLR = purlin::copySmemBytes<PurlinAtomLR, purlin::Regime::latency>();
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  auto kernelTRNonChunked = allGather<PurlinAtomTR, nonChunkedConfig>;
  auto kernelTR128NonChunked = allGather<PurlinAtomTR128, nonChunkedConfig>;
  auto kernelTRChunked = allGather<PurlinAtomTR, chunkedConfig>;
  auto kernelLR = allGather<PurlinAtomLR, purlin::CollectiveConfigLR>;
  {
    if (kSTR > maxSharedMemory) {
      const auto errmsg = std::string("Required shared memory ").append(std::to_string(kSTR))
      .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
      throw std::runtime_error(errmsg);
    }
    CHECK_CUDA(cudaFuncSetAttribute(kernelTRNonChunked, cudaFuncAttributeMaxDynamicSharedMemorySize, kSTR));
    CHECK_CUDA(cudaFuncSetAttribute(kernelTR128NonChunked, cudaFuncAttributeMaxDynamicSharedMemorySize, kSTR128));
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
  opts.maxSuperBlockSize = opts.maxSuperBlockSize <= 0 ? (world == 2 ? 32 : (32 / world)) : opts.maxSuperBlockSize;
  const auto CTAsUpperLR = cuda::std::min(64U, cuda::std::bit_floor(static_cast<uint32_t>(num_sms)));

  CHECK_CUDA(cudaMallocAsync(&srcBuff, opts.maxBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&dstBuff, opts.maxBytes * world, stream));
  CHECK_CUDA(cudaMallocAsync(&refBuff, opts.maxBytes * world, stream));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  const auto seed = bench::broadcastRandomSeed(rank);
  bench::reportSeed(rank, seed);
  auto agk = [&](const auto& blocks, const Args& kArgs, const purlin::Context& kCtx, const bool isLR, const int& runs) {
    if (isLR) {
      for (int i = 0; i < runs; ++i) {
        allGather<PurlinAtomLR, purlin::CollectiveConfigLR>
        <<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>(kArgs, kCtx);
      }
    }
    else {
      if (kArgs.bytes <= CHUNK_SIZE) {
        if (world >= 4 && (kArgs.bytes >= t128Lower && kArgs.bytes <= t128Higher)) {
          for (int i = 0; i < runs; ++i) {
            allGather<PurlinAtomTR128, nonChunkedConfig>
            <<<blocks, PurlinAtomTR128::THREADS, kSTR128, stream>>>(kArgs, kCtx);
          }
        }
        else {
          for (int i = 0; i < runs; ++i) {
            allGather<PurlinAtomTR, nonChunkedConfig>
            <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, kCtx);
          }
        }
      }
      else {
        for (int i = 0; i < runs; ++i) {
          allGather<PurlinAtomTR, chunkedConfig>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, kCtx);
        }
      }
    }
  };
  matx::cudaExecutor exec{stream};
  bench::Measurement measurement{};
  for (size_t localBytes = opts.minBytes; localBytes <= opts.maxBytes; localBytes *= 2) {
    const auto putBlocks = localBytes <= CHUNK_SIZE ? nonChunkedConfig::PUT_BLOCKS : chunkedConfig::PUT_BLOCKS;
    const auto superUpper = cuda::round_down(
      cuda::std::bit_floor(static_cast<uint32_t>(num_sms - putBlocks)), world) / world;
    const auto maxSuperBlockSize = cuda::std::min(static_cast<uint>(opts.maxSuperBlockSize), superUpper);
    // fill buffer with this rank's seeded random stream
    static_assert(purlin::MAX_ACCESS_ALIGNMENT % sizeof(float) == 0);
    const auto elems = localBytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(srcBuff);
    bench::fillRandomReduction(tS, elems, bench::gatherSeed(seed, rank), stream);
    const auto isLR = getRegime(localBytes, world) == purlin::Regime::latency;
    int blocks = 0;
    if (isLR) {
      blocks = cuda::std::min(cuda::ceil_div(localBytes, PurlinAtomLR::THREADS*sizeof(purlin::LRP::RT)),
        static_cast<size_t>(CTAsUpperLR));
    }
    else {
      auto blocksNeeded = static_cast<int>(min((localBytes / PurlinAtomTR::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxSuperBlockSize)) * world);
      blocksNeeded = localBytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
      cuda::std::min(blocksNeeded, 32) : blocksNeeded;
      blocks = putBlocks + blocksNeeded;
      if (blocksNeeded < world) {
        // non-pipelined path
        blocks = putBlocks + (cuda::std::min(cuda::ceil_div(localBytes,
          static_cast<size_t>(PurlinAtomTR::THREADS*PurlinAtomTR::BaseConfig::ALIGNMENT_BYTES)),
          static_cast<size_t>(maxSuperBlockSize)) * world);
      }
    }
    const Args kArgs{
      .src = srcBuff,
      .dst = dstBuff,
      .bytes = localBytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    // correctness run: replay every peer's seeded fill locally as the reference
    agk(blocks, kArgs, ctx, isLR, 1);
    auto* tRefFill = reinterpret_cast<float*>(refBuff);
    for (int peer = 0; peer < world; ++peer) {
      bench::fillRandomReduction(tRefFill + peer * elems, elems,
        bench::gatherSeed(seed, peer), stream);
    }
    auto ag_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(dstBuff), {1, static_cast<matx::index_t>(elems * world)});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff), {1, static_cast<matx::index_t>(elems * world)});
    // bitwise check
    (ag_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);

    float t_ms = 0.0f;
    if (opts.graphLaunches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      agk(blocks, kArgs, ctx, isLR, opts.runs);
      CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

      CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // warmup
      CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // time total launches = opts.runs * opts.graphLaunches
      const int total_launches = opts.runs * opts.graphLaunches;

      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < opts.graphLaunches; ++i) {
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
      agk(blocks, kArgs, ctx, isLR, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      agk(blocks, kArgs, ctx, isLR, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }

    measurement.errorPercentage = (1.0 - (static_cast<double>(ag_matches()) / static_cast<double>(tR.TotalSize()))) * 100.0;
    measurement.milliseconds = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &measurement, sizeof(bench::Measurement) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const auto usedThreads = (world >= 4 && kArgs.bytes >= t128Lower && kArgs.bytes <= t128Higher) ?
    PurlinAtomTR128::THREADS : PurlinAtomTR::THREADS;
    if (rank == 0) {
      const auto gb = (world * static_cast<double>(localBytes)) / 1e9;
      const auto purlin_algBW = gb / (measurement.milliseconds * 1e-3);
      printf("%d, %lu, %lu, %lf, %lf, %lf, %d, %s, %d, %s, %s, %s, %s, %d, %s, %s, %d, %s, %d, %d, %d\n",
        world, localBytes, world * localBytes,measurement.milliseconds, purlin_algBW, measurement.errorPercentage, nArch,
        prop.name,
        isLR ? PurlinAtomLR::THREADS : usedThreads,
        isLR ? "N/A" : std::to_string(pipeStages).c_str(),
        isLR ? "N/A" : std::to_string(elementsPerThread).c_str(),
        isLR ? "N/A" : std::to_string(unrollFactor).c_str(),
        isLR ? std::to_string(worldUnroll).c_str() : "N/A",
        num_sms,
        isLR ? "N/A" : std::to_string(putBlocks).c_str(),
        isLR ? "N/A" : std::to_string(blocks - putBlocks).c_str(),
        blocks,
        isLR ? "N/A" : std::to_string(CHUNK_SIZE / (1024UL * 1024)).c_str(),
        opts.graphLaunches > 0 ? opts.runs : opts.warmup, opts.runs,opts.graphLaunches);
    }
  }
  CHECK_CUDA(cudaFreeAsync(srcBuff, stream));
  CHECK_CUDA(cudaFreeAsync(dstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  purlin::finalize(managed, stream);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_finalize();
}
// ./ag <minLocalBytes> <maxLocalBytes> <maxSuperBlockSize> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  Options opts{};
  opts.maxSuperBlockSize = -1; // -1 -> autotuned
  opts.graphLaunches = 8;
  opts.warmup = 128;
  opts.runs = 128;
  if (argc > 1) opts.minBytes = bench::parseSize(argv[1]);
  if (argc > 2) opts.maxBytes = bench::parseSize(argv[2]);
  if (argc > 3) opts.maxSuperBlockSize = std::stoi(argv[3]);
  if (argc > 4) opts.graphLaunches = std::stoi(argv[4]);
  if (argc > 5) opts.runs = std::stoi(argv[5]);
  if (argc > 6) opts.warmup = std::stoi(argv[6]);
  if (!cuda::is_power_of_two(opts.minBytes) || !cuda::is_power_of_two(opts.maxBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minBytes % purlin::MAX_ACCESS_ALIGNMENT != 0 || opts.maxBytes % purlin::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(purlin::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  agHost(opts);
}
