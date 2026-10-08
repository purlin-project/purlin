#include <cstddef>
#include <stdexcept>
#include <string>

#include <purlin/core.cuh>
#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/kernel.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

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
void agHost(const bench::KernelOptions& options) {
  bench::PurlinRuntime runtime;
  const auto world = runtime.world;
  const auto rank = runtime.rank;
  const auto stream = runtime.stream;
  const auto& prop = runtime.deviceProperties;
  auto& ctx = runtime.context;
  const auto num_sms = prop.multiProcessorCount;
  if (rank == 0) {
    printf("world,localBytes,globalBytes,purlin(ms),purlin(GB/s),error(%%),nArch,GPUName,threads,"
           "pipeStages,stageExtent,unrollFactor,worldUnroll,"
           "SMsOnGPU,putBlocks,consumerBlocks,blocks,chunkSize(MiB),warmup,runs,graph_launches\n");
  }
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
  bench::configureKernel(allGather<PurlinAtomTR, nonChunkedConfig>, kSTR, prop);
  bench::configureKernel(allGather<PurlinAtomTR128, nonChunkedConfig>, kSTR128, prop);
  bench::configureKernel(allGather<PurlinAtomTR, chunkedConfig>, kSTR, prop);
  bench::configureKernel(allGather<PurlinAtomLR, purlin::CollectiveConfigLR>, kSLR, prop);
  const auto blockLimit = options.maxBlocks <= 0 ? (world == 2 ? 32 : 32 / world) : options.maxBlocks;
  const auto CTAsUpperLR = cuda::std::min(64U, cuda::std::bit_floor(static_cast<uint32_t>(num_sms)));
  const auto maxCTAs = cuda::std::min(static_cast<size_t>(num_sms), purlin::MAX_NUM_CTAS);

  const size_t maximumTotal = bench::checkedMultiply(options.maxBytes, world);
  bench::DeviceBuffer<cuda::std::byte> source(options.maxBytes, stream);
  bench::DeviceBuffer<cuda::std::byte> destination(maximumTotal, stream);
  bench::DeviceBuffer<cuda::std::byte> reference(maximumTotal, stream);
  const auto seed = bench::broadcastRandomSeed(rank, options.seed);
  bench::reportSeed(rank, seed);
  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t localBytes) {
    const auto putBlocks = localBytes <= CHUNK_SIZE ? nonChunkedConfig::PUT_BLOCKS : chunkedConfig::PUT_BLOCKS;
    const size_t total = bench::checkedMultiply(localBytes, world);
    bench::fillRandomBytes(source.get(), localBytes, bench::gatherSeed(seed, rank), stream);
    const auto isLR = getRegime(localBytes, world) == purlin::Regime::latency;
    int blocks = 0;
    if (isLR) {
      blocks = cuda::std::min(cuda::ceil_div(localBytes, PurlinAtomLR::THREADS*sizeof(purlin::LRP::RT)),
        static_cast<size_t>(CTAsUpperLR));
    }
    else {
      if (static_cast<size_t>(putBlocks + world) > maxCTAs) {
        throw std::runtime_error("Not enough blocks for all-gather producers and consumers");
      }
      // The context only has synchronization storage for MAX_NUM_CTAS blocks.
      const auto superUpper = cuda::round_down(cuda::std::bit_floor(maxCTAs - putBlocks), world) / world;
      const auto maxSuperBlockSize = cuda::std::min(static_cast<size_t>(blockLimit), superUpper);
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
    if (blocks < 1 || static_cast<size_t>(blocks) > maxCTAs) {
      throw std::runtime_error("All-gather grid exceeds the supported block count");
    }
    const Args kArgs{
      .src = source.get(),
      .dst = destination.get(),
      .bytes = localBytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    const auto operation = [&] {
      if (isLR) {
        allGather<PurlinAtomLR, purlin::CollectiveConfigLR>
          <<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>(kArgs, ctx);
      } else if (localBytes > CHUNK_SIZE) {
        allGather<PurlinAtomTR, chunkedConfig>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
      } else if (world >= 4 && localBytes >= t128Lower && localBytes <= t128Higher) {
        allGather<PurlinAtomTR128, nonChunkedConfig>
          <<<blocks, PurlinAtomTR128::THREADS, kSTR128, stream>>>(kArgs, ctx);
      } else {
        allGather<PurlinAtomTR, nonChunkedConfig>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
      }
    };
    operation();
    CHECK_CUDA(cudaGetLastError());
    for (int peer = 0; peer < world; ++peer) {
      bench::fillRandomBytes(reference.get() + peer * localBytes, localBytes,
        bench::gatherSeed(seed, peer), stream);
    }
    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxByteMismatches(destination.get(), reference.get(), total, stream), total);
    const double milliseconds = bench::measureOperation(stream, MPI_COMM_WORLD, options, operation);
    const auto usedThreads = (world >= 4 && kArgs.bytes >= t128Lower && kArgs.bytes <= t128Higher) ?
    PurlinAtomTR128::THREADS : PurlinAtomTR::THREADS;
    if (rank == 0) {
      printf("%d, %lu, %lu, %lf, %lf, %lf, %d, %s, %d, %s, %s, %s, %s, %d, %s, %s, %d, %s, %d, %d, %d\n",
        world, localBytes, world * localBytes, milliseconds, bench::bandwidth(total, milliseconds), errorPercentage, nArch,
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
        bench::effectiveWarmup(options.graphLaunches, options.runs, options.warmup), options.runs,options.graphLaunches);
    }
  });
}

// ./ag [minBytes] [maxBytes] [maxBlocks] [graphLaunches] [runs] [warmup] [seed]
int main(int argc, char** argv) {
  try {
    const auto options = bench::parseKernelOptions(argc, argv);
    bench::validatePurlinOptions(options);
    agHost(options);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
