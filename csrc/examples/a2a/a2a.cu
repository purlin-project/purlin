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

#include <purlin/host/codesign.cuh>

constexpr auto threads = 128;
constexpr auto unrollFactor = 4;
constexpr auto alignment = 16;

constexpr auto pipeStages = 8;
constexpr auto elementsPerThread = 2;

constexpr auto nArch = purlin::normalizeArch<ARCH>();
constexpr auto worldUnroll = 2;

template<int NArch>
__host__ constexpr size_t all2allLatencyThreshold(const int world) {
  switch (world) {
    case 2: return purlin::host::All2AllCodesign<NArch, 2>::LATENCY_THRESHOLD;
    case 4: return purlin::host::All2AllCodesign<NArch, 4>::LATENCY_THRESHOLD;
    case 8: return purlin::host::All2AllCodesign<NArch, 8>::LATENCY_THRESHOLD;
    default: return purlin::host::All2AllCodesign<NArch, purlin::host::FALLBACK>::LATENCY_THRESHOLD;
  }
}

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
// 2MiB -> 4MiB <= globalBytes <= 16MiB
// 4MiB -> 32MiB <= globalBytes <= 128MiB
constexpr size_t CHUNK_SIZE = 2 * 1024 * 1024;
constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
constexpr int CHUNKED_PUT_BLOCKS = 32;
constexpr int LOCAL_PUT_BLOCKS = 8;
template<typename PurlinAtom, typename CollConfig>
__launch_bounds__(PurlinAtom::THREADS, 1)
__global__ void all2all(const __grid_constant__ Args kArgs, const __grid_constant__ purlin::Context ctx) {
  extern __shared__ __align__(bench::sharedMemoryAlignment) cuda::std::byte workspace[];
  const purlin::SnacArgs<cuda::fast_mod_div<long int>> args{
    .dst = kArgs.dst,
    .src = kArgs.src,
    .bytes = kArgs.bytes,
    .workspace = workspace,
    .blocks = kArgs.blocks,
    .collBlocks = static_cast<int>(kArgs.blocks),
  };
  purlin::all2all<PurlinAtom, CollConfig>(args, ctx);
}

__host__
void a2aHost(const bench::KernelOptions& options) {
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
           "SMsOnGPU,stagingBlocks,localPutBlocks,consumerBlocks,blocks,chunkSize(MiB),warmup,runs,graph_launches\n");
  }
  if (NON_CHUNKED_PUT_BLOCKS % world != 0) {
    throw std::runtime_error("non-chunked put blocks: " + std::to_string(NON_CHUNKED_PUT_BLOCKS) +
      " must be a multiple of world");
  }
  if (CHUNKED_PUT_BLOCKS % world != 0) {
    throw std::runtime_error("chunked put blocks: " + std::to_string(CHUNKED_PUT_BLOCKS) +
      " must be a multiple of world");
  }
  using PurlinAtomLR = purlin::Atom<nArch, LRConfig>;
  using PurlinAtomTR = purlin::Atom<nArch, TRConfig>;
  using PurlinAtomTR128 = purlin::Atom<nArch, TR128Config>;
  using nonChunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::nonChunked,
    purlin::UNUSED,
    purlin::UNUSED,
    CHUNK_SIZE,
    LOCAL_PUT_BLOCKS
  >;
  using chunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::chunked,
    purlin::UNUSED,
    purlin::UNUSED,
    CHUNK_SIZE,
    LOCAL_PUT_BLOCKS
  >;
  constexpr auto kSTR = purlin::copySmemBytes<PurlinAtomTR>();
  constexpr auto kSTR128 = purlin::copySmemBytes<PurlinAtomTR128>();
  constexpr auto kSLR = purlin::copySmemBytes<PurlinAtomLR, purlin::Regime::latency>();
  bench::configureKernel(all2all<PurlinAtomTR, nonChunkedConfig>, kSTR, prop);
  bench::configureKernel(all2all<PurlinAtomTR128, nonChunkedConfig>, kSTR128, prop);
  bench::configureKernel(all2all<PurlinAtomTR, chunkedConfig>, kSTR, prop);
  bench::configureKernel(all2all<PurlinAtomLR, purlin::CollectiveConfigLR>, kSLR, prop);
  const auto blockLimit = options.maxBlocks <= 0 ? (world == 2 ? 32 : 32 / world) : options.maxBlocks;
  const auto CTAsUpperLR = cuda::std::min(64U, cuda::std::bit_floor(static_cast<uint32_t>(num_sms)));
  const auto maxCTAs = cuda::std::min(static_cast<size_t>(num_sms), purlin::MAX_NUM_CTAS);

  const size_t maximumTotal = bench::checkedMultiply(options.maxBytes, world);
  bench::DeviceBuffer<cuda::std::byte> source(maximumTotal, stream);
  bench::DeviceBuffer<cuda::std::byte> destination(maximumTotal, stream);
  bench::DeviceBuffer<cuda::std::byte> reference(maximumTotal, stream);
  const auto seed = bench::broadcastRandomSeed(rank, options.seed);
  bench::reportSeed(rank, seed);
  const auto actualWorld = world - 1;
  const auto nNonChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
    cuda::round_down(NON_CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
  const auto nChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
    cuda::round_down(CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t localBytes) {
    const auto stagingBlocks = (localBytes <= CHUNK_SIZE ? nNonChunkedPB : nChunkedPB)* actualWorld;
    const auto putBlocks = stagingBlocks + LOCAL_PUT_BLOCKS;
    ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
    const size_t total = bench::checkedMultiply(localBytes, world);
    // Each source/destination pair has its own reproducible byte stream.
    for (int peer = 0; peer < world; ++peer) {
      bench::fillRandomBytes(source.get() + peer * localBytes, localBytes,
        bench::pairSeed(seed, rank, peer), stream);
      bench::fillRandomBytes(reference.get() + peer * localBytes, localBytes,
        bench::pairSeed(seed, peer, rank), stream);
    }
    const auto isLR = localBytes <= all2allLatencyThreshold<nArch>(world);
    int blocks = 0;
    if (isLR) {
      blocks = cuda::std::min(cuda::ceil_div(localBytes, PurlinAtomLR::THREADS*sizeof(purlin::LRP::RT)),
        static_cast<size_t>(CTAsUpperLR));
    }
    else {
      if (putBlocks + actualWorld > maxCTAs) {
        throw std::runtime_error("Not enough blocks for all-to-all producers and consumers");
      }
      const auto superUpper = cuda::std::bit_floor((maxCTAs - putBlocks) / actualWorld);
      const auto maxSuperBlockSize = cuda::std::min(static_cast<size_t>(blockLimit), superUpper);
      auto blocksNeeded = static_cast<int>(min((localBytes / PurlinAtomTR::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxSuperBlockSize)) * actualWorld);
      blocksNeeded = localBytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
      cuda::round_down(cuda::std::min(blocksNeeded, 32), actualWorld) : blocksNeeded;
      blocks = putBlocks + blocksNeeded;
      if (blocksNeeded < actualWorld) {
        // non-pipelined path
        blocks = putBlocks + (cuda::std::min(cuda::ceil_div(localBytes,
          static_cast<size_t>(PurlinAtomTR::THREADS*PurlinAtomTR::BaseConfig::ALIGNMENT_BYTES)),
          static_cast<size_t>(maxSuperBlockSize)) * actualWorld);
      }
    }
    if (blocks < 1 || static_cast<size_t>(blocks) > maxCTAs) {
      throw std::runtime_error("All-to-all grid exceeds the supported block count");
    }
    const Args kArgs{
      .src = source.get(),
      .dst = destination.get(),
      .bytes = localBytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    const auto operation = [&] {
      if (isLR) {
        all2all<PurlinAtomLR, purlin::CollectiveConfigLR>
          <<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>(kArgs, ctx);
      } else if (localBytes > CHUNK_SIZE) {
        all2all<PurlinAtomTR, chunkedConfig>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
      } else if (world >= 4 && localBytes >= t128Lower && localBytes <= t128Higher) {
        all2all<PurlinAtomTR128, nonChunkedConfig>
          <<<blocks, PurlinAtomTR128::THREADS, kSTR128, stream>>>(kArgs, ctx);
      } else {
        all2all<PurlinAtomTR, nonChunkedConfig>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
      }
    };
    operation();
    CHECK_CUDA(cudaGetLastError());
    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxByteMismatches(destination.get(), reference.get(), total, stream), total);
    const double milliseconds = bench::measureOperation(stream, MPI_COMM_WORLD, options, operation);
    const auto usedThreads = (world >= 4 && kArgs.bytes >= t128Lower && kArgs.bytes <= t128Higher) ?
    PurlinAtomTR128::THREADS : PurlinAtomTR::THREADS;
    if (rank == 0) {
      printf("%d, %lu, %lu, %lf, %lf, %lf, %d, %s, %d, %s, %s, %s, %s, %d, %s, %s, %s, %d, %s, %d, %d, %d\n",
        world, localBytes, world * localBytes, milliseconds, bench::bandwidth(total, milliseconds), errorPercentage, nArch,
        prop.name,
        isLR ? PurlinAtomLR::THREADS : usedThreads,
        isLR ? "N/A" : std::to_string(pipeStages).c_str(),
        isLR ? "N/A" : std::to_string(elementsPerThread).c_str(),
        isLR ? "N/A" : std::to_string(unrollFactor).c_str(),
        isLR ? std::to_string(worldUnroll).c_str() : "N/A",
        num_sms,
        isLR ? "N/A" : std::to_string(stagingBlocks).c_str(),
        isLR ? "N/A" : std::to_string(LOCAL_PUT_BLOCKS).c_str(),
        isLR ? "N/A" : std::to_string(blocks - putBlocks).c_str(),
        blocks,
        isLR ? "N/A" : std::to_string(CHUNK_SIZE / (1024UL * 1024)).c_str(),
        bench::effectiveWarmup(options.graphLaunches, options.runs, options.warmup), options.runs,options.graphLaunches);
    }
  });
}

// ./a2a [minBytes] [maxBytes] [maxBlocks] [graphLaunches] [runs] [warmup] [seed]
int main(int argc, char** argv) {
  try {
    const auto options = bench::parseKernelOptions(argc, argv);
    bench::validatePurlinOptions(options);
    a2aHost(options);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
