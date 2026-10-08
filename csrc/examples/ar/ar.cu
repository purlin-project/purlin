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

constexpr auto threads = 128; // A100: 256;
constexpr auto unrollFactor = 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 8; //A100: 8;
constexpr auto elementsPerThread = 2; // A100: 2;
constexpr auto worldUnroll = 2;
constexpr auto nArch = purlin::normalizeArch<ARCH>();
using TRConfig = purlin::Configuration<
    threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor,
    worldUnroll
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

using DataType = __half;
// 2MiB -> 4MiB <= bytes <= 16MiB,
// 4MiB -> 32MiB <= bytes <= 128MiB
// 8MiB -> 256 MiB <=  bytes
constexpr size_t CHUNK_SIZE = 2 * 1024 * 1024;
constexpr int NON_CHUNKED_PUT_BLOCKS = 16;
constexpr int CHUNKED_PUT_BLOCKS = 16;
constexpr int GATHER_BLOCKS = 16;

__host__ __forceinline__
  auto getRedRegime(const size_t& bytes, const int& world) {
  if (world == 8) {
    if (bytes <= 128 * 1024) {
      return purlin::Regime::latency;
    }
    return purlin::Regime::throughput;
  }
  if (bytes <= purlin::RED_LATENCY_BOUND_THRESHOLD) {
    return purlin::Regime::latency;
  }
  return purlin::Regime::throughput;
}

template<typename PurlinAtom, typename Element, typename CollConfig, purlin::AllReducePath path>
__launch_bounds__(PurlinAtom::THREADS, 1)
__global__ void allReduce(const __grid_constant__ Args kArgs, const __grid_constant__ purlin::Context ctx) {
  extern __shared__ __align__(bench::sharedMemoryAlignment) cuda::std::byte workspace[];
  const purlin::SnacArgs<cuda::fast_mod_div<long int>> args{
    .dst = kArgs.dst,
    .src = kArgs.src,
    .bytes = kArgs.bytes,
    .workspace = workspace,
    .blocks = kArgs.blocks,
    .collBlocks = static_cast<int>(kArgs.blocks),
  };
  // this collective could be fused into a larger kernel
  purlin::allReduce<PurlinAtom, CollConfig, path, purlin::ReduceOp::add, Element>(args, ctx);
}

__host__
void arHost(const bench::KernelOptions& options) {
  const bool predictable = ARCH >= 900 && options.reductionMode == purlin::ReductionMode::nonDeterministic;
  bench::PurlinRuntime runtime;
  const auto world = runtime.world;
  const auto rank = runtime.rank;
  const auto stream = runtime.stream;
  const auto& prop = runtime.deviceProperties;
  auto& ctx = runtime.context;
  const auto num_sms = prop.multiProcessorCount;
  if (NON_CHUNKED_PUT_BLOCKS % world != 0) {
    throw std::runtime_error("non-chunked put blocks: " + std::to_string(NON_CHUNKED_PUT_BLOCKS) +
      " must be a multiple of world");
  }
  if (CHUNKED_PUT_BLOCKS % world != 0) {
    throw std::runtime_error("chunked put blocks: " + std::to_string(CHUNKED_PUT_BLOCKS) +
      " must be a multiple of world");
  }
  if (GATHER_BLOCKS % world != 0) {
    throw std::runtime_error("gather blocks: " + std::to_string(GATHER_BLOCKS) + " must be a multiple of world");
  }
  if (rank == 0) {
    printf("world,bytes,datatype,purlin(ms),purlin(GB/s),error_vs_oracle(%%),"
           "nArch,GPUName,threads,pipeStages,stageExtent,unrollFactor,worldUnroll,"
           "SMsOnGPU,putBlocks,reduceBlocks,gatherBlocks,blocks,chunkSize(MiB),warmup,runs,graph_launches\n");
  }
  using PurlinAtomLR = purlin::Atom<nArch, LRConfig>;
  using PurlinAtomTR = purlin::Atom<nArch, TRConfig>;
  using nonChunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::nonChunked,
    NON_CHUNKED_PUT_BLOCKS,
    GATHER_BLOCKS,
    CHUNK_SIZE
  >;
  using chunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::chunked,
    CHUNKED_PUT_BLOCKS,
    GATHER_BLOCKS,
    CHUNK_SIZE
  >;
  constexpr auto kSTR = purlin::snacSmemBytes<PurlinAtomTR>();
  constexpr auto kSLR = purlin::redSmemBytes<PurlinAtomLR, purlin::Regime::latency>();
  bench::configureKernel(
    allReduce<PurlinAtomLR, DataType, purlin::CollectiveConfigLR, purlin::AllReducePath::direct>, kSLR, prop);
  if (world == 2) {
    bench::configureKernel(
      allReduce<PurlinAtomTR, DataType, nonChunkedConfig, purlin::AllReducePath::direct>, kSTR, prop);
    bench::configureKernel(
      allReduce<PurlinAtomTR, DataType, chunkedConfig, purlin::AllReducePath::direct>, kSTR, prop);
  } else {
    bench::configureKernel(
      allReduce<PurlinAtomTR, DataType, nonChunkedConfig, purlin::AllReducePath::composed>, kSTR, prop);
    bench::configureKernel(
      allReduce<PurlinAtomTR, DataType, chunkedConfig, purlin::AllReducePath::composed>, kSTR, prop);
  }
  const auto CTAsUpperLR = cuda::std::min(64U, cuda::std::bit_floor(static_cast<uint32_t>(num_sms)));
  const auto blockLimit = options.maxBlocks <= 0 ? (world == 2 ? 32 : 16) : options.maxBlocks;
  const size_t maximumElements = options.maxBytes / sizeof(DataType);
  const size_t maximumTotal = bench::checkedMultiply(maximumElements, world);
  bench::DeviceBuffer<DataType> source(maximumElements, stream);
  bench::DeviceBuffer<DataType> destination(maximumElements, stream);
  bench::DeviceBuffer<DataType> referenceSources(maximumTotal, stream);
  bench::DeviceBuffer<DataType> reference(maximumElements, stream);
  auto* sourceBytes = reinterpret_cast<cuda::std::byte*>(source.get());
  auto* destinationBytes = reinterpret_cast<cuda::std::byte*>(destination.get());
  const auto seed = bench::broadcastRandomSeed(rank, options.seed);
  bench::reportSeed(rank, seed);
  const auto maxCTAs = cuda::std::min(purlin::MAX_NUM_CTAS, static_cast<size_t>(num_sms));
  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
    const auto localBytes = bytes / world;
    const auto putBlocks = (world == 2 ? bytes : localBytes) <= CHUNK_SIZE ? nonChunkedConfig::PUT_BLOCKS : chunkedConfig::PUT_BLOCKS;
    const auto transferBlocks = putBlocks + (world == 2 ? 0 : GATHER_BLOCKS);
    const auto maxReduceBlocks = cuda::std::min(static_cast<uint>(blockLimit),
    cuda::std::bit_floor(static_cast<uint32_t>(num_sms - transferBlocks)));
    const size_t elements = bytes / sizeof(DataType);
    bench::fillRandomReduction(source.get(), elements, bench::allReduceSeed(seed, rank), stream, predictable);
    bench::fillRandomAllReduceReferenceSources(referenceSources.get(), elements, seed, world, stream, predictable);
    bench::computeReductionReference(referenceSources.get(), reference.get(), elements, world, stream);
    const auto isLR = getRedRegime(bytes, world) == purlin::Regime::latency;
    size_t blocks = 0;
    if (isLR) {
      blocks = cuda::std::min(cuda::ceil_div(bytes, PurlinAtomLR::THREADS*sizeof(purlin::LRP::RT)), static_cast<size_t>(CTAsUpperLR));
    }
    else {
      auto blocksNeeded = cuda::std::min(bytes / PurlinAtomTR::RED_PIPELINE_BYTES,
        bytes / (world * PurlinAtomTR::STAGE_BYTES));
      blocksNeeded = static_cast<int>(min(blocksNeeded,static_cast<size_t>(maxReduceBlocks)));
      blocks = transferBlocks + blocksNeeded;
      if (blocksNeeded < 1) {
        // non-pipelined path
        blocks = transferBlocks + cuda::std::min(cuda::ceil_div(bytes / world,
          PurlinAtomTR::THREADS*PurlinAtomTR::BaseConfig::ALIGNMENT_BYTES), static_cast<size_t>(maxReduceBlocks));
      }
    }
    if (blocks < 1) {
      throw std::runtime_error("Blocks must be >= 1");
    }
    if (blocks > maxCTAs) {
      throw std::runtime_error("Blocks must be <= " + std::to_string(maxCTAs));
    }
    const Args kArgs{
      .src = sourceBytes,
      .dst = destinationBytes,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{static_cast<long int>(blocks)}
    };
    const auto operation = [&] {
      if (isLR) {
        allReduce<PurlinAtomLR, DataType, purlin::CollectiveConfigLR, purlin::AllReducePath::direct>
          <<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>(kArgs, ctx);
      } else if (world == 2) {
        if (bytes <= CHUNK_SIZE) {
          allReduce<PurlinAtomTR, DataType, nonChunkedConfig, purlin::AllReducePath::direct>
            <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        } else {
          allReduce<PurlinAtomTR, DataType, chunkedConfig, purlin::AllReducePath::direct>
            <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
      } else if (localBytes <= CHUNK_SIZE) {
        allReduce<PurlinAtomTR, DataType, nonChunkedConfig, purlin::AllReducePath::composed>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
      } else {
        allReduce<PurlinAtomTR, DataType, chunkedConfig, purlin::AllReducePath::composed>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
      }
    };
    operation();
    CHECK_CUDA(cudaGetLastError());
    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxMismatches(destination.get(), reference.get(), elements, stream), elements);
    const double milliseconds = bench::measureOperation(stream, MPI_COMM_WORLD, options, operation);
    if (rank == 0) {
      printf("%d, %lu, %s, %lf, %lf, %lf, %d, %s, %d, %s, %s, %s, %d, %d, %s, %s, %s, %d, %s, %d, %d, %d\n",
        world, bytes, bench::dataTypeName<DataType>(), milliseconds, bench::bandwidth(bytes, milliseconds), errorPercentage,
        nArch, prop.name,
        isLR ? PurlinAtomLR::THREADS : PurlinAtomTR::THREADS,
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
        bench::effectiveWarmup(options.graphLaunches, options.runs, options.warmup),
        options.runs, options.graphLaunches);
    }
  });
}

// ./ar [minBytes] [maxBytes] [maxBlocks] [graphLaunches] [runs] [warmup] [seed]
int main(int argc, char** argv) {
  try {
    const auto options = bench::parseKernelOptions(argc, argv);
    bench::validatePurlinOptions(options);
    arHost(options);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
