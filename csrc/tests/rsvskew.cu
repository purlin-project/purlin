// Benchmarks reduceScatterV with deterministic, uneven shards. Every rank
// derives the same 16-byte-aligned sizes from the shared seed. Correctness uses
// Purlin's rank-ordered reference, accumulated locally from seeded replays of
// every peer's contribution.
//
//   RSVSKEW_SKEW=0,25,50  skew percentages (default: 0,25,50)
//   RSVSKEW_SEED=12345    shard-size seed (default: 12345)
//   RSVSKEW_SPARSE=k      give k shards 64 KiB and divide the rest evenly
//
// Command-line sizes are nominal bytes per shard. Example environment:
//   NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none
// Run: mpirun -n 8 ./cmake-build-release/testRSVSKEW 512K 32M 8 32 32
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>
#include <purlin/benchmark/variable_counts.cuh>

#include <purlin/host/reduceScatter.cuh>

using DataType = __nv_bfloat16;

namespace {

constexpr size_t SPLIT_ALIGNMENT = 16;

using bench::splitmix64;

std::vector<size_t> skewSizes(const uint64_t seed, const int world,
  const size_t total, const int skewPercent) {
  constexpr int64_t SCALE = 1 << 20;
  std::vector<int64_t> weights(world);
  int64_t weightSum = 0;
  for (int r = 0; r < world; ++r) {
    const auto h = splitmix64(seed ^ (static_cast<uint64_t>(r) + 1));
    const auto centered = static_cast<int64_t>(h % (2 * SCALE + 1)) - SCALE;
    weights[r] = SCALE + (skewPercent * centered) / 100;
    weightSum += weights[r];
  }
  std::vector<size_t> sizes(world);
  std::vector<uint64_t> remainders(world);
  size_t assigned = 0;
  for (int r = 0; r < world; ++r) {
    const uint64_t raw = (static_cast<uint64_t>(total) *
      static_cast<uint64_t>(weights[r])) / static_cast<uint64_t>(weightSum);
    sizes[r] = raw / SPLIT_ALIGNMENT * SPLIT_ALIGNMENT;
    remainders[r] = raw - sizes[r];
    assigned += sizes[r];
  }
  std::vector<int> order(world);
  for (int r = 0; r < world; ++r) order[r] = r;
  std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) {
    return remainders[a] > remainders[b];
  });
  size_t leftover = total - assigned;
  for (int pick = 0; leftover >= SPLIT_ALIGNMENT; pick = (pick + 1) % world) {
    sizes[order[pick]] += SPLIT_ALIGNMENT;
    leftover -= SPLIT_ALIGNMENT;
  }
  sizes[order[0]] += leftover;
  return sizes;
}

// Sparse mode gives 64 KiB to the k lowest hashes and divides the remainder
// evenly. Reports encode this mode as a skew value of -k.
std::vector<size_t> sparseSizes(const uint64_t seed, const int world,
  const size_t total, const int sparseCount) {
  constexpr size_t TINY = 64UL * 1024UL;
  std::vector<int> order(world);
  for (int r = 0; r < world; ++r) order[r] = r;
  std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) {
    return splitmix64(seed ^ (static_cast<uint64_t>(a) + 101)) <
           splitmix64(seed ^ (static_cast<uint64_t>(b) + 101));
  });
  std::vector<size_t> sizes(world, 0);
  const int big = world - sparseCount;
  if (big <= 0) throw std::invalid_argument("sparse count must leave one big entry");
  const size_t bulk = total - TINY * static_cast<size_t>(sparseCount);
  const size_t share = bulk / static_cast<size_t>(big) / SPLIT_ALIGNMENT * SPLIT_ALIGNMENT;
  for (int i = 0; i < world; ++i) sizes[order[i]] = i < sparseCount ? TINY : share;
  sizes[order[world - 1]] += total - TINY * static_cast<size_t>(sparseCount) -
    share * static_cast<size_t>(big);
  return sizes;
}

std::vector<int> parseSkewList(const char* text) {
  std::vector<int> skews;
  std::string s(text);
  size_t pos = 0;
  while (pos < s.size()) {
    const auto comma = s.find(',', pos);
    const auto token = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    skews.push_back(std::stoi(token));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (skews.empty()) throw std::invalid_argument("RSVSKEW_SKEW is empty");
  return skews;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    const bool predictable = ARCH >= 900 && options.reductionMode == purlin::ReductionMode::nonDeterministic;
    bench::PurlinRuntime runtime;
    const uint32_t dataSeed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, dataSeed);

    const char* skewEnv = std::getenv("RSVSKEW_SKEW");
    auto skews = parseSkewList(skewEnv != nullptr ? skewEnv : "0,25,50");
    uint64_t seed = 12345;
    if (const char* seedEnv = std::getenv("RSVSKEW_SEED")) seed = std::stoull(seedEnv);
    MPI_CHECK(MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD));
    const char* sparseEnv = std::getenv("RSVSKEW_SPARSE");
    const int sparseCount = sparseEnv != nullptr ? std::stoi(sparseEnv) : 0;
    if (sparseCount > 0) skews = {0};

    size_t maximumOwnBytes = 0;
    size_t maximumTotalBytes = 0;
    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      for (const int skew : skews) {
        const auto sizes = sparseCount > 0
          ? sparseSizes(seed, runtime.world, bytes * static_cast<size_t>(runtime.world), sparseCount)
          : skewSizes(seed, runtime.world, bytes * static_cast<size_t>(runtime.world), skew);
        maximumOwnBytes = std::max(maximumOwnBytes, sizes[runtime.rank]);
        maximumTotalBytes = std::max(maximumTotalBytes, bench::totalBytes(sizes));
      }
    });

    bench::DeviceBuffer<DataType> source(maximumTotalBytes / sizeof(DataType), runtime.stream);
    bench::DeviceBuffer<DataType> destination(maximumOwnBytes / sizeof(DataType), runtime.stream);
    bench::DeviceBuffer<DataType> referenceSources(bench::checkedMultiply(
      maximumOwnBytes / sizeof(DataType), runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> reference(maximumOwnBytes / sizeof(DataType), runtime.stream);
    bench::DeviceBuffer<size_t> deviceSizes(runtime.world, runtime.stream);
    auto* sourceBytes = reinterpret_cast<cuda::std::byte*>(source.get());
    auto* destinationBytes = reinterpret_cast<cuda::std::byte*>(destination.get());

    if (runtime.rank == 0) {
      std::printf("collective,seed,skew(%%),nominalBytes,purlin(us),error(%%)\n");
    }

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      for (const int skew : skews) {
        const auto sizes = sparseCount > 0
          ? sparseSizes(seed, runtime.world, bytes * static_cast<size_t>(runtime.world), sparseCount)
          : skewSizes(seed, runtime.world, bytes * static_cast<size_t>(runtime.world), skew);
        const auto offsets = bench::offsets(sizes);
        runtime.context.vState = bench::makePurlinVState(sizes, offsets, runtime.rank);
        CHECK_CUDA(cudaMemcpyAsync(deviceSizes.get(), sizes.data(),
          sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));

        for (int destinationRank = 0; destinationRank < runtime.world; ++destinationRank) {
          bench::fillRandomReduction(
            source.get() + offsets[destinationRank] / sizeof(DataType),
            sizes[destinationRank] / sizeof(DataType),
            bench::reduceScatterSeed(dataSeed, runtime.rank, destinationRank), runtime.stream, predictable);
        }
        const size_t localElements = sizes[runtime.rank] / sizeof(DataType);
        bench::fillRandomReduceScatterReferenceSources(referenceSources.get(),
          localElements, dataSeed, runtime.world, runtime.rank, runtime.stream, predictable);
        bench::computeReductionReference(referenceSources.get(), reference.get(),
          localElements, runtime.world, runtime.stream);

        const auto purlinOperation = [&] {
          purlin::dispatchReductionMode(options.reductionMode, [&]<purlin::ReductionMode mode> {
            purlin::reduceScatterV<ARCH, DataType, purlin::ReduceOp::add, mode>(sourceBytes, destinationBytes,
              deviceSizes.get(), runtime.context, runtime.stream);
          });
        };
        purlinOperation();

        const double errorPercentage = bench::maxErrorPercentage(
          bench::matxMismatches(destination.get(), reference.get(), localElements,
            runtime.stream), localElements);

        const double purlinMilliseconds = bench::measureOperation(
          runtime.stream, MPI_COMM_WORLD, options, purlinOperation);

        if (runtime.rank == 0) {
          std::printf("reduce_scatter_v_skew,%llu,%d,%zu,%.4f,%.4f\n",
            static_cast<unsigned long long>(seed), sparseCount > 0 ? -sparseCount : skew, bytes,
            purlinMilliseconds * 1e3, errorPercentage);
          std::fflush(stdout);
        }
      }
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
