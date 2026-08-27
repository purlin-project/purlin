// allGatherV under seeded random skew: every rank derives the same
// contribution-size vector from a broadcast seed (weight 1 + s*uniform(-1,1)
// per rank, normalized to world x nominal, 16B-aligned), validates against
// NCCL's grouped-broadcast emulation, and times both with graph capture.
//
//   AGVSKEW_SKEW=0,25,50   percent skew levels (default "0,25,50")
//   AGVSKEW_SEED=12345     size-vector seed (default 12345)
//
// Sizes are the nominal per-rank contribution. Example:
//   NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
//     mpirun -n 8 ./cmake-build-release/testAGVSKEW 512K 32M 8 32 32
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/benchmark.cuh"
#include "common/data.cuh"
#include "common/device_buffer.cuh"
#include "common/matx_validation.cuh"
#include "common/nccl_collectives.cuh"
#include "common/nccl_communicator.cuh"
#include "common/purlin_report.cuh"
#include "common/purlin_runtime.cuh"
#include "common/variable_counts.cuh"

#include <purlin/host/allGather.cuh>

namespace {

constexpr size_t SPLIT_ALIGNMENT = 16;

uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Deterministic on every rank: pure integer arithmetic from the shared seed.
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

// Sparse mode (MoE-style raggedness): the k lowest-hash entries get 64KB
// each, the rest share the remainder evenly. Reported with skew = -k.
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
  if (skews.empty()) throw std::invalid_argument("AGVSKEW_SKEW is empty");
  return skews;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    bench::PurlinRuntime runtime;
    bench::NcclCommunicator nccl;
    nccl.initialize(runtime.rank, runtime.world);

    const char* skewEnv = std::getenv("AGVSKEW_SKEW");
    auto skews = parseSkewList(skewEnv != nullptr ? skewEnv : "0,25,50");
    uint64_t seed = 12345;
    if (const char* seedEnv = std::getenv("AGVSKEW_SEED")) seed = std::stoull(seedEnv);
    MPI_CHECK(MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD));
    const char* sparseEnv = std::getenv("AGVSKEW_SPARSE");
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

    bench::DeviceBuffer<cuda::std::byte> source(maximumOwnBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(maximumTotalBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> ncclDestination(maximumTotalBytes, runtime.stream);
    bench::DeviceBuffer<size_t> deviceSizes(runtime.world, runtime.stream);

    if (runtime.rank == 0) {
      std::printf("collective,seed,skew(%%),nominalBytes,purlin(us),nccl(us),ratio,error(%%)\n");
    }

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      for (const int skew : skews) {
        const auto sizes = sparseCount > 0
          ? sparseSizes(seed, runtime.world, bytes * static_cast<size_t>(runtime.world), sparseCount)
          : skewSizes(seed, runtime.world, bytes * static_cast<size_t>(runtime.world), skew);
        const auto offsets = bench::offsets(sizes);
        const size_t total = bench::totalBytes(sizes);
        runtime.context.vState = bench::makePurlinVState(sizes, offsets, runtime.rank);
        CHECK_CUDA(cudaMemcpyAsync(deviceSizes.get(), sizes.data(),
          sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));

        bench::fillBytePattern(source.get(), sizes[runtime.rank], runtime.rank, runtime.stream);
        const auto purlinOperation = [&] {
          purlin::allGatherV<ARCH>(source.get(), destination.get(), deviceSizes.get(),
            runtime.context, runtime.stream);
        };
        const auto ncclOperation = [&] {
          bench::ncclAllGatherV(source.get(), ncclDestination.get(), sizes, offsets,
            runtime.rank, runtime.world, nccl.get(), runtime.stream);
        };
        purlinOperation();
        ncclOperation();

        const double errorPercentage = bench::maxErrorPercentage(
          bench::matxByteMismatches(destination.get(), ncclDestination.get(), total,
            runtime.stream), total);

        const double purlinMilliseconds = bench::measureOperation(
          runtime.stream, MPI_COMM_WORLD, options, purlinOperation);
        const double ncclMilliseconds = bench::measureOperation(
          runtime.stream, MPI_COMM_WORLD, options, ncclOperation);

        if (runtime.rank == 0) {
          std::printf("all_gather_v_skew,%llu,%d,%zu,%.4f,%.4f,%.4f,%.4f\n",
            static_cast<unsigned long long>(seed), sparseCount > 0 ? -sparseCount : skew, bytes,
            purlinMilliseconds * 1e3, ncclMilliseconds * 1e3,
            ncclMilliseconds / purlinMilliseconds, errorPercentage);
          std::fflush(stdout);
        }
      }
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
