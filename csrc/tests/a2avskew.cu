// Benchmarks all2allV with deterministic, uneven peer splits and compares it
// with NCCL grouped send/receive. Every rank derives the same 16-byte-aligned
// split matrix from the shared seed, so no metadata exchange is needed.
//
//   A2AVSKEW_SKEW=0,25,50  skew percentages (default: 0,25,50)
//   A2AVSKEW_SEED=12345    split-matrix seed (default: 12345)
//   A2AVSKEW_HOTRING=90    send 90% to the next rank and split the rest evenly
//
// Command-line sizes are totals per source rank. Example environment:
//   NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none
// Run: mpirun -n 8 ./cmake-build-release/testA2AVSKEW 512K 32M 8 32 32
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/nccl_collectives.cuh>
#include <purlin/benchmark/nccl_communicator.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>
#include <purlin/benchmark/variable_counts.cuh>

#include <purlin/host/all2all.cuh>

namespace {

constexpr size_t SPLIT_ALIGNMENT = 16;

uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Build one matrix row with integer-only weights, then distribute alignment
// residue by largest remainder. This produces identical splits on every rank.
std::vector<size_t> skewRow(const uint64_t seed, const int src, const int world,
  const size_t total, const int skewPercent) {
  constexpr int64_t SCALE = 1 << 20;
  std::vector<int64_t> weights(world);
  int64_t weightSum = 0;
  for (int dst = 0; dst < world; ++dst) {
    const auto h = splitmix64(splitmix64(seed ^ (static_cast<uint64_t>(src) << 32)) ^
      static_cast<uint64_t>(dst));
    const auto centered = static_cast<int64_t>(h % (2 * SCALE + 1)) - SCALE;
    weights[dst] = SCALE + (skewPercent * centered) / 100;
    weightSum += weights[dst];
  }
  std::vector<size_t> splits(world);
  std::vector<uint64_t> remainders(world);
  size_t assigned = 0;
  for (int dst = 0; dst < world; ++dst) {
    // The configured bounds keep this product well within uint64_t.
    const uint64_t raw = (static_cast<uint64_t>(total) *
      static_cast<uint64_t>(weights[dst])) / static_cast<uint64_t>(weightSum);
    splits[dst] = raw / SPLIT_ALIGNMENT * SPLIT_ALIGNMENT;
    remainders[dst] = raw - splits[dst];
    assigned += splits[dst];
  }
  std::vector<int> order(world);
  for (int dst = 0; dst < world; ++dst) order[dst] = dst;
  std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) {
    return remainders[a] > remainders[b];
  });
  size_t leftover = total - assigned;
  for (int pick = 0; leftover >= SPLIT_ALIGNMENT; pick = (pick + 1) % world) {
    splits[order[pick]] += SPLIT_ALIGNMENT;
    leftover -= SPLIT_ALIGNMENT;
  }
  splits[order[0]] += leftover; // The row total is already 16-byte aligned.
  return splits;
}

// Send the requested fraction to the next rank and divide the remainder among
// all other destinations, including the source rank.
std::vector<size_t> hotRingRow(const int src, const int world, const size_t total,
  const int fracPercent) {
  const int ring = (src + 1) % world;
  const size_t hot = (total * static_cast<size_t>(fracPercent)) / 100 /
    SPLIT_ALIGNMENT * SPLIT_ALIGNMENT;
  const size_t coldBase = (total - hot) / static_cast<size_t>(world - 1) /
    SPLIT_ALIGNMENT * SPLIT_ALIGNMENT;
  std::vector<size_t> splits(world, coldBase);
  splits[ring] = hot;
  size_t assigned = hot + coldBase * static_cast<size_t>(world - 1);
  splits[(ring + 1) % world] += total - assigned;
  return splits;
}

struct SplitMatrix {
  std::vector<std::vector<size_t>> rows; // All source/destination rows.
  std::vector<size_t> sends;             // This rank's row.
  std::vector<size_t> receives;          // This rank's column.
};

SplitMatrix makeMatrix(const uint64_t seed, const int rank, const int world,
  const size_t total, const int skewPercent, const int hotRingPercent) {
  SplitMatrix m;
  m.rows.resize(world);
  for (int src = 0; src < world; ++src) {
    m.rows[src] = hotRingPercent > 0
      ? hotRingRow(src, world, total, hotRingPercent)
      : skewRow(seed, src, world, total, skewPercent);
  }
  m.sends = m.rows[rank];
  m.receives.resize(world);
  for (int src = 0; src < world; ++src) m.receives[src] = m.rows[src][rank];
  return m;
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
  if (skews.empty()) throw std::invalid_argument("A2AVSKEW_SKEW is empty");
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

    const char* skewEnv = std::getenv("A2AVSKEW_SKEW");
    auto skews = parseSkewList(skewEnv != nullptr ? skewEnv : "0,25,50");
    const char* hotEnv = std::getenv("A2AVSKEW_HOTRING");
    const int hotRingPercent = hotEnv != nullptr ? std::stoi(hotEnv) : 0;
    uint64_t seed = 12345;
    if (const char* seedEnv = std::getenv("A2AVSKEW_SEED")) seed = std::stoull(seedEnv);
    MPI_CHECK(MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD));
    if (hotRingPercent > 0) skews = {hotRingPercent};

    // Precompute the largest buffer required by any sweep point.
    size_t maximumBufferBytes = 0;
    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      for (const int skew : skews) {
        const auto m = makeMatrix(seed, runtime.rank, runtime.world, bytes,
          hotRingPercent > 0 ? 0 : skew, hotRingPercent);
        maximumBufferBytes = std::max({maximumBufferBytes,
          bench::totalBytes(m.sends), bench::totalBytes(m.receives)});
      }
    });

    bench::DeviceBuffer<cuda::std::byte> source(maximumBufferBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(maximumBufferBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> ncclDestination(maximumBufferBytes, runtime.stream);
    bench::DeviceBuffer<size_t> deviceSends(runtime.world, runtime.stream);
    bench::DeviceBuffer<size_t> deviceReceives(runtime.world, runtime.stream);

    if (runtime.rank == 0) {
      std::printf("collective,mode,seed,skew(%%),rowBytes,purlin(us),nccl(us),ratio,error(%%)\n");
    }

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      for (const int skew : skews) {
        const auto m = makeMatrix(seed, runtime.rank, runtime.world, bytes,
          hotRingPercent > 0 ? 0 : skew, hotRingPercent);
        const auto sendOffsets = bench::offsets(m.sends);
        const auto receiveOffsets = bench::offsets(m.receives);
        const size_t sendTotal = bench::totalBytes(m.sends);
        const size_t receiveTotal = bench::totalBytes(m.receives);

        bench::validatePeerCounts(m.sends, m.receives, MPI_COMM_WORLD);
        runtime.context.vState = bench::makePurlinAllToAllVState(
          m.sends, m.receives, sendOffsets, runtime.rank);
        CHECK_CUDA(cudaMemcpyAsync(deviceSends.get(), m.sends.data(),
          sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));
        CHECK_CUDA(cudaMemcpyAsync(deviceReceives.get(), m.receives.data(),
          sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));

        bench::fillBytePattern(source.get(), sendTotal, runtime.rank, runtime.stream);
        const auto purlinOperation = [&] {
          purlin::all2allV<ARCH>(source.get(), destination.get(), deviceSends.get(),
            deviceReceives.get(), runtime.context, runtime.stream);
        };
        const auto ncclOperation = [&] {
          bench::ncclAllToAllV(source.get(), ncclDestination.get(), m.sends, m.receives,
            sendOffsets, receiveOffsets, runtime.rank, runtime.world,
            nccl.get(), runtime.stream);
        };
        purlinOperation();
        ncclOperation();

        const double errorPercentage = bench::maxErrorPercentage(
          bench::matxByteMismatches(destination.get(), ncclDestination.get(),
            receiveTotal, runtime.stream), receiveTotal);

        const double purlinMilliseconds = bench::measureOperation(
          runtime.stream, MPI_COMM_WORLD, options, purlinOperation);
        const double ncclMilliseconds = bench::measureOperation(
          runtime.stream, MPI_COMM_WORLD, options, ncclOperation);

        if (runtime.rank == 0) {
          std::printf("all_to_all_v_skew,%s,%llu,%d,%zu,%.4f,%.4f,%.4f,%.4f\n",
            hotRingPercent > 0 ? "hotring" : "random",
            static_cast<unsigned long long>(seed), skew, bytes,
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
