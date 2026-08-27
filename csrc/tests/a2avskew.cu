// all2allV under seeded random skew: every rank derives the same split matrix
// from a broadcast seed, validates against NCCL grouped send/recv, and times
// both with the graph-capture methodology. Skew s draws each weight from
// 1 + s*uniform(-1,1) via an integer hash of (seed, src, dst), row-normalized
// and 16B-aligned, so all ranks compute identical splits with no exchange.
//
//   A2AVSKEW_SKEW=0,25,50   percent skew levels to sweep (default "0,25,50")
//   A2AVSKEW_SEED=12345     split-matrix seed (default 12345)
//   A2AVSKEW_HOTRING=90     hot-ring mode instead: rank r sends frac% of its
//                           row to rank r+1, remainder split evenly
//
// Sizes are the per-rank row total. Example:
//   NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
//     mpirun -n 8 ./cmake-build-release/testA2AVSKEW 512K 32M 8 32 32
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

#include <purlin/host/all2all.cuh>

namespace {

constexpr size_t SPLIT_ALIGNMENT = 16;

uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// One row of the split matrix. Pure integer arithmetic so every rank computes
// bit-identical splits. weight = SCALE + skewPercent * centered(hash) / 100,
// centered in [-SCALE, SCALE]; splits = total * weight / sum, aligned down to
// 16B with the residue distributed by largest remainder (ties by index).
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
    // total <= 2^31, weight <= 2^21: the product fits u64 comfortably
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
  splits[order[0]] += leftover; // total is 16B-aligned, so this is zero
  return splits;
}

// Degenerate skew: rank r sends frac% of its row to rank r+1, the remainder
// split evenly across every other destination (self included).
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
  std::vector<std::vector<size_t>> rows; // rows[src][dst]
  std::vector<size_t> sends;             // this rank's row
  std::vector<size_t> receives;          // this rank's column
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

    // Exact buffer bound: walk every sweep point's matrix up front.
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
