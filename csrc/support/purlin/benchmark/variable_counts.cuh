#ifndef PURLIN_SUPPORT_BENCHMARK_VARIABLE_COUNTS_CUH
#define PURLIN_SUPPORT_BENCHMARK_VARIABLE_COUNTS_CUH

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "checks.cuh"

namespace bench {

inline std::vector<size_t> offsets(const std::vector<size_t>& sizes) {
  std::vector<size_t> result(sizes.size());
  size_t offset = 0;
  for (size_t index = 0; index < sizes.size(); ++index) {
    result[index] = offset;
    offset += sizes[index];
  }
  return result;
}

inline size_t totalBytes(const std::vector<size_t>& sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), size_t{0});
}

inline size_t maximumBytes(const std::vector<size_t>& sizes) {
  if (sizes.empty()) throw std::invalid_argument("Cannot take the maximum of an empty size vector");
  return std::ranges::max(sizes);
}

// Apportion total bytes with weights [4, 1, ..., 1] in 128-byte units.
// Floor each quota, then give the remaining units to the largest fractional
// remainders (ties go to the lowest index). Small totals may yield zero splits.
inline std::vector<size_t> weightedSplits(const size_t total, const int world) {
  constexpr size_t alignment = 128;
  if (world <= 0) throw std::invalid_argument("Split world size must be positive");
  if (total == 0 || total % alignment != 0) {
    throw std::invalid_argument("Split total bytes must be a positive multiple of 128");
  }
  const size_t units = total / alignment;
  const size_t weightSum = static_cast<size_t>(world) + 3;
  std::vector<size_t> sizes(world);
  std::vector<size_t> remainders(world);
  std::vector<int> order(world);
  size_t assigned = 0;
  for (int rank = 0; rank < world; ++rank) {
    // units <= SIZE_MAX / 128, so multiplying by four cannot overflow.
    const size_t numerator = units * (rank == 0 ? 4 : 1);
    sizes[rank] = (numerator / weightSum) * alignment;
    remainders[rank] = numerator % weightSum;
    assigned += numerator / weightSum;
    order[rank] = rank;
  }
  std::stable_sort(order.begin(), order.end(), [&](const int left, const int right) {
    return remainders[left] > remainders[right];
  });
  for (size_t index = 0; index < units - assigned; ++index) {
    sizes[order[index]] += alignment;
  }
  return sizes;
}

// For AGV and RSV, total is the sum of the shared partition sizes, not a
// per-rank base size. Rank zero always owns the larger partition.
inline std::vector<size_t> allGatherSizes(const size_t total, const int world) {
  return weightedSplits(total, world);
}

inline std::vector<size_t> reduceScatterSizes(const size_t total, const int world) {
  return weightedSplits(total, world);
}

inline std::vector<size_t> allToAllSplitsForSource(const size_t total,
  const int source, const int world) {
  if (source < 0 || source >= world) {
    throw std::invalid_argument("All-to-all source rank is out of range");
  }
  auto splits = weightedSplits(total, world);
  const int shift = (source + 1) % world;
  // Rotate the entire rounded vector, including remainder tie-breaking. This
  // puts the larger split on the next rank and preserves every column sum.
  std::rotate(splits.begin(), splits.end() - shift, splits.end());
  return splits;
}

inline std::vector<size_t> allToAllSendSplits(const size_t total,
  const int rank, const int world) {
  return allToAllSplitsForSource(total, rank, world);
}

inline std::vector<size_t> allToAllReceiveSplits(const size_t total,
  const int rank, const int world) {
  if (rank < 0 || rank >= world) {
    throw std::invalid_argument("All-to-all receive rank is out of range");
  }
  std::vector<size_t> splits(world);
  for (int peer = 0; peer < world; ++peer) {
    splits[peer] = allToAllSplitsForSource(total, peer, world)[rank];
  }
  return splits;
}

inline void validatePeerCounts(const std::vector<size_t>& sends,
  const std::vector<size_t>& receives, MPI_Comm communicator) {
  const int world = static_cast<int>(sends.size());
  std::vector<unsigned long long> sendValues(world);
  std::vector<unsigned long long> receivedValues(world);
  for (int peer = 0; peer < world; ++peer) sendValues[peer] = sends[peer];
  MPI_CHECK(MPI_Alltoall(sendValues.data(), 1, MPI_UNSIGNED_LONG_LONG,
    receivedValues.data(), 1, MPI_UNSIGNED_LONG_LONG, communicator));
  for (int peer = 0; peer < world; ++peer) {
    if (receivedValues[peer] != receives[peer]) {
      throw std::runtime_error("All-to-all send and receive counts are inconsistent");
    }
  }
}

} // namespace bench

#endif
