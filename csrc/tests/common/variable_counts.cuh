#ifndef PURLIN_TESTS_COMMON_VARIABLE_COUNTS_CUH
#define PURLIN_TESTS_COMMON_VARIABLE_COUNTS_CUH

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

inline std::vector<size_t> allGatherSizes(const size_t bytes, const int world) {
  std::vector<size_t> sizes(world, bytes);
  if (bytes >= static_cast<size_t>(128 * world)) {
    for (int rank = 1; rank < world; ++rank) sizes[rank] -= 128;
  }
  return sizes;
}

inline std::vector<size_t> reduceScatterSizes(const size_t bytes, const int world) {
  std::vector<size_t> sizes(world, bytes);
  if (bytes >= static_cast<size_t>(128 * world)) {
    for (int rank = 1; rank < world; ++rank) {
      sizes[0] -= 128;
      sizes[rank] += 128;
    }
  }
  return sizes;
}

inline std::vector<size_t> allToAllSplitsForSource(const size_t total,
  const int source, const int world) {
  const size_t peerBase = (total / static_cast<size_t>(world)) / 32 * 32;
  std::vector<size_t> splits(world, peerBase);
  splits[world - 1] += total - peerBase * static_cast<size_t>(world);
  if (peerBase >= static_cast<size_t>(64 * world)) {
    for (int destination = 0; destination < world - 1; ++destination) {
      const long long delta = ((source + destination) % 2) == 0 ? -128 : 128;
      splits[destination] = static_cast<size_t>(
        static_cast<long long>(splits[destination]) + delta);
      splits[world - 1] = static_cast<size_t>(
        static_cast<long long>(splits[world - 1]) - delta);
    }
  }
  return splits;
}

inline std::vector<size_t> allToAllSendSplits(const size_t total,
  const int rank, const int world) {
  return allToAllSplitsForSource(total, rank, world);
}

inline std::vector<size_t> allToAllReceiveSplits(const size_t total,
  const int rank, const int world) {
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
