#ifndef PURLIN_SUPPORT_BENCHMARK_VARIABLE_COUNTS_CUH
#define PURLIN_SUPPORT_BENCHMARK_VARIABLE_COUNTS_CUH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
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

// Every split is a whole number of units. The unit is the largest power of two
// that divides the total and still leaves eight units per rank, never below the
// 128-byte access alignment. Power-of-two units keep every partition size and
// offset aligned, and from 64 KiB up each partition is a whole number of copy
// pipelines even after it is divided among sixteen blocks.
inline size_t splitUnit(const size_t total, const int world) {
  constexpr size_t alignment = 128;
  constexpr size_t unitsPerRank = 8;
  if (world <= 0 || world > 32) throw std::invalid_argument("Split world size must be in [1, 32]");
  if (total == 0 || total % alignment != 0) {
    throw std::invalid_argument("Split total bytes must be a positive multiple of 128");
  }
  size_t unit = alignment;
  while (unit <= total / 2 && total % (unit * 2) == 0 &&
         total / (unit * 2) >= unitsPerRank * static_cast<size_t>(world)) {
    unit *= 2;
  }
  return unit;
}

// The Zipf exponent, in eighths. It is 0.125 unless BENCH_ZIPF_EXPONENT selects
// 0 (uniform), 0.25, 0.5 or 1. The fractional weights need only square roots and
// divisions, which IEEE-754 rounds identically everywhere, so every
// implementation of this policy derives the same splits.
inline int zipfExponentEighths() {
  const char* text = std::getenv("BENCH_ZIPF_EXPONENT");
  const std::string value(text == nullptr ? "" : text);
  if (value.empty() || value == "0.125") return 1;
  if (value == "0.25") return 2;
  if (value == "0.5") return 4;
  if (value == "1") return 8;
  if (value == "0") return 0;
  throw std::invalid_argument("BENCH_ZIPF_EXPONENT must be 0, 0.125, 0.25, 0.5 or 1");
}

// Apportion total bytes among the ranks by Zipf weights: rank r receives units
// in proportion to 1 / (r + 1)^s. Floor each quota, then give the remaining
// units to the largest fractional remainders (ties go to the lowest rank). With
// s = 1, integer weights lcm(1..world) / (r + 1) keep the result exact. Totals
// below eight units per rank coarsen the shape and, for the steeper exponents,
// may yield zero splits.
inline std::vector<size_t> zipfSplits(const size_t total, const int world) {
  const size_t unit = splitUnit(total, world);
  const size_t units = total / unit;
  const int eighths = zipfExponentEighths();
  std::vector<size_t> sizes(world);
  std::vector<int> order(world);
  size_t assigned = 0;
  if (eighths == 8) {
    size_t scale = 1;
    for (int rank = 1; rank <= world; ++rank) scale = std::lcm(scale, static_cast<size_t>(rank));
    std::vector<size_t> weights(world);
    size_t weightSum = 0;
    for (int rank = 0; rank < world; ++rank) {
      weights[rank] = scale / static_cast<size_t>(rank + 1);
      weightSum += weights[rank];
    }
    if (units > std::numeric_limits<size_t>::max() / weights[0]) {
      throw std::invalid_argument("Split total bytes are too large for this world size");
    }
    std::vector<size_t> remainders(world);
    for (int rank = 0; rank < world; ++rank) {
      const size_t numerator = units * weights[rank];
      sizes[rank] = (numerator / weightSum) * unit;
      remainders[rank] = numerator % weightSum;
      assigned += numerator / weightSum;
      order[rank] = rank;
    }
    std::stable_sort(order.begin(), order.end(), [&](const int left, const int right) {
      return remainders[left] > remainders[right];
    });
  } else {
    // Doubles represent unit counts exactly only up to 2^53; benchmark totals
    // use a few hundred units.
    if (units > (size_t{1} << 32)) {
      throw std::invalid_argument("Split total bytes are too large for a fractional Zipf exponent");
    }
    std::vector<double> weights(world);
    double weightSum = 0.0;
    for (int rank = 0; rank < world; ++rank) {
      // Each square root halves the exponent: 1/2, 1/4, 1/8.
      double root = static_cast<double>(rank + 1);
      for (int exponent = eighths; exponent > 0 && exponent < 8; exponent *= 2) root = std::sqrt(root);
      weights[rank] = eighths == 0 ? 1.0 : 1.0 / root;
      weightSum += weights[rank];
    }
    std::vector<double> remainders(world);
    for (int rank = 0; rank < world; ++rank) {
      const double ideal = static_cast<double>(units) * weights[rank] / weightSum;
      const double whole = std::floor(ideal);
      sizes[rank] = static_cast<size_t>(whole) * unit;
      remainders[rank] = ideal - whole;
      assigned += static_cast<size_t>(whole);
      order[rank] = rank;
    }
    std::stable_sort(order.begin(), order.end(), [&](const int left, const int right) {
      return remainders[left] > remainders[right];
    });
  }
  // Floors leave fewer than one unit per rank to hand out.
  if (assigned > units || units - assigned > static_cast<size_t>(world)) {
    throw std::invalid_argument("Split apportionment lost precision");
  }
  for (size_t index = 0; index < units - assigned; ++index) {
    sizes[order[index]] += unit;
  }
  return sizes;
}

// For AGV and RSV, total is the sum of the shared partition sizes, not a
// per-rank base size. Rank zero always owns the largest partition.
inline std::vector<size_t> allGatherSizes(const size_t total, const int world) {
  return zipfSplits(total, world);
}

inline std::vector<size_t> reduceScatterSizes(const size_t total, const int world) {
  return zipfSplits(total, world);
}

inline std::vector<size_t> allToAllSplitsForSource(const size_t total,
  const int source, const int world) {
  if (source < 0 || source >= world) {
    throw std::invalid_argument("All-to-all source rank is out of range");
  }
  auto splits = zipfSplits(total, world);
  const int shift = (source + 1) % world;
  // Rotate the entire rounded vector, including remainder tie-breaking. This
  // puts the largest split on the next rank and preserves every column sum.
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
