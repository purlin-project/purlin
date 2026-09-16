#include <iostream>
#include <limits>

#include <purlin/benchmark/variable_counts.cuh>

namespace {
void require(const bool condition) {
  if (!condition) throw std::runtime_error("Variable-count split invariant failed");
}

template<typename Function>
void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("Invalid split argument was accepted");
}
}

int main() {
  try {
    require(bench::allGatherSizes(8192, 4) == std::vector<size_t>{4736, 1152, 1152, 1152});
    require(bench::reduceScatterSizes(8192, 4) == std::vector<size_t>{4736, 1152, 1152, 1152});
    require(bench::allToAllSendSplits(8192, 0, 4) == std::vector<size_t>{1152, 4736, 1152, 1152});
    require(bench::allToAllSendSplits(8192, 3, 4) == std::vector<size_t>{4736, 1152, 1152, 1152});
    require(bench::weightedSplits(2048, 3) == std::vector<size_t>{1408, 384, 256});
    require(bench::weightedSplits(128, 4) == std::vector<size_t>{128, 0, 0, 0});

    for (const int world : {1, 2, 3, 4, 8, 16, 32}) {
      std::vector<size_t> previous(world, 0);
      for (int exponent = 7; exponent <= 30; ++exponent) {
        const size_t total = size_t{1} << exponent;
        const auto base = bench::allGatherSizes(total, world);
        require(base == bench::reduceScatterSizes(total, world));
        require(bench::totalBytes(base) == total);
        const auto offsets = bench::offsets(base);
        require(offsets.back() + base.back() == total);
        for (int rank = 0; rank < world; ++rank) {
          require(base[rank] % 128 == 0);
          require(base[rank] >= previous[rank]);
          const size_t numerator = (total / 128) * (rank == 0 ? 4 : 1);
          const size_t floor = numerator / (world + 3);
          require(base[rank] / 128 == floor || base[rank] / 128 == floor + 1);
          const auto sends = bench::allToAllSendSplits(total, rank, world);
          const auto receives = bench::allToAllReceiveSplits(total, rank, world);
          require(bench::totalBytes(sends) == total);
          require(bench::totalBytes(receives) == total);
          require(sends[(rank + 1) % world] == base[0]);
          for (int peer = 0; peer < world; ++peer) {
            require(receives[peer] == bench::allToAllSendSplits(total, peer, world)[rank]);
          }
        }
        previous = base;
      }
    }
    const size_t largeTotal = std::numeric_limits<size_t>::max() / 128 * 128;
    require(bench::totalBytes(bench::weightedSplits(largeTotal, 4)) == largeTotal);
    for (const size_t total : {0, 127, 129}) {
      rejects([&] { bench::weightedSplits(total, 4); });
    }
    for (const int world : {0, -1}) {
      rejects([&] { bench::weightedSplits(1024, world); });
    }
    for (const int rank : {-1, 4}) {
      rejects([&] { bench::allToAllSendSplits(1024, rank, 4); });
      rejects([&] { bench::allToAllReceiveSplits(1024, rank, 4); });
    }
    std::cout << "Variable-count split tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
