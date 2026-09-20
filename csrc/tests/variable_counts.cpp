#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

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
    // The default exponent is 0.125: four ranks split a total as 9:8:8:7.
    unsetenv("BENCH_ZIPF_EXPONENT");
    require(bench::allGatherSizes(8192, 4) == std::vector<size_t>{2304, 2048, 2048, 1792});
    require(bench::reduceScatterSizes(8192, 4) == std::vector<size_t>{2304, 2048, 2048, 1792});
    require(bench::allToAllSendSplits(8192, 0, 4) == std::vector<size_t>{1792, 2304, 2048, 2048});
    require(bench::allToAllSendSplits(8192, 3, 4) == std::vector<size_t>{2304, 2048, 2048, 1792});
    // Eight ranks: 9:9:8:8:8:8:7:7 sixty-fourths of the total, here in 8 KiB units.
    require(bench::splitUnit(512 * 1024, 8) == 8 * 1024);
    require(bench::zipfSplits(512 * 1024, 8) == std::vector<size_t>{
      73728, 73728, 65536, 65536, 65536, 65536, 57344, 57344});
    require(bench::splitUnit(size_t{4} << 20, 8) == 64 * 1024);
    require(bench::zipfSplits(2048, 3) == std::vector<size_t>{768, 640, 640});
    require(bench::zipfSplits(128, 4) == std::vector<size_t>{128, 0, 0, 0});
    // Exponent 1 uses exact integer weights; equal remainders go to the lowest rank.
    setenv("BENCH_ZIPF_EXPONENT", "1", 1);
    require(bench::zipfSplits(8192, 4) == std::vector<size_t>{3840, 2048, 1280, 1024});
    require(bench::zipfSplits(1024, 3) == std::vector<size_t>{512, 256, 256});
    require(bench::zipfSplits(2048, 3) == std::vector<size_t>{1152, 512, 384});
    unsetenv("BENCH_ZIPF_EXPONENT");

    // The invariants hold for every supported Zipf exponent.
    const std::vector<std::pair<std::string, std::vector<size_t>>> exponents{
      {"", {9, 9, 8, 8, 8, 8, 7, 7}}, {"1", {23, 12, 8, 6, 5, 4, 3, 3}},
      {"0.5", {15, 10, 8, 7, 7, 6, 6, 5}}, {"0.25", {11, 9, 8, 8, 7, 7, 7, 7}},
      {"0.125", {9, 9, 8, 8, 8, 8, 7, 7}}, {"0", {8, 8, 8, 8, 8, 8, 8, 8}}};
    for (const auto& [exponent, sixtyFourths] : exponents) {
    if (exponent.empty()) unsetenv("BENCH_ZIPF_EXPONENT");
    else setenv("BENCH_ZIPF_EXPONENT", exponent.c_str(), 1);
    auto eightRanks = bench::zipfSplits(size_t{4} << 20, 8);
    for (auto& size : eightRanks) size /= 64 * 1024;
    require(eightRanks == sixtyFourths);
    for (const int world : {1, 2, 3, 4, 8, 16, 32}) {
      std::vector<size_t> previous(world, 0);
      for (int exponent = 7; exponent <= 30; ++exponent) {
        const size_t total = size_t{1} << exponent;
        const auto base = bench::allGatherSizes(total, world);
        require(base == bench::reduceScatterSizes(total, world));
        require(bench::totalBytes(base) == total);
        const auto offsets = bench::offsets(base);
        require(offsets.back() + base.back() == total);
        // Units are powers of two of at least 128 bytes, and eight per rank once the total allows.
        const size_t unit = bench::splitUnit(total, world);
        require(unit >= 128 && (unit & (unit - 1)) == 0 && total % unit == 0);
        require(unit == 128 || total / unit >= size_t{8} * world);
        for (int rank = 0; rank < world; ++rank) {
          require(base[rank] % unit == 0 && offsets[rank] % unit == 0);
          require(base[rank] >= previous[rank]);
          // Zipf order: no rank receives more than a lower-numbered rank.
          require(rank == 0 || base[rank] <= base[rank - 1]);
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
    }
    setenv("BENCH_ZIPF_EXPONENT", "2", 1);
    rejects([&] { bench::zipfSplits(1024, 4); });
    unsetenv("BENCH_ZIPF_EXPONENT");
    // Only the exact integer weights of exponent 1 can split a total this large.
    const size_t largeTotal = std::numeric_limits<size_t>::max() / 128 * 128;
    rejects([&] { bench::zipfSplits(largeTotal, 4); });
    setenv("BENCH_ZIPF_EXPONENT", "1", 1);
    require(bench::totalBytes(bench::zipfSplits(largeTotal, 4)) == largeTotal);
    unsetenv("BENCH_ZIPF_EXPONENT");
    for (const size_t total : {0, 127, 129}) {
      rejects([&] { bench::zipfSplits(total, 4); });
    }
    for (const int world : {0, -1, 33}) {
      rejects([&] { bench::zipfSplits(1024, world); });
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
