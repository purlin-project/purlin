#ifndef PURLIN_SUPPORT_BENCHMARK_BENCHMARK_CUH
#define PURLIN_SUPPORT_BENCHMARK_BENCHMARK_CUH

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "checks.cuh"

namespace bench {

struct Options {
  size_t minBytes = 128;
  size_t maxBytes = 128 * 1024 * 1024;
  int graphLaunches = 8;
  int runs = 128;
  int warmup = 128;
  uint32_t seed = 0; // 0 draws a random data seed; nonzero replays that seed.
};

struct Measurement {
  double milliseconds = 0.0;
  double errorPercentage = 0.0;
};

inline constexpr int sharedMemoryAlignment = 128;

inline size_t parseSize(const std::string& text) {
  size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument("Size must be a positive finite value: " + text);
  }

  size_t multiplier = 1;
  if (consumed < text.size()) {
    if (consumed + 1 != text.size()) {
      throw std::invalid_argument("Invalid size: " + text);
    }
    switch (text[consumed]) {
      case 'k':
      case 'K': multiplier = 1024ull; break;
      case 'm':
      case 'M': multiplier = 1024ull * 1024ull; break;
      case 'g':
      case 'G': multiplier = 1024ull * 1024ull * 1024ull; break;
      default: throw std::invalid_argument("Invalid size suffix: " + text);
    }
  }

  const long double bytes = static_cast<long double>(value) * multiplier;
  if (bytes > static_cast<long double>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("Size is too large: " + text);
  }
  return static_cast<size_t>(bytes);
}

inline Options parseOptions(const int argc, char** argv) {
  Options options{};
  if (argc > 1) options.minBytes = parseSize(argv[1]);
  if (argc > 2) options.maxBytes = parseSize(argv[2]);
  if (argc > 3) options.graphLaunches = std::stoi(argv[3]);
  if (argc > 4) options.runs = std::stoi(argv[4]);
  if (argc > 5) options.warmup = std::stoi(argv[5]);
  if (argc > 6) {
    const unsigned long seed = std::stoul(argv[6]);
    if (seed > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("Seed must fit in 32 bits");
    }
    options.seed = static_cast<uint32_t>(seed);
  }
  if (argc > 7) {
    throw std::invalid_argument(
      "Usage: <program> [minBytes] [maxBytes] [graphLaunches] [runs] [warmup] [seed]");
  }
  if (options.minBytes > options.maxBytes) {
    throw std::invalid_argument("minBytes must not exceed maxBytes");
  }
  if (!std::has_single_bit(options.minBytes) || !std::has_single_bit(options.maxBytes)) {
    throw std::invalid_argument("Minimum and maximum sizes must be powers of two");
  }
  if (options.graphLaunches < 0 || options.runs <= 0 || options.warmup < 0) {
    throw std::invalid_argument("graphLaunches and warmup must be non-negative; runs must be positive");
  }
  return options;
}

inline size_t checkedMultiply(const size_t left, const size_t right) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
    throw std::overflow_error("Benchmark buffer size overflow");
  }
  return left * right;
}

template<typename Function>
inline void forEachPowerOfTwoSize(const size_t minimum, const size_t maximum, Function&& function) {
  for (size_t bytes = minimum;;) {
    function(bytes);
    if (bytes >= maximum) break;
    if (bytes > maximum / 2) {
      throw std::invalid_argument("Maximum size is not reachable by doubling from the minimum size");
    }
    bytes *= 2;
  }
}

inline double maxAcrossRanks(const double value, MPI_Comm communicator = MPI_COMM_WORLD) {
  double maximum = 0.0;
  MPI_CHECK(MPI_Allreduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, communicator));
  return maximum;
}

inline unsigned long long maxAcrossRanks(const unsigned long long value,
  MPI_Comm communicator = MPI_COMM_WORLD) {
  unsigned long long maximum = 0;
  MPI_CHECK(MPI_Allreduce(&value, &maximum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator));
  return maximum;
}

inline double maxErrorPercentage(const unsigned long long errors,
  const size_t checkedValues, MPI_Comm communicator = MPI_COMM_WORLD) {
  const double localPercentage = checkedValues == 0
    ? (errors == 0 ? 0.0 : 100.0)
    : 100.0 * static_cast<double>(errors) / static_cast<double>(checkedValues);
  return maxAcrossRanks(localPercentage, communicator);
}

template<typename OptionsLike, typename Operation>
inline double measureOperation(cudaStream_t stream, MPI_Comm communicator,
  const OptionsLike& options, Operation&& operation) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  float milliseconds = 0.0f;
  const int graphLaunches = options.graphLaunches;
  CHECK_CUDA(cudaStreamSynchronize(stream));
  MPI_CHECK(MPI_Barrier(communicator));

  if (graphLaunches > 0) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int iteration = 0; iteration < options.runs; ++iteration) operation();
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

    // Warm one captured batch before timing.
    CHECK_CUDA(cudaGraphLaunch(executable, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(communicator));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int launch = 0; launch < graphLaunches; ++launch) {
      CHECK_CUDA(cudaGraphLaunch(executable, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= static_cast<float>(options.runs * graphLaunches);

    CHECK_CUDA(cudaGraphExecDestroy(executable));
    CHECK_CUDA(cudaGraphDestroy(graph));
  } else {
    for (int iteration = 0; iteration < options.warmup; ++iteration) operation();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(communicator));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int iteration = 0; iteration < options.runs; ++iteration) operation();
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= static_cast<float>(options.runs);
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return maxAcrossRanks(static_cast<double>(milliseconds), communicator);
}

inline const char* mode(const int graphLaunches) {
  return graphLaunches > 0 ? "graph" : "stream";
}

inline int effectiveWarmup(const int graphLaunches, const int runs, const int warmup) {
  return graphLaunches > 0 ? runs : warmup;
}

inline double bandwidth(const size_t logicalBytes, const double milliseconds) {
  return static_cast<double>(logicalBytes) / 1e9 / (milliseconds * 1e-3);
}

inline int reportFailure(const std::exception& error) {
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  if (initialized) MPI_Finalized(&finalized);
  std::fprintf(stderr, "Benchmark failed: %s\n", error.what());
  if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  return EXIT_FAILURE;
}

} // namespace bench

#endif
