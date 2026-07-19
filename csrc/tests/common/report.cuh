#ifndef PURLIN_TESTS_COMMON_REPORT_CUH
#define PURLIN_TESTS_COMMON_REPORT_CUH

#include <cstddef>
#include <cstdio>

#include "benchmark.cuh"
#include "nccl_runtime.cuh"

namespace bench {

inline void printHeader(const NcclRuntime& runtime) {
  if (runtime.rank == 0) {
    std::puts("collective,world,totalBytes,datatype,lat(us),bw(GB/s),error(%),"
              "GPUName,warmup,runs,graph_launches");
  }
}

inline void printResult(const NcclRuntime& runtime, const Options& options,
  const char* collective, const size_t totalBytes, const char* datatype,
  const size_t logicalBytes, const double milliseconds,
  const double errorPercentage) {
  if (runtime.rank != 0) return;
  const double algorithmBandwidth = bandwidth(logicalBytes, milliseconds);
  std::printf("%s, %d, %zu, %s, %.4f, %.4f, %.9g, \"%s\", %d, %d, %d\n",
    collective, runtime.world, totalBytes, datatype, milliseconds * 1000.0,
    algorithmBandwidth,
    errorPercentage, runtime.deviceProperties.name,
    effectiveWarmup(options.graphLaunches, options.runs, options.warmup),
    options.runs, options.graphLaunches);
}

} // namespace bench

#endif
