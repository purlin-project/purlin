#ifndef PURLIN_SUPPORT_BENCHMARK_PURLIN_REPORT_CUH
#define PURLIN_SUPPORT_BENCHMARK_PURLIN_REPORT_CUH

#include <cstddef>
#include <cstdio>

#include "benchmark.cuh"
#include "purlin_runtime.cuh"

namespace bench {

struct PurlinResult {
  const char* collective = "";
  const char* datatype = "";
  size_t totalBytes = 0;
  size_t logicalBytes = 0;
  double purlinMilliseconds = 0.0;
  double errorPercentage = 0.0;
};

__host__
inline void printPurlinHeader(const PurlinRuntime& runtime) {
  if (runtime.rank == 0) {
    std::puts("collective,world,totalBytes,datatype,lat(us),bw(GB/s),error(%),"
              "GPUName,warmup,runs,graph_launches");
  }
}

__host__
inline void printPurlinResult(const PurlinRuntime& runtime, const Options& options,
  const PurlinResult& result) {
  if (runtime.rank != 0) return;
  std::printf("%s, %d, %zu, %s, %.4f, %.4f",
    result.collective, runtime.world, result.totalBytes, result.datatype,
    result.purlinMilliseconds * 1000.0,
    bandwidth(result.logicalBytes, result.purlinMilliseconds));
  std::printf(", %.9g, \"%s\", %d, %d, %d\n", result.errorPercentage,
    runtime.deviceProperties.name,
    effectiveWarmup(options.graphLaunches, options.runs, options.warmup),
    options.runs, options.graphLaunches);
}

} // namespace bench

#endif
