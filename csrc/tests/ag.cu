#include <cstddef>
#include <stdexcept>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

#include <purlin/host/allGather.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);
    const uint32_t seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);

    bench::DeviceBuffer<cuda::std::byte> source(options.maxBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(
      bench::checkedMultiply(options.maxBytes, runtime.world), runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> reference(
      bench::checkedMultiply(options.maxBytes, runtime.world), runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const size_t total = bench::checkedMultiply(bytes, runtime.world);
      bench::fillRandomBytes(source.get(), bytes,
        bench::gatherSeed(seed, runtime.rank), runtime.stream);
      purlin::allGather<ARCH>(source.get(), destination.get(), bytes, runtime.context, runtime.stream);
      // Replay every peer's seeded fill locally to build the expected output.
      for (int peer = 0; peer < runtime.world; ++peer) {
        bench::fillRandomBytes(reference.get() + peer * bytes, bytes,
          bench::gatherSeed(seed, peer), runtime.stream);
      }

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(), total,
          runtime.stream), total);

      const auto operation = [&] {
        purlin::allGather<ARCH>(source.get(), destination.get(), bytes, runtime.context, runtime.stream);
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      bench::printPurlinResult(runtime, options, {
        .collective = "all_gather",
        .datatype = "uint8",
        .totalBytes = total,
        .logicalBytes = total,
        .purlinMilliseconds = milliseconds,
        .errorPercentage = errorPercentage,
      });
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
