#include <cstddef>
#include <stdexcept>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/nccl_collectives.cuh>
#include <purlin/benchmark/nccl_runtime.cuh>
#include <purlin/benchmark/report.cuh>
#include <purlin/benchmark/variable_counts.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::NcclRuntime runtime(argc, argv);
    bench::printHeader(runtime);

    const auto maximumSizes = bench::allGatherSizes(options.maxBytes, runtime.world);
    bench::DeviceBuffer<std::byte> source(
      bench::maximumBytes(maximumSizes), runtime.stream);
    bench::DeviceBuffer<std::byte> destination(
      bench::totalBytes(maximumSizes), runtime.stream);
    bench::DeviceBuffer<std::byte> reference(
      bench::totalBytes(maximumSizes), runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const auto sizes = bench::allGatherSizes(bytes, runtime.world);
      const auto offsets = bench::offsets(sizes);
      const size_t total = bench::totalBytes(sizes);
      bench::fillBytePattern(source.get(), sizes[runtime.rank], runtime.rank, runtime.stream);
      for (int sourceRank = 0; sourceRank < runtime.world; ++sourceRank) {
        bench::fillBytePattern(reference.get() + offsets[sourceRank], sizes[sourceRank],
          sourceRank, runtime.stream);
      }
      bench::ncclAllGatherV(source.get(), destination.get(), sizes, offsets, runtime.rank,
        runtime.world, runtime.communicator, runtime.stream);

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(), total,
          runtime.stream), total);

      const auto operation = [&] {
        bench::ncclAllGatherV(source.get(), destination.get(), sizes, offsets, runtime.rank,
          runtime.world, runtime.communicator, runtime.stream);
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      runtime.checkAsyncError();

      bench::printResult(runtime, options, "all_gather_v", total, "uint8",
        total, milliseconds, errorPercentage);
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
