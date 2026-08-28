#include <cstddef>
#include <stdexcept>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/nccl_runtime.cuh>
#include <purlin/benchmark/report.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::NcclRuntime runtime(argc, argv);
    bench::printHeader(runtime);

    bench::DeviceBuffer<std::byte> source(options.maxBytes, runtime.stream);
    bench::DeviceBuffer<std::byte> destination(
      bench::checkedMultiply(options.maxBytes, runtime.world), runtime.stream);
    bench::DeviceBuffer<std::byte> reference(
      bench::checkedMultiply(options.maxBytes, runtime.world), runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const size_t total = bench::checkedMultiply(bytes, runtime.world);
      bench::fillBytePattern(source.get(), bytes, runtime.rank, runtime.stream);
      for (int sourceRank = 0; sourceRank < runtime.world; ++sourceRank) {
        bench::fillBytePattern(reference.get() + sourceRank * bytes, bytes,
          sourceRank, runtime.stream);
      }
      NCCL_CHECK(ncclAllGather(source.get(), destination.get(), bytes, ncclUint8,
        runtime.communicator, runtime.stream));

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(), total,
          runtime.stream), total);

      const auto operation = [&] {
        NCCL_CHECK(ncclAllGather(source.get(), destination.get(), bytes, ncclUint8,
          runtime.communicator, runtime.stream));
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      runtime.checkAsyncError();

      bench::printResult(runtime, options, "all_gather", total, "uint8",
        total, milliseconds, errorPercentage);
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
