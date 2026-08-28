#include <cstddef>
#include <stdexcept>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/nccl_collectives.cuh>
#include <purlin/benchmark/nccl_runtime.cuh>
#include <purlin/benchmark/report.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::NcclRuntime runtime(argc, argv);
    bench::printHeader(runtime);

    const size_t maximumTotal = bench::checkedMultiply(options.maxBytes, runtime.world);
    bench::DeviceBuffer<std::byte> source(maximumTotal, runtime.stream);
    bench::DeviceBuffer<std::byte> destination(maximumTotal, runtime.stream);
    bench::DeviceBuffer<std::byte> reference(maximumTotal, runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t peerBytes) {
      const size_t total = bench::checkedMultiply(peerBytes, runtime.world);
      bench::fillBytePattern(source.get(), total, runtime.rank, runtime.stream);
      for (int sourceRank = 0; sourceRank < runtime.world; ++sourceRank) {
        bench::fillBytePattern(reference.get() + sourceRank * peerBytes, peerBytes,
          sourceRank, runtime.stream, runtime.rank * peerBytes);
      }
      bench::ncclAllToAll(source.get(), destination.get(), peerBytes, runtime.rank,
        runtime.world, runtime.communicator, runtime.stream);

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(), total,
          runtime.stream), total);

      const auto operation = [&] {
        bench::ncclAllToAll(source.get(), destination.get(), peerBytes, runtime.rank,
          runtime.world, runtime.communicator, runtime.stream);
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      runtime.checkAsyncError();

      bench::printResult(runtime, options, "all_to_all", total, "uint8",
        total, milliseconds, errorPercentage);
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
