#include <cstddef>
#include <stdexcept>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

#include <purlin/host/all2all.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);
    const uint32_t seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);

    const size_t maximumTotal = bench::checkedMultiply(options.maxBytes, runtime.world);
    bench::DeviceBuffer<cuda::std::byte> source(maximumTotal, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(maximumTotal, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> reference(maximumTotal, runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t peerBytes) {
      const size_t total = bench::checkedMultiply(peerBytes, runtime.world);
      // Each (source, destination) chunk is its own seeded stream, so the
      // receiver can replay its incoming chunks without any communication.
      for (int peer = 0; peer < runtime.world; ++peer) {
        bench::fillRandomBytes(source.get() + peer * peerBytes, peerBytes,
          bench::pairSeed(seed, runtime.rank, peer), runtime.stream);
        bench::fillRandomBytes(reference.get() + peer * peerBytes, peerBytes,
          bench::pairSeed(seed, peer, runtime.rank), runtime.stream);
      }
      purlin::all2all<ARCH>(source.get(), destination.get(), peerBytes, runtime.context, runtime.stream);

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(), total,
          runtime.stream), total);

      const auto operation = [&] {
        purlin::all2all<ARCH>(source.get(), destination.get(), peerBytes, runtime.context, runtime.stream);
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      bench::printPurlinResult(runtime, options, {
        .collective = "all_to_all",
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
