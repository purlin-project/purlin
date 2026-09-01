#include <cstddef>
#include <stdexcept>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>
#include <purlin/benchmark/variable_counts.cuh>

#include <purlin/host/allGather.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);
    const uint32_t seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);

    const auto maximumSizes = bench::allGatherSizes(options.maxBytes, runtime.world);
    const size_t maximumPeerBytes = bench::maximumBytes(maximumSizes);
    const size_t maximumTotalBytes = bench::totalBytes(maximumSizes);

    bench::DeviceBuffer<cuda::std::byte> source(maximumPeerBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(maximumTotalBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> reference(maximumTotalBytes, runtime.stream);
    bench::DeviceBuffer<size_t> deviceSizes(runtime.world, runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const auto sizes = bench::allGatherSizes(bytes, runtime.world);
      const auto offsets = bench::offsets(sizes);
      const size_t total = bench::totalBytes(sizes);
      runtime.context.vState = bench::makePurlinVState(sizes, offsets, runtime.rank);
      CHECK_CUDA(cudaMemcpyAsync(deviceSizes.get(), sizes.data(),
        sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));

      bench::fillRandomBytes(source.get(), sizes[runtime.rank],
        bench::gatherSeed(seed, runtime.rank), runtime.stream);
      // Replay every peer's seeded fill locally to build the expected output.
      for (int peer = 0; peer < runtime.world; ++peer) {
        bench::fillRandomBytes(reference.get() + offsets[peer], sizes[peer],
          bench::gatherSeed(seed, peer), runtime.stream);
      }
      const auto purlinOperation = [&] {
        purlin::allGatherV<ARCH>(source.get(), destination.get(), deviceSizes.get(),
          runtime.context, runtime.stream);
      };
      purlinOperation();

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(), total,
          runtime.stream), total);

      const double purlinMilliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, purlinOperation);

      bench::printPurlinResult(runtime, options, {
        .collective = "all_gather_v",
        .datatype = "uint8",
        .totalBytes = total,
        .logicalBytes = total,
        .purlinMilliseconds = purlinMilliseconds,
        .errorPercentage = errorPercentage,
      });
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
