#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>
#include <purlin/benchmark/variable_counts.cuh>

#include <purlin/host/all2all.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);
    const uint32_t seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);

    const auto maximumSends = bench::allToAllSendSplits(
      options.maxBytes, runtime.rank, runtime.world);
    const auto maximumReceives = bench::allToAllReceiveSplits(
      options.maxBytes, runtime.rank, runtime.world);
    const size_t maximumBufferBytes = std::max(
      bench::totalBytes(maximumSends), bench::totalBytes(maximumReceives));

    bench::DeviceBuffer<cuda::std::byte> source(maximumBufferBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(maximumBufferBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> reference(maximumBufferBytes, runtime.stream);
    bench::DeviceBuffer<size_t> deviceSends(runtime.world, runtime.stream);
    bench::DeviceBuffer<size_t> deviceReceives(runtime.world, runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const auto sends = bench::allToAllSendSplits(bytes, runtime.rank, runtime.world);
      const auto receives = bench::allToAllReceiveSplits(bytes, runtime.rank, runtime.world);
      const auto sendOffsets = bench::offsets(sends);
      const auto receiveOffsets = bench::offsets(receives);
      const size_t sendTotal = bench::totalBytes(sends);
      const size_t receiveTotal = bench::totalBytes(receives);

      bench::validatePeerCounts(sends, receives, MPI_COMM_WORLD);
      runtime.context.vState = bench::makePurlinAllToAllVState(
        sends, receives, sendOffsets, runtime.rank);
      CHECK_CUDA(cudaMemcpyAsync(deviceSends.get(), sends.data(),
        sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));
      CHECK_CUDA(cudaMemcpyAsync(deviceReceives.get(), receives.data(),
        sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));

      // Each (source, destination) chunk is its own seeded stream, so the
      // receiver can replay its incoming chunks without any communication.
      for (int peer = 0; peer < runtime.world; ++peer) {
        bench::fillRandomBytes(source.get() + sendOffsets[peer], sends[peer],
          bench::pairSeed(seed, runtime.rank, peer), runtime.stream);
        bench::fillRandomBytes(reference.get() + receiveOffsets[peer], receives[peer],
          bench::pairSeed(seed, peer, runtime.rank), runtime.stream);
      }
      const auto purlinOperation = [&] {
        purlin::all2allV<ARCH>(source.get(), destination.get(), deviceSends.get(),
          deviceReceives.get(), runtime.context, runtime.stream);
      };
      purlinOperation();

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(),
          receiveTotal, runtime.stream), receiveTotal);

      const double purlinMilliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, purlinOperation);

      bench::printPurlinResult(runtime, options, {
        .collective = "all_to_all_v",
        .datatype = "uint8",
        .totalBytes = sendTotal,
        .logicalBytes = receiveTotal,
        .purlinMilliseconds = purlinMilliseconds,
        .errorPercentage = errorPercentage,
      });
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
