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

    const auto maximumSends = bench::allToAllSendSplits(options.maxBytes, runtime.rank, runtime.world);
    const auto maximumReceives = bench::allToAllReceiveSplits(options.maxBytes, runtime.rank, runtime.world);
    bench::DeviceBuffer<std::byte> source(
      bench::totalBytes(maximumSends), runtime.stream);
    bench::DeviceBuffer<std::byte> destination(
      bench::totalBytes(maximumReceives), runtime.stream);
    bench::DeviceBuffer<std::byte> reference(
      bench::totalBytes(maximumReceives), runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const auto sends = bench::allToAllSendSplits(bytes, runtime.rank, runtime.world);
      const auto receives = bench::allToAllReceiveSplits(bytes, runtime.rank, runtime.world);
      const auto sendOffsets = bench::offsets(sends);
      const auto receiveOffsets = bench::offsets(receives);
      bench::validatePeerCounts(sends, receives, MPI_COMM_WORLD);

      const size_t sendTotal = bench::totalBytes(sends);
      const size_t receiveTotal = bench::totalBytes(receives);
      bench::fillBytePattern(source.get(), sendTotal, runtime.rank, runtime.stream);
      for (int sourceRank = 0; sourceRank < runtime.world; ++sourceRank) {
        const auto peerSends = bench::allToAllSplitsForSource(
          bytes, sourceRank, runtime.world);
        const auto peerOffsets = bench::offsets(peerSends);
        bench::fillBytePattern(reference.get() + receiveOffsets[sourceRank],
          receives[sourceRank], sourceRank, runtime.stream, peerOffsets[runtime.rank]);
      }
      bench::ncclAllToAllV(source.get(), destination.get(), sends, receives,
        sendOffsets, receiveOffsets, runtime.rank, runtime.world,
        runtime.communicator, runtime.stream);

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(), receiveTotal,
          runtime.stream), receiveTotal);

      const auto operation = [&] {
        bench::ncclAllToAllV(source.get(), destination.get(), sends, receives,
          sendOffsets, receiveOffsets, runtime.rank, runtime.world,
          runtime.communicator, runtime.stream);
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      runtime.checkAsyncError();

      bench::printResult(runtime, options, "all_to_all_v", sendTotal, "uint8",
        std::max(sendTotal, receiveTotal), milliseconds, errorPercentage);
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
