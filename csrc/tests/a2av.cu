#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "common/benchmark.cuh"
#include "common/data.cuh"
#include "common/device_buffer.cuh"
#include "common/matx_validation.cuh"
#include "common/nccl_collectives.cuh"
#include "common/nccl_communicator.cuh"
#include "common/purlin_report.cuh"
#include "common/purlin_runtime.cuh"
#include "common/variable_counts.cuh"

#include <purlin/host/all2all.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    bench::PurlinRuntime runtime;
    bench::NcclCommunicator nccl;
    nccl.initialize(runtime.rank, runtime.world);
    bench::printPurlinHeader(runtime);

    const auto maximumSends = bench::allToAllSendSplits(
      options.maxBytes, runtime.rank, runtime.world);
    const auto maximumReceives = bench::allToAllReceiveSplits(
      options.maxBytes, runtime.rank, runtime.world);
    const size_t maximumBufferBytes = std::max(
      bench::totalBytes(maximumSends), bench::totalBytes(maximumReceives));
    if (bench::totalBytes(maximumSends) > runtime.context.stagingTRSize) {
      throw std::runtime_error("AllToAllV send bytes exceed the Purlin staging limit");
    }

    bench::DeviceBuffer<cuda::std::byte> source(maximumBufferBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(maximumBufferBytes, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> ncclDestination(maximumBufferBytes, runtime.stream);
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

      bench::fillBytePattern(source.get(), sendTotal, runtime.rank, runtime.stream);
      const auto purlinOperation = [&] {
        purlin::all2allV<ARCH>(source.get(), destination.get(), deviceSends.get(),
          deviceReceives.get(), runtime.context, runtime.stream);
      };
      const auto ncclOperation = [&] {
        bench::ncclAllToAllV(source.get(), ncclDestination.get(), sends, receives,
          sendOffsets, receiveOffsets, runtime.rank, runtime.world,
          nccl.get(), runtime.stream);
      };
      purlinOperation();
      ncclOperation();

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), ncclDestination.get(),
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
