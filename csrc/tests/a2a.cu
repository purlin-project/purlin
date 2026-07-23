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

#include <purlin/host/all2all.cuh>

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    bench::PurlinRuntime runtime;
    bench::NcclCommunicator nccl;
    nccl.initialize(runtime.rank, runtime.world);
    bench::printPurlinHeader(runtime);

    const size_t maximumTotal = bench::checkedMultiply(options.maxBytes, runtime.world);
    bench::DeviceBuffer<cuda::std::byte> source(maximumTotal, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(maximumTotal, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> reference(maximumTotal, runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t peerBytes) {
      const size_t total = bench::checkedMultiply(peerBytes, runtime.world);
      bench::fillBytePattern(source.get(), total, runtime.rank, runtime.stream);
      purlin::all2all<ARCH>(source.get(), destination.get(), peerBytes, runtime.context, runtime.stream);
      bench::ncclAllToAll(source.get(), reference.get(), peerBytes, runtime.rank,
        runtime.world, nccl.get(), runtime.stream);

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
