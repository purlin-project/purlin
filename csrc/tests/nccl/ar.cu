#include <cstddef>
#include <stdexcept>

#include <cuda_fp16.h>

#include "benchmark.cuh"
#include "data.cuh"
#include "device_buffer.cuh"
#include "matx_validation.cuh"
#include "nccl_collectives.cuh"
#include "nccl_runtime.cuh"
#include "report.cuh"

using DataType = __half;

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    if (options.minBytes % sizeof(DataType) != 0 || options.maxBytes % sizeof(DataType) != 0) {
      throw std::invalid_argument("AllReduce byte sizes must be divisible by sizeof(DataType)");
    }
    bench::NcclRuntime runtime(argc, argv);
    bench::printHeader(runtime);

    const size_t maximumElements = options.maxBytes / sizeof(DataType);
    bench::DeviceBuffer<DataType> source(maximumElements, runtime.stream);
    bench::DeviceBuffer<DataType> destination(maximumElements, runtime.stream);
    bench::DeviceBuffer<DataType> referenceSources(
      bench::checkedMultiply(maximumElements, runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> reference(maximumElements, runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const size_t elements = bytes / sizeof(DataType);
      bench::fillReductionPattern(source.get(), elements, runtime.rank, 0, runtime.stream);
      bench::fillPatternReferenceSources(
        referenceSources.get(), elements, runtime.world, 0, runtime.stream);
      bench::computeReductionReference(referenceSources.get(), reference.get(),
        elements, runtime.world, runtime.stream);
      NCCL_CHECK(ncclAllReduce(source.get(), destination.get(), elements,
        bench::ncclDataType<DataType>(),
        ncclSum, runtime.communicator, runtime.stream));

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxMismatches(destination.get(), reference.get(), elements, runtime.stream),
        elements);

      const auto operation = [&] {
        NCCL_CHECK(ncclAllReduce(source.get(), destination.get(), elements,
          bench::ncclDataType<DataType>(),
          ncclSum, runtime.communicator, runtime.stream));
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      runtime.checkAsyncError();

      bench::printResult(runtime, options, "all_reduce",
        bytes, bench::dataTypeName<DataType>(),
        bytes, milliseconds, errorPercentage);
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
