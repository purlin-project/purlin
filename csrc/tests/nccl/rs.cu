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
      throw std::invalid_argument("ReduceScatter byte sizes must be divisible by sizeof(DataType)");
    }
    bench::NcclRuntime runtime(argc, argv);
    bench::printHeader(runtime);

    const size_t maximumLocalElements = options.maxBytes / sizeof(DataType);
    bench::DeviceBuffer<DataType> source(
      bench::checkedMultiply(maximumLocalElements, runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> destination(maximumLocalElements, runtime.stream);
    bench::DeviceBuffer<DataType> referenceSources(
      bench::checkedMultiply(maximumLocalElements, runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> reference(maximumLocalElements, runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t localBytes) {
      const size_t localElements = localBytes / sizeof(DataType);
      const size_t totalElements = bench::checkedMultiply(localElements, runtime.world);
      bench::fillReductionPattern(source.get(), totalElements, runtime.rank, 0, runtime.stream);
      bench::fillPatternReferenceSources(referenceSources.get(), localElements,
        runtime.world, runtime.rank * localElements, runtime.stream);
      bench::computeReductionReference(referenceSources.get(), reference.get(),
        localElements, runtime.world, runtime.stream);
      NCCL_CHECK(ncclReduceScatter(source.get(), destination.get(), localElements,
        bench::ncclDataType<DataType>(), ncclSum, runtime.communicator, runtime.stream));

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxMismatches(destination.get(), reference.get(), localElements,
          runtime.stream), localElements);

      const auto operation = [&] {
        NCCL_CHECK(ncclReduceScatter(source.get(), destination.get(), localElements,
          bench::ncclDataType<DataType>(), ncclSum, runtime.communicator, runtime.stream));
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      runtime.checkAsyncError();

      const size_t totalBytes = bench::checkedMultiply(localBytes, runtime.world);
      bench::printResult(runtime, options, "reduce_scatter", totalBytes,
        bench::dataTypeName<DataType>(), totalBytes, milliseconds, errorPercentage);
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
