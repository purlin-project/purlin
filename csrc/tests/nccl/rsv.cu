#include <cstddef>
#include <stdexcept>

#include <cuda_fp16.h>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/nccl_collectives.cuh>
#include <purlin/benchmark/nccl_runtime.cuh>
#include <purlin/benchmark/report.cuh>
#include <purlin/benchmark/variable_counts.cuh>

using DataType = __half;

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    if (options.minBytes % sizeof(DataType) != 0 || options.maxBytes % sizeof(DataType) != 0) {
      throw std::invalid_argument("ReduceScatterV byte sizes must be divisible by sizeof(DataType)");
    }
    bench::NcclRuntime runtime(argc, argv);
    bench::printHeader(runtime);

    const auto maximumSizes = bench::reduceScatterSizes(options.maxBytes, runtime.world);
    bench::DeviceBuffer<DataType> source(
      bench::totalBytes(maximumSizes) / sizeof(DataType), runtime.stream);
    bench::DeviceBuffer<DataType> destination(
      maximumSizes[runtime.rank] / sizeof(DataType), runtime.stream);
    bench::DeviceBuffer<DataType> referenceSources(bench::checkedMultiply(
      maximumSizes[runtime.rank] / sizeof(DataType), runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> reference(
      maximumSizes[runtime.rank] / sizeof(DataType), runtime.stream);

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const auto sizes = bench::reduceScatterSizes(bytes, runtime.world);
      const auto offsets = bench::offsets(sizes);
      for (const size_t size : sizes) {
        if (size % sizeof(DataType) != 0) {
          throw std::invalid_argument("ReduceScatterV split is not DataType-aligned");
        }
      }

      const size_t total = bench::totalBytes(sizes);
      const size_t localElements = sizes[runtime.rank] / sizeof(DataType);
      bench::fillReductionPattern(source.get(), total / sizeof(DataType), runtime.rank,
        0, runtime.stream);
      bench::fillPatternReferenceSources(referenceSources.get(), localElements,
        runtime.world, offsets[runtime.rank] / sizeof(DataType), runtime.stream);
      bench::computeReductionReference(referenceSources.get(), reference.get(),
        localElements, runtime.world, runtime.stream);
      bench::ncclReduceScatterV(source.get(), destination.get(), sizes, offsets,
        runtime.rank, runtime.world, runtime.communicator, runtime.stream);

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxMismatches(destination.get(), reference.get(), localElements,
          runtime.stream), localElements);

      const auto operation = [&] {
        bench::ncclReduceScatterV(source.get(), destination.get(), sizes, offsets,
          runtime.rank, runtime.world, runtime.communicator, runtime.stream);
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      runtime.checkAsyncError();

      bench::printResult(runtime, options, "reduce_scatter_v", total,
        bench::dataTypeName<DataType>(), total, milliseconds, errorPercentage);
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
