#include <cstddef>
#include <stdexcept>

#include <cuda_fp16.h>

#include "common/benchmark.cuh"
#include "common/data.cuh"
#include "common/device_buffer.cuh"
#include "common/matx_validation.cuh"
#include "common/purlin_report.cuh"
#include "common/purlin_runtime.cuh"
#include "common/variable_counts.cuh"

#include <purlin/host/reduceScatter.cuh>

using DataType = __nv_bfloat16;

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    if (options.minBytes % sizeof(DataType) != 0 || options.maxBytes % sizeof(DataType) != 0) {
      throw std::invalid_argument("ReduceScatterV byte sizes must be divisible by sizeof(DataType)");
    }
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);

    const auto maximumSizes = bench::reduceScatterSizes(options.maxBytes, runtime.world);
    const size_t maximumTotalBytes = bench::totalBytes(maximumSizes);
    if (maximumTotalBytes > runtime.context.stagingTRSize) {
      throw std::runtime_error("ReduceScatterV total bytes exceed the Purlin staging limit");
    }

    bench::DeviceBuffer<DataType> source(
      maximumTotalBytes / sizeof(DataType), runtime.stream);
    bench::DeviceBuffer<DataType> destination(
      maximumSizes[runtime.rank] / sizeof(DataType), runtime.stream);
    bench::DeviceBuffer<DataType> referenceSources(bench::checkedMultiply(
      maximumSizes[runtime.rank] / sizeof(DataType), runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> reference(
      maximumSizes[runtime.rank] / sizeof(DataType), runtime.stream);
    bench::DeviceBuffer<size_t> deviceSizes(runtime.world, runtime.stream);
    auto* sourceBytes = reinterpret_cast<cuda::std::byte*>(source.get());
    auto* destinationBytes = reinterpret_cast<cuda::std::byte*>(destination.get());

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const auto sizes = bench::reduceScatterSizes(bytes, runtime.world);
      const auto offsets = bench::offsets(sizes);
      const size_t total = bench::totalBytes(sizes);
      for (const size_t size : sizes) {
        if (size % sizeof(DataType) != 0) {
          throw std::runtime_error("ReduceScatterV produced a misaligned peer size");
        }
      }
      runtime.context.vState = bench::makePurlinVState(sizes, offsets, runtime.rank);
      CHECK_CUDA(cudaMemcpyAsync(deviceSizes.get(), sizes.data(),
        sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));

      const uint32_t seed = bench::broadcastRandomSeed(runtime.rank);
      for (int destinationRank = 0; destinationRank < runtime.world; ++destinationRank) {
        bench::fillRandomReduction(
          source.get() + offsets[destinationRank] / sizeof(DataType),
          sizes[destinationRank] / sizeof(DataType),
          bench::reduceScatterSeed(seed, runtime.rank, destinationRank), runtime.stream);
      }
      const size_t localElements = sizes[runtime.rank] / sizeof(DataType);
      bench::fillRandomReduceScatterReferenceSources(referenceSources.get(),
        localElements, seed, runtime.world, runtime.rank, runtime.stream);
      bench::computeReductionReference(referenceSources.get(), reference.get(),
        localElements, runtime.world, runtime.stream);
      purlin::reduceScatterV<ARCH, DataType>(sourceBytes, destinationBytes, deviceSizes.get(),
        runtime.context, runtime.stream);

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxMismatches(destination.get(), reference.get(), localElements,
          runtime.stream), localElements);

      const auto operation = [&] {
        purlin::reduceScatterV<ARCH, DataType>(sourceBytes, destinationBytes, deviceSizes.get(),
          runtime.context, runtime.stream);
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);

      bench::printPurlinResult(runtime, options, {
        .collective = "reduce_scatter_v",
        .datatype = bench::dataTypeName<DataType>(),
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
