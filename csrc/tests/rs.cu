#include <cstddef>
#include <stdexcept>

#include <cuda_fp16.h>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

#include <purlin/host/reduceScatter.cuh>

using DataType = __nv_bfloat16;

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    const bool predictable = ARCH >= 900 && options.reductionMode == purlin::ReductionMode::nonDeterministic;
    if (options.minBytes % sizeof(DataType) != 0 || options.maxBytes % sizeof(DataType) != 0) {
      throw std::invalid_argument("ReduceScatter byte sizes must be divisible by sizeof(DataType)");
    }
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);
    const uint32_t seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);

    const size_t maximumLocalElements = options.maxBytes / sizeof(DataType);
    bench::DeviceBuffer<DataType> source(
      bench::checkedMultiply(maximumLocalElements, runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> destination(maximumLocalElements, runtime.stream);
    bench::DeviceBuffer<DataType> referenceSources(
      bench::checkedMultiply(maximumLocalElements, runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> reference(maximumLocalElements, runtime.stream);
    auto* sourceBytes = reinterpret_cast<cuda::std::byte*>(source.get());
    auto* destinationBytes = reinterpret_cast<cuda::std::byte*>(destination.get());

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t localBytes) {
      const size_t localElements = localBytes / sizeof(DataType);
      for (int destinationRank = 0; destinationRank < runtime.world; ++destinationRank) {
        bench::fillRandomReduction(source.get() + destinationRank * localElements,
          localElements, bench::reduceScatterSeed(seed, runtime.rank, destinationRank),
          runtime.stream, predictable);
      }
      bench::fillRandomReduceScatterReferenceSources(referenceSources.get(),
        localElements, seed, runtime.world, runtime.rank, runtime.stream, predictable);
      bench::computeReductionReference(referenceSources.get(), reference.get(),
        localElements, runtime.world, runtime.stream);
      purlin::dispatchReductionMode(options.reductionMode, [&]<purlin::ReductionMode mode> {
        purlin::reduceScatter<ARCH, DataType, purlin::ReduceOp::add, mode>(sourceBytes, destinationBytes, localBytes,
          runtime.context, runtime.stream);
      });

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxMismatches(destination.get(), reference.get(), localElements,
          runtime.stream), localElements);

      const auto operation = [&] {
        purlin::dispatchReductionMode(options.reductionMode, [&]<purlin::ReductionMode mode> {
          purlin::reduceScatter<ARCH, DataType, purlin::ReduceOp::add, mode>(sourceBytes, destinationBytes, localBytes,
            runtime.context, runtime.stream);
        });
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      const size_t totalBytes = bench::checkedMultiply(localBytes, runtime.world);
      bench::printPurlinResult(runtime, options, {
        .collective = "reduce_scatter",
        .datatype = bench::dataTypeName<DataType>(),
        .totalBytes = totalBytes,
        .logicalBytes = totalBytes,
        .purlinMilliseconds = milliseconds,
        .errorPercentage = errorPercentage,
      });
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
