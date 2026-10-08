#include <cstddef>
#include <stdexcept>

#include <cuda_fp16.h>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

#include <purlin/host/allReduce.cuh>

using DataType = __nv_bfloat16;

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    const bool predictable = ARCH >= 900 && options.reductionMode == purlin::ReductionMode::nonDeterministic;
    if (options.minBytes % sizeof(DataType) != 0 || options.maxBytes % sizeof(DataType) != 0) {
      throw std::invalid_argument("AllReduce byte sizes must be divisible by sizeof(DataType)");
    }
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);
    const uint32_t seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);

    const size_t maximumElements = options.maxBytes / sizeof(DataType);
    bench::DeviceBuffer<DataType> source(maximumElements, runtime.stream);
    bench::DeviceBuffer<DataType> destination(maximumElements, runtime.stream);
    bench::DeviceBuffer<DataType> referenceSources(
      bench::checkedMultiply(maximumElements, runtime.world), runtime.stream);
    bench::DeviceBuffer<DataType> reference(maximumElements, runtime.stream);
    auto* sourceBytes = reinterpret_cast<cuda::std::byte*>(source.get());
    auto* destinationBytes = reinterpret_cast<cuda::std::byte*>(destination.get());

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const size_t elements = bytes / sizeof(DataType);
      bench::fillRandomReduction(source.get(), elements,
        bench::allReduceSeed(seed, runtime.rank), runtime.stream, predictable);
      bench::fillRandomAllReduceReferenceSources(
        referenceSources.get(), elements, seed, runtime.world, runtime.stream, predictable);
      bench::computeReductionReference(referenceSources.get(), reference.get(),
        elements, runtime.world, runtime.stream);
      purlin::dispatchReductionMode(options.reductionMode, [&]<purlin::ReductionMode mode> {
        purlin::allReduce<ARCH, DataType, purlin::ReduceOp::add, mode>(sourceBytes, destinationBytes, bytes,
          runtime.context, runtime.stream);
      });

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxMismatches(destination.get(), reference.get(), elements, runtime.stream),
        elements);

      const auto operation = [&] {
        purlin::dispatchReductionMode(options.reductionMode, [&]<purlin::ReductionMode mode> {
          purlin::allReduce<ARCH, DataType, purlin::ReduceOp::add, mode>(sourceBytes, destinationBytes, bytes,
            runtime.context, runtime.stream);
        });
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      bench::printPurlinResult(runtime, options, {
        .collective = "all_reduce",
        .datatype = bench::dataTypeName<DataType>(),
        .totalBytes = bytes,
        .logicalBytes = bytes,
        .purlinMilliseconds = milliseconds,
        .errorPercentage = errorPercentage,
      });
    });
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
