// Zero-staging suite for the fixed-size collectives: all2all, reduceScatter and
// allReduce. AllGather has its own file, tests/agzs.cu, which also carries the
// rendezvous controls.
//
// Every source buffer here lives on the NVSHMEM symmetric heap, because that is
// what zero-staging requires; destinations are ordinary device memory, which is
// all the contract asks for.
#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

#include <purlin/host/all2all.cuh>
#include <purlin/host/allReduce.cuh>
#include <purlin/host/reduceScatter.cuh>

using DataType = __nv_bfloat16;

namespace {

constexpr auto ZERO = purlin::Staging::zero;

void runAll2All(bench::PurlinRuntime& runtime, const bench::Options& options,
  const uint32_t seed) {
  const size_t maximumTotal = bench::checkedMultiply(options.maxBytes, runtime.world);
  bench::SymmetricBuffer source(maximumTotal, runtime.world, runtime.rank, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> destination(maximumTotal, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> reference(maximumTotal, runtime.stream);
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();

  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t peerBytes) {
    const size_t total = bench::checkedMultiply(peerBytes, runtime.world);
    for (int peer = 0; peer < runtime.world; ++peer) {
      bench::fillRandomBytes(source.get() + peer * peerBytes, peerBytes,
        bench::pairSeed(seed, runtime.rank, peer), runtime.stream);
      bench::fillRandomBytes(reference.get() + peer * peerBytes, peerBytes,
        bench::pairSeed(seed, peer, runtime.rank), runtime.stream);
    }
    purlin::all2all<ARCH, ZERO>(source.get(), destination.get(), peerBytes,
      runtime.context, runtime.stream);

    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxByteMismatches(destination.get(), reference.get(), total, runtime.stream), total);

    const auto operation = [&] {
      purlin::all2all<ARCH, ZERO>(source.get(), destination.get(), peerBytes,
        runtime.context, runtime.stream);
    };
    const double milliseconds = bench::measureOperation(
      runtime.stream, MPI_COMM_WORLD, options, operation);
    bench::printPurlinResult(runtime, options, {
      .collective = "all_to_all_zs",
      .datatype = "uint8",
      .totalBytes = total,
      .logicalBytes = total,
      .purlinMilliseconds = milliseconds,
      .errorPercentage = errorPercentage,
    });
  });
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
}

void runReduceScatter(bench::PurlinRuntime& runtime, const bench::Options& options,
  const uint32_t seed) {
  const size_t maximumLocalElements = options.maxBytes / sizeof(DataType);
  const size_t maximumTotal = bench::checkedMultiply(options.maxBytes, runtime.world);
  bench::SymmetricBuffer source(maximumTotal, runtime.world, runtime.rank, runtime.stream);
  bench::DeviceBuffer<DataType> destination(maximumLocalElements, runtime.stream);
  bench::DeviceBuffer<DataType> referenceSources(
    bench::checkedMultiply(maximumLocalElements, runtime.world), runtime.stream);
  bench::DeviceBuffer<DataType> reference(maximumLocalElements, runtime.stream);
  auto* typedSource = reinterpret_cast<DataType*>(source.get());
  auto* destinationBytes = reinterpret_cast<cuda::std::byte*>(destination.get());
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();

  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t localBytes) {
    const size_t localElements = localBytes / sizeof(DataType);
    for (int destinationRank = 0; destinationRank < runtime.world; ++destinationRank) {
      bench::fillRandomReduction(typedSource + destinationRank * localElements,
        localElements, bench::reduceScatterSeed(seed, runtime.rank, destinationRank),
        runtime.stream);
    }
    bench::fillRandomReduceScatterReferenceSources(referenceSources.get(),
      localElements, seed, runtime.world, runtime.rank, runtime.stream);
    bench::computeReductionReference(referenceSources.get(), reference.get(),
      localElements, runtime.world, runtime.stream);
    purlin::reduceScatter<ARCH, DataType, purlin::ReduceOp::add, ZERO>(
      source.get(), destinationBytes, localBytes, runtime.context, runtime.stream);

    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxMismatches(destination.get(), reference.get(), localElements, runtime.stream),
      localElements);

    const auto operation = [&] {
      purlin::reduceScatter<ARCH, DataType, purlin::ReduceOp::add, ZERO>(
        source.get(), destinationBytes, localBytes, runtime.context, runtime.stream);
    };
    const double milliseconds = bench::measureOperation(
      runtime.stream, MPI_COMM_WORLD, options, operation);
    const size_t totalBytes = bench::checkedMultiply(localBytes, runtime.world);
    bench::printPurlinResult(runtime, options, {
      .collective = "reduce_scatter_zs",
      .datatype = bench::dataTypeName<DataType>(),
      .totalBytes = totalBytes,
      .logicalBytes = totalBytes,
      .purlinMilliseconds = milliseconds,
      .errorPercentage = errorPercentage,
    });
  });
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
}

void runAllReduce(bench::PurlinRuntime& runtime, const bench::Options& options,
  const uint32_t seed) {
  const size_t maximumElements = options.maxBytes / sizeof(DataType);
  bench::SymmetricBuffer source(options.maxBytes, runtime.world, runtime.rank, runtime.stream);
  // Not in place: peers read this rank's source while it writes its own
  // destination, so aliasing them is undefined behaviour under zero-staging.
  bench::DeviceBuffer<DataType> destination(maximumElements, runtime.stream);
  bench::DeviceBuffer<DataType> referenceSources(
    bench::checkedMultiply(maximumElements, runtime.world), runtime.stream);
  bench::DeviceBuffer<DataType> reference(maximumElements, runtime.stream);
  auto* typedSource = reinterpret_cast<DataType*>(source.get());
  auto* destinationBytes = reinterpret_cast<cuda::std::byte*>(destination.get());
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();

  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
    const size_t elements = bytes / sizeof(DataType);
    bench::fillRandomReduction(typedSource, elements,
      bench::allReduceSeed(seed, runtime.rank), runtime.stream);
    bench::fillRandomAllReduceReferenceSources(
      referenceSources.get(), elements, seed, runtime.world, runtime.stream);
    bench::computeReductionReference(referenceSources.get(), reference.get(),
      elements, runtime.world, runtime.stream);
    purlin::allReduce<ARCH, DataType, purlin::ReduceOp::add, ZERO>(
      source.get(), destinationBytes, bytes, runtime.context, runtime.stream);

    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxMismatches(destination.get(), reference.get(), elements, runtime.stream),
      elements);

    const auto operation = [&] {
      purlin::allReduce<ARCH, DataType, purlin::ReduceOp::add, ZERO>(
        source.get(), destinationBytes, bytes, runtime.context, runtime.stream);
    };
    const double milliseconds = bench::measureOperation(
      runtime.stream, MPI_COMM_WORLD, options, operation);
    bench::printPurlinResult(runtime, options, {
      .collective = "all_reduce_zs",
      .datatype = bench::dataTypeName<DataType>(),
      .totalBytes = bytes,
      .logicalBytes = bytes,
      .purlinMilliseconds = milliseconds,
      .errorPercentage = errorPercentage,
    });
  });
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::validatePurlinOptions(options);
    if (options.minBytes % sizeof(DataType) != 0 || options.maxBytes % sizeof(DataType) != 0) {
      throw std::invalid_argument("reduction sizes must be divisible by sizeof(DataType)");
    }
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);
    const uint32_t seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);

    runAll2All(runtime, options, seed);
    runReduceScatter(runtime, options, seed);
    runAllReduce(runtime, options, seed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
