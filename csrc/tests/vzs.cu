// Zero-staging suite for the variable-size collectives: allGatherV and
// allToAllV.
//
// allToAllV is the interesting one. It is the only layout pair whose producer-
// side offset a consumer cannot derive: it partitions by a world x world matrix
// and each rank holds only its own row and column, so it knows how many bytes a
// peer will send but not where in that peer's buffer they start. The producer
// therefore ships the offset alongside the epoch flag in one 16-byte packet.
// This file is what exercises that path.
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>
#include <purlin/benchmark/variable_counts.cuh>

#include <cuda_bf16.h>

#include <purlin/host/all2all.cuh>
#include <purlin/host/allGather.cuh>
#include <purlin/host/reduceScatter.cuh>

using DataType = __nv_bfloat16;

namespace {

constexpr auto ZERO = purlin::Staging::zero;

void runAllGatherV(bench::PurlinRuntime& runtime, const bench::Options& options,
  const uint32_t seed) {
  const auto maximumSizes = bench::allGatherSizes(options.maxBytes, runtime.world);
  const size_t maximumPeerBytes = bench::maximumBytes(maximumSizes);
  const size_t maximumTotalBytes = bench::totalBytes(maximumSizes);

  bench::SymmetricBuffer source(maximumPeerBytes, runtime.world, runtime.rank, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> destination(maximumTotalBytes, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> reference(maximumTotalBytes, runtime.stream);
  bench::DeviceBuffer<size_t> deviceSizes(runtime.world, runtime.stream);
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();

  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
    const auto sizes = bench::allGatherSizes(bytes, runtime.world);
    const auto sizeOffsets = bench::offsets(sizes);
    const size_t total = bench::totalBytes(sizes);
    runtime.context.vState = bench::makePurlinVState(sizes, sizeOffsets, runtime.rank);
    CHECK_CUDA(cudaMemcpyAsync(deviceSizes.get(), sizes.data(),
      sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));

    bench::fillRandomBytes(source.get(), sizes[runtime.rank],
      bench::gatherSeed(seed, runtime.rank), runtime.stream);
    // Replay every peer's seeded contribution at its own offset.
    for (int peer = 0; peer < runtime.world; ++peer) {
      bench::fillRandomBytes(reference.get() + sizeOffsets[peer], sizes[peer],
        bench::gatherSeed(seed, peer), runtime.stream);
    }
    const auto operation = [&] {
      purlin::allGatherV<ARCH, ZERO>(source.get(), destination.get(), deviceSizes.get(),
        runtime.context, runtime.stream);
    };
    operation();

    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxByteMismatches(destination.get(), reference.get(), total,
        runtime.stream), total);
    const double milliseconds = bench::measureOperation(
      runtime.stream, MPI_COMM_WORLD, options, operation);
    bench::printPurlinResult(runtime, options, {
      .collective = "all_gather_v_zs",
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

void runAll2AllV(bench::PurlinRuntime& runtime, const bench::Options& options,
  const uint32_t seed) {
  const auto maximumSends = bench::allToAllSendSplits(options.maxBytes, runtime.rank, runtime.world);
  const auto maximumReceives = bench::allToAllReceiveSplits(options.maxBytes, runtime.rank, runtime.world);
  const size_t maximumBufferBytes = std::max(
    bench::totalBytes(maximumSends), bench::totalBytes(maximumReceives));

  bench::SymmetricBuffer source(maximumBufferBytes, runtime.world, runtime.rank, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> destination(maximumBufferBytes, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> reference(maximumBufferBytes, runtime.stream);
  bench::DeviceBuffer<size_t> deviceSends(runtime.world, runtime.stream);
  bench::DeviceBuffer<size_t> deviceReceives(runtime.world, runtime.stream);
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();

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

    for (int peer = 0; peer < runtime.world; ++peer) {
      bench::fillRandomBytes(source.get() + sendOffsets[peer], sends[peer],
        bench::pairSeed(seed, runtime.rank, peer), runtime.stream);
      bench::fillRandomBytes(reference.get() + receiveOffsets[peer], receives[peer],
        bench::pairSeed(seed, peer, runtime.rank), runtime.stream);
    }
    const auto operation = [&] {
      purlin::all2allV<ARCH, ZERO>(source.get(), destination.get(), deviceSends.get(),
        deviceReceives.get(), runtime.context, runtime.stream);
    };
    operation();

    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxByteMismatches(destination.get(), reference.get(),
        receiveTotal, runtime.stream), receiveTotal);
    const double milliseconds = bench::measureOperation(
      runtime.stream, MPI_COMM_WORLD, options, operation);
    bench::printPurlinResult(runtime, options, {
      .collective = "all_to_all_v_zs",
      .datatype = "uint8",
      .totalBytes = receiveTotal,
      .logicalBytes = receiveTotal,
      .purlinMilliseconds = milliseconds,
      .errorPercentage = errorPercentage,
    });
  });
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
}

void runReduceScatterV(bench::PurlinRuntime& runtime, const bench::Options& options,
  const uint32_t seed) {
  const auto maximumSizes = bench::reduceScatterSizes(options.maxBytes, runtime.world);
  const size_t maximumTotalBytes = bench::totalBytes(maximumSizes);
  const size_t maximumLocalElements = maximumSizes[runtime.rank] / sizeof(DataType);

  bench::SymmetricBuffer source(maximumTotalBytes, runtime.world, runtime.rank, runtime.stream);
  bench::DeviceBuffer<DataType> destination(maximumLocalElements, runtime.stream);
  bench::DeviceBuffer<DataType> referenceSources(
    bench::checkedMultiply(maximumLocalElements, runtime.world), runtime.stream);
  bench::DeviceBuffer<DataType> reference(maximumLocalElements, runtime.stream);
  bench::DeviceBuffer<size_t> deviceSizes(runtime.world, runtime.stream);
  auto* typedSource = reinterpret_cast<DataType*>(source.get());
  auto* destinationBytes = reinterpret_cast<cuda::std::byte*>(destination.get());
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();

  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
    const auto sizes = bench::reduceScatterSizes(bytes, runtime.world);
    const auto sizeOffsets = bench::offsets(sizes);
    const size_t total = bench::totalBytes(sizes);
    for (const size_t size : sizes) {
      if (size % sizeof(DataType) != 0) {
        throw std::runtime_error("reduceScatterV produced a misaligned peer size");
      }
    }
    runtime.context.vState = bench::makePurlinVState(sizes, sizeOffsets, runtime.rank);
    CHECK_CUDA(cudaMemcpyAsync(deviceSizes.get(), sizes.data(),
      sizeof(size_t) * runtime.world, cudaMemcpyHostToDevice, runtime.stream));

    for (int destinationRank = 0; destinationRank < runtime.world; ++destinationRank) {
      bench::fillRandomReduction(typedSource + sizeOffsets[destinationRank] / sizeof(DataType),
        sizes[destinationRank] / sizeof(DataType),
        bench::reduceScatterSeed(seed, runtime.rank, destinationRank), runtime.stream);
    }
    const size_t localElements = sizes[runtime.rank] / sizeof(DataType);
    bench::fillRandomReduceScatterReferenceSources(referenceSources.get(),
      localElements, seed, runtime.world, runtime.rank, runtime.stream);
    bench::computeReductionReference(referenceSources.get(), reference.get(),
      localElements, runtime.world, runtime.stream);

    const auto operation = [&] {
      purlin::reduceScatterV<ARCH, DataType, purlin::ReduceOp::add, ZERO>(
        source.get(), destinationBytes, deviceSizes.get(), runtime.context, runtime.stream);
    };
    operation();

    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxMismatches(destination.get(), reference.get(), localElements, runtime.stream),
      localElements);
    const double milliseconds = bench::measureOperation(
      runtime.stream, MPI_COMM_WORLD, options, operation);
    bench::printPurlinResult(runtime, options, {
      .collective = "reduce_scatter_v_zs",
      .datatype = bench::dataTypeName<DataType>(),
      .totalBytes = total,
      .logicalBytes = total,
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
    bench::PurlinRuntime runtime;
    bench::printPurlinHeader(runtime);
    const uint32_t seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);

    runAllGatherV(runtime, options, seed);
    runAll2AllV(runtime, options, seed);
    runReduceScatterV(runtime, options, seed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
