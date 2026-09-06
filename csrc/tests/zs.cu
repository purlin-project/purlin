// Zero-staging suite for the fixed-size collectives: all2all, reduceScatter and
// allReduce. AllGather has its own file, tests/agzs.cu, which also carries the
// rendezvous controls.
//
// Every source buffer here lives on the NVSHMEM symmetric heap, because that is
// what zero-staging requires; destinations are ordinary device memory, which is
// all the contract asks for.
#include <cstddef>
#include <stdexcept>

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

// The intermediate residency is a caller choice: ctx.peerDst moves the fused
// composition's shard into dst, and null leaves it in staging. Only the second
// is bound by staging capacity, so both are swept.
void runAllReduce(bench::PurlinRuntime& runtime, const bench::Options& options,
  const uint32_t seed, const bool intermediateInDst) {
  const size_t maximumElements = options.maxBytes / sizeof(DataType);
  bench::SymmetricBuffer source(options.maxBytes, runtime.world, runtime.rank, runtime.stream);
  // A peer-visible destination lets the fused path keep its intermediate in dst
  // and skip staging entirely. Not in place: at two ranks that is undefined
  // behaviour, since the bypass writes the bytes its peer is reading.
  bench::SymmetricBuffer destinationBuffer(options.maxBytes, runtime.world, runtime.rank,
    runtime.stream);
  auto* destination = reinterpret_cast<DataType*>(destinationBuffer.get());
  bench::DeviceBuffer<DataType> referenceSources(
    bench::checkedMultiply(maximumElements, runtime.world), runtime.stream);
  bench::DeviceBuffer<DataType> reference(maximumElements, runtime.stream);
  auto* typedSource = reinterpret_cast<DataType*>(source.get());
  auto* destinationBytes = destinationBuffer.get();
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();
  runtime.context.peerDst = intermediateInDst ? destinationBuffer.peers() : nullptr;

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
      bench::matxMismatches(destination, reference.get(), elements, runtime.stream),
      elements);

    const auto operation = [&] {
      purlin::allReduce<ARCH, DataType, purlin::ReduceOp::add, ZERO>(
        source.get(), destinationBytes, bytes, runtime.context, runtime.stream);
    };
    const double milliseconds = bench::measureOperation(
      runtime.stream, MPI_COMM_WORLD, options, operation);
    bench::printPurlinResult(runtime, options, {
      .collective = intermediateInDst ? "all_reduce_zs" : "all_reduce_zs_staged_dst",
      .datatype = bench::dataTypeName<DataType>(),
      .totalBytes = bytes,
      .logicalBytes = bytes,
      .purlinMilliseconds = milliseconds,
      .errorPercentage = errorPercentage,
    });
  });
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
  runtime.context.peerDst = nullptr;
}


// The world-2 bypass reduces straight into the packed layout, which hides the
// fused reduce-then-gather composition -- the only path that can hold its
// intermediate in the destination. Force the composition on so both residencies
// of that intermediate are covered wherever this suite runs, not only above two
// ranks.
void runFusedAllReduce(bench::PurlinRuntime& runtime, const bench::Options& options,
  const uint32_t seed, const bool intermediateInDst) {
  using Cfg = purlin::Configuration<128, 16, 8, 2, 2>;
  using AtomT = purlin::Atom<ARCH, Cfg>;
  // One gather block per peer is what fetches each shard, and under zero-staging
  // it is also the per-peer wait the exit rendezvous is built on.
  constexpr int gatherBlocks = purlin::MAX_RANKS_PER_DOMAIN;
  constexpr int reduceBlocks = 32;
  using ZeroCfg = purlin::WithZeroStaging<purlin::CollectiveConfig<
    purlin::CollectiveType::nonChunked, 32, gatherBlocks, 4UL * 1024 * 1024>>;

  const size_t maximumElements = options.maxBytes / sizeof(DataType);
  bench::SymmetricBuffer source(options.maxBytes, runtime.world, runtime.rank, runtime.stream);
  bench::SymmetricBuffer destinationBuffer(options.maxBytes, runtime.world, runtime.rank,
    runtime.stream);
  bench::DeviceBuffer<DataType> referenceSources(
    bench::checkedMultiply(maximumElements, runtime.world), runtime.stream);
  bench::DeviceBuffer<DataType> reference(maximumElements, runtime.stream);
  auto* typedSource = reinterpret_cast<DataType*>(source.get());
  auto* destination = reinterpret_cast<DataType*>(destinationBuffer.get());
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();
  runtime.context.peerDst = intermediateInDst ? destinationBuffer.peers() : nullptr;

  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
    // The composition shards the payload, so a size it cannot split evenly says
    // nothing about the path under test.
    if (bytes % (static_cast<size_t>(runtime.world) * 16) != 0) {
      return;
    }
    // A staged intermediate is laid out one shard per rank across the sense
    // half, so it needs the whole payload's worth of staging. This calls the
    // launcher directly and so skips the tuned entry's capacity check, which
    // hands anything larger to the staged cyclic band; honour the same ceiling
    // here rather than asking for a configuration the library declines.
    if (!intermediateInDst && bytes > runtime.context.stagingTRSize) {
      return;
    }
    const size_t elements = bytes / sizeof(DataType);
    bench::fillRandomReduction(typedSource, elements,
      bench::allReduceSeed(seed, runtime.rank), runtime.stream);
    bench::fillRandomAllReduceReferenceSources(
      referenceSources.get(), elements, seed, runtime.world, runtime.stream);
    bench::computeReductionReference(referenceSources.get(), reference.get(),
      elements, runtime.world, runtime.stream);

    const auto operation = [&] {
      purlin::launchAllReduceThroughput<AtomT, DataType, ZeroCfg, purlin::World2Bypass::no>(
        source.get(), destinationBuffer.get(), bytes, runtime.context,
        gatherBlocks, reduceBlocks, runtime.stream);
    };
    operation();

    const double errorPercentage = bench::maxErrorPercentage(
      bench::matxMismatches(destination, reference.get(), elements, runtime.stream), elements);
    const double milliseconds = bench::measureOperation(
      runtime.stream, MPI_COMM_WORLD, options, operation);
    bench::printPurlinResult(runtime, options, {
      .collective = intermediateInDst ? "all_reduce_fused_dst_zs" : "all_reduce_fused_staged_zs",
      .datatype = bench::dataTypeName<DataType>(),
      .totalBytes = bytes,
      .logicalBytes = bytes,
      .purlinMilliseconds = milliseconds,
      .errorPercentage = errorPercentage,
    });
  });
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
  runtime.context.peerDst = nullptr;
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
    runAllReduce(runtime, options, seed, /*intermediateInDst=*/true);
    runAllReduce(runtime, options, seed, /*intermediateInDst=*/false);
    runFusedAllReduce(runtime, options, seed, /*intermediateInDst=*/true);
    runFusedAllReduce(runtime, options, seed, /*intermediateInDst=*/false);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
