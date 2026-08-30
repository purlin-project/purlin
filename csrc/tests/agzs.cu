// Zero-staging AllGather.
//
// Two checks run here, and the second is the one that matters. The first is an
// ordinary correctness sweep against NCCL. The second overwrites the source
// buffer immediately after every call, with no synchronisation, which is the
// only thing that exercises the exit rendezvous: delete the rendezvous and the
// correctness sweep still passes, because nothing there ever reuses a buffer
// while a peer might still be reading it.
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

#include <purlin/host/allGather.cuh>

namespace {

// Overwrite the source with a value no pattern produces, and hold it there, so
// a peer still reading this rank's buffer picks up 0xFF rather than a subtle
// mismatch against a neighbouring iteration.
__global__ void poisonKernel(std::byte* source, const size_t bytes, const long long cycles) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = index; i < bytes; i += stride) {
    reinterpret_cast<unsigned char*>(source)[i] = 0xffu;
  }
  __threadfence_system();
  const long long start = clock64();
  while (clock64() - start < cycles) {
    __threadfence_block();
  }
}

// Occupy SMs so a rank's consumers run slower without delaying its entry: skew
// only counts if it lands inside the read window, since the entry signal already
// makes every rank wait for every other.
__global__ void occupyKernel(const long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
    __threadfence_block();
  }
}

// Reuse stress. Iteration k fills the source with a pattern derived from k,
// gathers into that iteration's slice of the destination, and immediately moves
// on -- the next fill is queued behind nothing but stream order. Every slice is
// verified at the end, so a peer that read a source after its owner had already
// rewritten it shows up as a mismatch in the slice it corrupted.
size_t runReuseStress(bench::PurlinRuntime& runtime, const size_t bytes, const int iterations,
  const uint32_t seed) {
  const int world = runtime.world;
  const int rank = runtime.rank;
  const size_t sliceBytes = bytes * static_cast<size_t>(world);
  const size_t totalBytes = sliceBytes * static_cast<size_t>(iterations);

  bench::SymmetricBuffer source(bytes, world, rank, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> destination(totalBytes, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> reference(totalBytes, runtime.stream);

  // The contention runs on its own stream so it overlaps the collective rather
  // than being ordered before it.
  cudaStream_t contention = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&contention, cudaStreamNonBlocking));
  int smCount = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, runtime.device));
  // Half the SMs: enough to slow this rank's readers materially, few enough
  // that the collective still makes progress rather than serialising behind it.
  const int contendingBlocks = smCount > 1 ? smCount / 2 : 1;
  const bool slowRank = rank == 0;

  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();

  for (int k = 0; k < iterations; ++k) {
    // Seed by (rank, iteration) so a stale read from a neighbouring iteration
    // cannot coincidentally match.
    bench::fillRandomBytes(source.get(), bytes, bench::pairSeed(seed, rank, k), runtime.stream);
    if (slowRank) {
      occupyKernel<<<contendingBlocks, 128, 0, contention>>>(300000);
      CHECK_CUDA(cudaGetLastError());
    }
    purlin::allGather<ARCH, purlin::Staging::zero>(
      source.get(),
      destination.get() + static_cast<size_t>(k) * sliceBytes,
      bytes, runtime.context, runtime.stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(contention));
  CHECK_CUDA(cudaStreamDestroy(contention));

  // Rank p's contribution to iteration k is the pattern it wrote at that
  // iteration, so the expected buffer is reproducible without any collective.
  for (int k = 0; k < iterations; ++k) {
    for (int peer = 0; peer < world; ++peer) {
      bench::fillRandomBytes(
        reference.get() + static_cast<size_t>(k) * sliceBytes + static_cast<size_t>(peer) * bytes,
        bytes, bench::pairSeed(seed, peer, k), runtime.stream);
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(runtime.stream));

  const auto mismatches = bench::matxByteMismatches(
    destination.get(), reference.get(), totalBytes, runtime.stream);
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
  return static_cast<size_t>(mismatches);
}

// Deterministic skew, and the check that actually has teeth.
//
// SM contention is not enough on its own: the entry signal already forces every
// rank to arrive before any rank reads, so the ranks resynchronise once per
// iteration and the drift stays under kernel-launch latency. Giving one rank far
// fewer blocks is different. Its entry still goes out immediately -- block 0 is
// scheduled first either way -- but its reads then take many times longer, so
// its peers finish, exit, and rewrite their own sources while it is still
// reading them.
//
// Grid sizes may differ across ranks: block-to-peer mapping, arrival counting
// and epoch marking are all rank-local, and signals are indexed by peer rather
// than by block.
size_t runAsymmetricSkew(bench::PurlinRuntime& runtime, const size_t bytes, const int iterations,
  const uint32_t seed) {
  using Cfg = purlin::Configuration<128, 16, 8, 2, 2>;
  using AtomT = purlin::Atom<ARCH, Cfg>;
  // The zero-staged configuration is what selects the pull protocol; nothing
  // below the host carries a separate residency axis.
  using NonChunked = purlin::WithZeroStaging<
    purlin::CollectiveConfig<purlin::CollectiveType::nonChunked, 32, 1, 4UL * 1024 * 1024>>;
  constexpr auto smem = purlin::copySmemBytes<AtomT>();

  const int world = runtime.world;
  const int rank = runtime.rank;
  const size_t sliceBytes = bytes * static_cast<size_t>(world);
  const size_t totalBytes = sliceBytes * static_cast<size_t>(iterations);

  bench::SymmetricBuffer source(bytes, world, rank, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> destination(totalBytes, runtime.stream);
  bench::DeviceBuffer<cuda::std::byte> reference(totalBytes, runtime.stream);
  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();

  // Rank 0 reads with a sixteenth of the blocks, so it is still reading long
  // after every other rank has finished and moved on.
  const int blocks = rank == 0 ? world * 2 : world * 32;

  for (int k = 0; k < iterations; ++k) {
    bench::fillRandomBytes(source.get(), bytes, bench::pairSeed(seed, rank, k), runtime.stream);
    purlin::launchAllGatherKernel<purlin::DataLayout::packed, AtomT, NonChunked, smem>(
        source.get(),
        destination.get() + static_cast<size_t>(k) * sliceBytes,
        bytes, runtime.context, nullptr, blocks, runtime.stream);
    // The fast rank poisons its source the moment its kernel completes and
    // holds the poison long enough to cover the slow rank's remaining reads.
    if (rank != 0) {
      poisonKernel<<<64, 256, 0, runtime.stream>>>(
        reinterpret_cast<std::byte*>(source.get()), bytes, 1000000);
      CHECK_CUDA(cudaGetLastError());
    }
  }

  for (int k = 0; k < iterations; ++k) {
    for (int peer = 0; peer < world; ++peer) {
      bench::fillRandomBytes(
        reference.get() + static_cast<size_t>(k) * sliceBytes + static_cast<size_t>(peer) * bytes,
        bytes, bench::pairSeed(seed, peer, k), runtime.stream);
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(runtime.stream));

  const auto mismatches = bench::matxByteMismatches(
    destination.get(), reference.get(), totalBytes, runtime.stream);
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
  return static_cast<size_t>(mismatches);
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

    // The source lives on the symmetric heap so peers can read it directly;
    // the destination is ordinary device memory, which is all zero-staging
    // asks for.
    bench::SymmetricBuffer source(options.maxBytes, runtime.world, runtime.rank, runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> destination(
      bench::checkedMultiply(options.maxBytes, runtime.world), runtime.stream);
    bench::DeviceBuffer<cuda::std::byte> reference(
      bench::checkedMultiply(options.maxBytes, runtime.world), runtime.stream);

    runtime.context.peerSrc = source.peers();
    runtime.context.mcSrc = source.mc();

    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const size_t total = bench::checkedMultiply(bytes, runtime.world);
      bench::fillRandomBytes(source.get(), bytes,
        bench::gatherSeed(seed, runtime.rank), runtime.stream);
      purlin::allGather<ARCH, purlin::Staging::zero>(
        source.get(), destination.get(), bytes, runtime.context, runtime.stream);
      // Replay every peer's seeded fill locally to build the expected output.
      for (int peer = 0; peer < runtime.world; ++peer) {
        bench::fillRandomBytes(reference.get() + peer * bytes, bytes,
          bench::gatherSeed(seed, peer), runtime.stream);
      }

      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(destination.get(), reference.get(), total,
          runtime.stream), total);

      const auto operation = [&] {
        purlin::allGather<ARCH, purlin::Staging::zero>(
          source.get(), destination.get(), bytes, runtime.context, runtime.stream);
      };
      const double milliseconds = bench::measureOperation(
        runtime.stream, MPI_COMM_WORLD, options, operation);
      bench::printPurlinResult(runtime, options, {
        .collective = "all_gather_zs",
        .datatype = "uint8",
        .totalBytes = total,
        .logicalBytes = total,
        .purlinMilliseconds = milliseconds,
        .errorPercentage = errorPercentage,
      });
    });

    runtime.context.peerSrc = nullptr;
    runtime.context.mcSrc = nullptr;

    // Sizes chosen to straddle the latency threshold, so both the packet path
    // and the pull protocol are stressed.
    constexpr int iterations = 64;
    for (const size_t bytes : {size_t{4096}, size_t{256} * 1024, size_t{4} * 1024 * 1024}) {
      if (bytes > options.maxBytes) continue;
      const auto mismatches = runReuseStress(runtime, bytes, iterations, seed);
      if (runtime.rank == 0) {
        std::printf("reuse_stress, bytes=%zu, iterations=%d, mismatches=%zu, %s\n",
          bytes, iterations, mismatches, mismatches == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
      }
      if (mismatches != 0) {
        throw std::runtime_error("zero-staging reuse stress failed at " +
          std::to_string(bytes) + " bytes with " + std::to_string(mismatches) +
          " mismatching bytes; a peer read a source buffer after its owner "
          "had already rewritten it, which points at the exit rendezvous");
      }
    }
    for (const size_t bytes : {size_t{256} * 1024, size_t{4} * 1024 * 1024}) {
      if (bytes > options.maxBytes) continue;
      const auto mismatches = runAsymmetricSkew(runtime, bytes, iterations, seed);
      if (runtime.rank == 0) {
        std::printf("asymmetric_skew, bytes=%zu, iterations=%d, mismatches=%zu, %s\n",
          bytes, iterations, mismatches, mismatches == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
      }
      if (mismatches != 0) {
        throw std::runtime_error("zero-staging asymmetric-skew check failed at " +
          std::to_string(bytes) + " bytes with " + std::to_string(mismatches) +
          " mismatching bytes; a peer read a source buffer after its owner had "
          "already rewritten it, which points at the exit rendezvous");
      }
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
