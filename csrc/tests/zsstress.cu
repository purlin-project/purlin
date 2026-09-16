// Two stresses for the multicast box, not part of the suite proper.
//
//   1. Reuse stress for the zero-staged allReduce: iteration k fills the
//      symmetric source with a pattern derived from k and reduces into slice k
//      of the destination, moving on immediately. Every slice is verified at the
//      end against the rank-ordered reference for that iteration, so a peer that
//      load-reduced a source after its owner had rewritten it corrupts the slice.
//      ctx.peerDst stays null: the intermediate lives in staging, which is the
//      residency multimem forces and the one whose only reuse protection is the
//      sense-bit double buffer. With NVLS live the fused kernel takes its
//      multimem branch; with PURLIN_DISABLE_MULTIMEM=1 it is the unicast control.
//   2. In-place stress for the staged latency-regime allReduce (src == dst) in
//      the peer-striped band (world > 4, bytes <= 16 KiB), where the multimem
//      send and the reduce must visit the same elements per thread. Each
//      iteration is verified synchronously because the next one overwrites src.
//
// args: minBytes maxBytes [graphLaunches ignored] iterations [warmup ignored] seed
#include <cstddef>
#include <cstdio>
#include <stdexcept>

#include <cuda_bf16.h>

#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/purlin_report.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

#include <purlin/host/allReduce.cuh>

using DataType = __nv_bfloat16;

namespace {

constexpr auto ZERO = purlin::Staging::zero;

__global__ void occupyKernel(const long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
    __threadfence_block();
  }
}

__host__ __device__ inline uint32_t iterationSeed(const uint32_t seed, const int k) {
  return bench::pairSeed(seed, 0x5a5a, k);
}

// Returns (total mismatches, worst single-iteration mismatches).
std::pair<unsigned long long, unsigned long long> runReuseStress(bench::PurlinRuntime& runtime,
  const size_t bytes, const int iterations, const uint32_t seed, const bool directDst = false) {
  const int world = runtime.world;
  const int rank = runtime.rank;
  const size_t elements = bytes / sizeof(DataType);
  const size_t totalElements = elements * static_cast<size_t>(iterations);

  bench::SymmetricBuffer source(bytes, world, rank, runtime.stream);
  // The direct-dst variant needs a symmetric destination so its multicast alias exists.
  bench::SymmetricBuffer symDestination(directDst ? bytes * static_cast<size_t>(iterations) : 16, world, rank, runtime.stream);
  bench::DeviceBuffer<DataType> plainDestination(directDst ? 8 : totalElements, runtime.stream);
  auto* destination = directDst ? reinterpret_cast<DataType*>(symDestination.get()) : plainDestination.get();
  bench::DeviceBuffer<DataType> referenceSources(
    bench::checkedMultiply(elements, world), runtime.stream);
  bench::DeviceBuffer<DataType> reference(elements, runtime.stream);
  auto* typedSource = reinterpret_cast<DataType*>(source.get());

  cudaStream_t contention = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&contention, cudaStreamNonBlocking));
  int smCount = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, runtime.device));
  const int contendingBlocks = smCount > 1 ? smCount / 2 : 1;
  const bool slowRank = rank == 0;

  runtime.context.peerSrc = source.peers();
  runtime.context.mcSrc = source.mc();
  runtime.context.peerDst = nullptr;

  for (int k = 0; k < iterations; ++k) {
    runtime.context.mcDst = directDst ? symDestination.mc() + static_cast<size_t>(k) * bytes : nullptr;
    bench::fillRandomReduction(typedSource, elements,
      bench::allReduceSeed(iterationSeed(seed, k), rank), runtime.stream);
    if (slowRank) {
      occupyKernel<<<contendingBlocks, 128, 0, contention>>>(300000);
      CHECK_CUDA(cudaGetLastError());
    }
    purlin::allReduce<ARCH, DataType, purlin::ReduceOp::add, ZERO>(
      source.get(),
      reinterpret_cast<cuda::std::byte*>(destination + static_cast<size_t>(k) * elements),
      bytes, runtime.context, runtime.stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(runtime.stream));
  CHECK_CUDA(cudaStreamSynchronize(contention));
  CHECK_CUDA(cudaStreamDestroy(contention));

  unsigned long long total = 0, worst = 0;
  for (int k = 0; k < iterations; ++k) {
    bench::fillRandomAllReduceReferenceSources(referenceSources.get(), elements,
      iterationSeed(seed, k), world, runtime.stream);
    bench::computeReductionReference(referenceSources.get(), reference.get(),
      elements, world, runtime.stream);
    const auto m = bench::matxMismatches(destination + static_cast<size_t>(k) * elements,
      reference.get(), elements, runtime.stream);
    total += m; if (m > worst) worst = m;
  }
  runtime.context.peerSrc = nullptr;
  runtime.context.mcSrc = nullptr;
  runtime.context.mcDst = nullptr;
  return {total, worst};
}

std::pair<unsigned long long, unsigned long long> runInPlaceLR(bench::PurlinRuntime& runtime,
  const size_t bytes, const int iterations, const uint32_t seed) {
  const int world = runtime.world;
  const int rank = runtime.rank;
  const size_t elements = bytes / sizeof(DataType);
  bench::DeviceBuffer<DataType> buffer(elements, runtime.stream);
  bench::DeviceBuffer<DataType> referenceSources(
    bench::checkedMultiply(elements, world), runtime.stream);
  bench::DeviceBuffer<DataType> reference(elements, runtime.stream);
  unsigned long long total = 0, worst = 0;
  for (int k = 0; k < iterations; ++k) {
    bench::fillRandomReduction(buffer.get(), elements,
      bench::allReduceSeed(iterationSeed(seed, k), rank), runtime.stream);
    auto* bytesPtr = reinterpret_cast<cuda::std::byte*>(buffer.get());
    purlin::allReduce<ARCH, DataType, purlin::ReduceOp::add>(
      bytesPtr, bytesPtr, bytes, runtime.context, runtime.stream);
    bench::fillRandomAllReduceReferenceSources(referenceSources.get(), elements,
      iterationSeed(seed, k), world, runtime.stream);
    bench::computeReductionReference(referenceSources.get(), reference.get(),
      elements, world, runtime.stream);
    const auto m = bench::matxMismatches(buffer.get(), reference.get(), elements, runtime.stream);
    total += m; if (m > worst) worst = m;
  }
  return {total, worst};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv);
    bench::PurlinRuntime runtime;
    const auto seed = bench::broadcastRandomSeed(runtime.rank, options.seed);
    bench::reportSeed(runtime.rank, seed);
    const int iterations = options.runs;
    const bool multimem = runtime.context.mcStagingTR != nullptr;
    if (runtime.rank == 0) {
      std::printf("check,bytes,iterations,multimem,mismatches,worst_iteration,result\n");
    }
    bool failed = false;
    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      for (const bool directDst : {false, true}) {
        const auto [total, worst] = runReuseStress(runtime, bytes, iterations, seed, directDst);
        unsigned long long gTotal = 0, gWorst = 0;
        MPI_Allreduce(&total, &gTotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&worst, &gWorst, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (runtime.rank == 0) {
          std::printf("%s,%zu,%d,%d,%llu,%llu,%s\n", directDst ? "reuse_stress_arzs_mcdst" : "reuse_stress_arzs",
            bytes, iterations, multimem ? 1 : 0, gTotal, gWorst, gTotal == 0 ? "PASS" : "FAIL");
          std::fflush(stdout);
        }
        failed = failed || gTotal != 0;
      }
    });
    const size_t lrMax = std::min<size_t>(options.maxBytes, 16UL * 1024UL);
    if (options.minBytes <= lrMax) {
      bench::forEachPowerOfTwoSize(options.minBytes, lrMax, [&](const size_t bytes) {
        const auto [total, worst] = runInPlaceLR(runtime, bytes, iterations, seed);
        unsigned long long gTotal = 0, gWorst = 0;
        MPI_Allreduce(&total, &gTotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&worst, &gWorst, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (runtime.rank == 0) {
          std::printf("inplace_lr_staged,%zu,%d,%d,%llu,%llu,%s\n", bytes, iterations, multimem ? 1 : 0,
            gTotal, gWorst, gTotal == 0 ? "PASS" : "FAIL");
          std::fflush(stdout);
        }
        failed = failed || gTotal != 0;
      });
    }
    return failed ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
