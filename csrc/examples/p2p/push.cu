#include <cstddef>
#include <stdexcept>
#include <string>

#include <contrib/symm_mem.cuh>
#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/benchmark/kernel.cuh>
#include <purlin/core.cuh>

constexpr auto threads = 256;
constexpr auto unrollFactor = ARCH < 800 ? 8 : 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 8;
constexpr auto elementsPerThread = 2;
constexpr auto nArch = purlin::normalizeArch<ARCH>();
using PurlinConfig = purlin::Configuration<
    threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor
>;

struct Args {
  cuda::std::byte* const src;
  cuda::std::byte* const dst;
  const size_t bytes;
  const cuda::fast_mod_div<long int> blocks;
};

constexpr int P2P_TURNOVER_THRESHOLD = ARCH >= 900 ? (1024 * 1024) : (512 * 1024);
template<typename PurlinAtom>
__launch_bounds__(PurlinAtom::THREADS, 1)
__global__ void p2pK(const __grid_constant__ Args kArgs) {
  extern __shared__ __align__(bench::sharedMemoryAlignment) cuda::std::byte workspace[];
  purlin::superCopy<PurlinAtom>(kArgs.dst, kArgs.src, kArgs.bytes, workspace, kArgs.blocks);
}

void p2pHost(const bench::KernelOptions& options) {
  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  if (world != 2) {
    nvshmem_finalize();
    throw std::runtime_error("Two processes required");
  }
  const auto device = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CHECK_CUDA(cudaSetDevice(device));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  using PurlinAtom = purlin::Atom<nArch, PurlinConfig>;
  constexpr auto sharedBytes = PurlinAtom::COPY_PIPELINE_SMEM_BYTES;
  bench::configureKernel(p2pK<PurlinAtom>, sharedBytes, prop);
  const auto blockLimit = cuda::std::min(options.maxBlocks, 64);
  if (blockLimit <= 0) throw std::invalid_argument("maxBlocks must be positive");
  if (rank == 0) {
    printf("bytes,purlin(ms),purlin(GB/s),error(%%),nArch,GPUName,threads,pipeStages,stageExtent,unrollFactor,"
           "SMsOnGPU,blocks,warmup,runs,graph_launches\n");
  }

  purlin::NvshmemMemory memory;
  auto symmetric = memory.allocate_zeroed(options.maxBytes, alignment, stream);
  auto* peerBuffer = static_cast<cuda::std::byte*>(symmetric.peers[1 - rank]);
  auto* localSymmetric = static_cast<cuda::std::byte*>(symmetric.local);
  const auto seed = bench::broadcastRandomSeed(rank, options.seed);
  bench::reportSeed(rank, seed);
  {
    bench::DeviceBuffer<cuda::std::byte> localBuffer(options.maxBytes, stream);
    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      // Both ranks replay the same data: one provides the source, the other the reference.
      bench::fillRandomBytes(localBuffer.get(), bytes, seed, stream);
      auto blocks = static_cast<int>(cuda::std::min(
        cuda::ceil_div(bytes, static_cast<size_t>(PurlinAtom::THREADS * alignment)),
        static_cast<size_t>(blockLimit)));
      if (bytes >= P2P_TURNOVER_THRESHOLD) {
        blocks = cuda::std::min(bytes / PurlinAtom::COPY_PIPELINE_BYTES, static_cast<size_t>(blockLimit));
      }
      const bool usedPipelining = bytes / blocks >= PurlinAtom::COPY_PIPELINE_BYTES;
      const Args args{
        .src = localBuffer.get(),
        .dst = peerBuffer,
        .bytes = bytes,
        .blocks = cuda::fast_mod_div<long int>{blocks}
      };
      const auto operation = [&] {
        if (rank == 0) p2pK<PurlinAtom><<<blocks, threads, sharedBytes, stream>>>(args);
      };
      nvshmemx_barrier_all_on_stream(stream);
      operation();
      CHECK_CUDA(cudaGetLastError());
      nvshmemx_barrier_all_on_stream(stream);
      const auto errors = rank == 1
        ? bench::matxByteMismatches(localSymmetric, localBuffer.get(), bytes, stream) : 0ull;
      const double errorPercentage = bench::maxErrorPercentage(errors, bytes);
      const double milliseconds = bench::measureOperation(stream, MPI_COMM_WORLD, options, operation, 0);
      if (rank == 0) {
        printf("%zu, %lf, %lf, %lf, %d, %s, %d, %s, %s, %d, %d, %d, %d, %d, %d\n",
          bytes, milliseconds, bench::bandwidth(bytes, milliseconds), errorPercentage, nArch, prop.name,
          threads,
          usedPipelining ? std::to_string(pipeStages).c_str() : "N/A",
          usedPipelining ? std::to_string(elementsPerThread).c_str() : "N/A",
          unrollFactor, prop.multiProcessorCount, blocks,
          bench::effectiveWarmup(options.graphLaunches, options.runs, options.warmup),
          options.runs, options.graphLaunches);
      }
    });
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  memory.deallocate(symmetric);
  CHECK_CUDA(cudaStreamDestroy(stream));
  nvshmem_finalize();
}

// ./push [minBytes] [maxBytes] [maxBlocks] [graphLaunches] [runs] [warmup] [seed]
int main(int argc, char** argv) {
  try {
    const auto options = bench::parseKernelOptions(argc, argv, ARCH >= 1000 ? 16 : 8);
    if (options.minBytes % alignment != 0 || options.maxBytes % alignment != 0) {
      throw std::invalid_argument("Sizes must be multiples of " + std::to_string(alignment) + " bytes");
    }
    p2pHost(options);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
