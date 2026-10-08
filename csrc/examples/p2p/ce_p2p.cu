#include <cstddef>
#include <stdexcept>
#include <string>

#include <contrib/symm_mem.cuh>
#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>

void p2pHost(const bench::Options& options) {
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
  if (rank == 0) {
    printf("bytes,ce(ms),ce(GB/s),error(%%),GPUName,warmup,runs,graph_launches\n");
  }
  purlin::NvshmemMemory memory;
  auto symmetric = memory.allocate_zeroed(options.maxBytes, bench::sharedMemoryAlignment, stream);
  auto* peerBuffer = static_cast<cuda::std::byte*>(symmetric.peers[1 - rank]);
  const auto seed = bench::broadcastRandomSeed(rank, options.seed);
  bench::reportSeed(rank, seed);
  {
    bench::DeviceBuffer<cuda::std::byte> source(options.maxBytes, stream);
    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      // Rank zero sends this data; rank one uses it as the reference.
      bench::fillRandomBytes(source.get(), bytes, seed, stream);
      const auto operation = [&] {
        if (rank == 0) {
          CHECK_CUDA(cudaMemcpyAsync(peerBuffer, source.get(), bytes, cudaMemcpyDeviceToDevice, stream));
        }
      };
      nvshmemx_barrier_all_on_stream(stream);
      operation();
      nvshmemx_barrier_all_on_stream(stream);
      const auto errors = rank == 1
        ? bench::matxByteMismatches(symmetric.local, source.get(), bytes, stream) : 0ull;
      const double errorPercentage = bench::maxErrorPercentage(errors, bytes);
      const double milliseconds = bench::measureOperation(stream, MPI_COMM_WORLD, options, operation, 0);
      if (rank == 0) {
        printf("%zu, %lf, %lf, %lf, %s, %d, %d, %d\n",
          bytes, milliseconds, bench::bandwidth(bytes, milliseconds), errorPercentage, prop.name,
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

// ./ce_p2p [minBytes] [maxBytes] [graphLaunches] [runs] [warmup] [seed]
int main(int argc, char** argv) {
  try {
    const auto options = bench::parseOptions(argc, argv, {.runs = 16, .warmup = 16});
    p2pHost(options);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
