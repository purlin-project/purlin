#include <cstddef>
#include <stdexcept>
#include <string>

#include <contrib/symm_mem.cuh>
#include <purlin/benchmark/benchmark.cuh>
#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/matx_validation.cuh>

void agHost(const bench::Options& options) {
  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  if (world < 2) {
    nvshmem_finalize();
    throw std::runtime_error("At least two processes required");
  }
  CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (rank == 0) {
    printf("world,localBytes,globalBytes,error(%%),warmup,runs,graph_launches,ce(ms),ce(GB/s)\n");
  }

  purlin::NvshmemMemory memory;
  const size_t maximumTotal = bench::checkedMultiply(options.maxBytes, world);
  auto symmetric = memory.allocate_zeroed(maximumTotal, bench::sharedMemoryAlignment, stream);
  auto* received = static_cast<cuda::std::byte*>(symmetric.local);
  const auto seed = bench::broadcastRandomSeed(rank, options.seed);
  bench::reportSeed(rank, seed);
  {
    bench::DeviceBuffer<cuda::std::byte> reference(maximumTotal, stream);
    bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](const size_t bytes) {
      const size_t total = bench::checkedMultiply(bytes, world);
      auto* source = received + rank * bytes;
      bench::fillRandomBytes(source, bytes, bench::gatherSeed(seed, rank), stream);
      for (int peer = 0; peer < world; ++peer) {
        bench::fillRandomBytes(reference.get() + peer * bytes, bytes, bench::gatherSeed(seed, peer), stream);
      }
      const auto operation = [&] {
        nvshmemx_sync_all_on_stream(stream);
        for (int offset = 1; offset < world; ++offset) {
          const int peer = (rank + offset) % world;
          auto* destination = static_cast<cuda::std::byte*>(symmetric.peers[peer]) + rank * bytes;
          CHECK_CUDA(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
        }
        nvshmemx_sync_all_on_stream(stream);
      };
      operation();
      const double errorPercentage = bench::maxErrorPercentage(
        bench::matxByteMismatches(received, reference.get(), total, stream), total);
      const double milliseconds = bench::measureOperation(stream, MPI_COMM_WORLD, options, operation);
      if (rank == 0) {
        printf("%d, %zu, %zu, %lf, %d, %d, %d, %lf, %lf\n",
          world, bytes, total, errorPercentage,
          bench::effectiveWarmup(options.graphLaunches, options.runs, options.warmup),
          options.runs, options.graphLaunches, milliseconds, bench::bandwidth(total, milliseconds));
      }
    });
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  memory.deallocate(symmetric);
  CHECK_CUDA(cudaStreamDestroy(stream));
  nvshmem_finalize();
}

// Preserve the copy-engine gather's timing argument order.
// ./ce_ag [minBytes] [maxBytes] [warmup] [runs] [graphLaunches]
int main(int argc, char** argv) {
  try {
    bench::Options options{.graphLaunches = 2, .runs = 256};
    if (argc > 1) options.minBytes = bench::parseSize(argv[1]);
    if (argc > 2) options.maxBytes = bench::parseSize(argv[2]);
    if (argc > 3) options.warmup = std::stoi(argv[3]);
    if (argc > 4) options.runs = std::stoi(argv[4]);
    if (argc > 5) options.graphLaunches = std::stoi(argv[5]);
    if (argc > 6) throw std::invalid_argument("Usage: ce_ag [minBytes] [maxBytes] [warmup] [runs] [graphLaunches]");
    bench::validateOptions(options);
    agHost(options);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
