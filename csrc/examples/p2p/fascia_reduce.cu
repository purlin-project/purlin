#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/kernel.cuh>
#include <purlin/benchmark/matx_validation.cuh>
#include <purlin/core.cuh>

namespace {
using Byte = cuda::std::byte;
using Element = __nv_bfloat16;
constexpr int threads = 256;
constexpr int alignment = 16;
constexpr int sourceCount = 8;
constexpr int unrollFactor = 2;
constexpr int worldUnroll = 8;
constexpr int trials = 4;
using Config = purlin::Configuration<threads, alignment, purlin::UNUSED,
  purlin::UNUSED, unrollFactor, worldUnroll, alignment>;
using Fascia = purlin::Atom<700, Config>;

struct Args {
  Byte* src;
  Byte* dst;
  size_t bytesPerBuffer;
  cuda::fast_mod_div<long int> blocks;
};

__launch_bounds__(threads, 1)
__global__ void reduceKernel(const __grid_constant__ Args args) {
  const auto [bytes, offset] = purlin::partition<alignment>(
    args.bytesPerBuffer, args.blocks, blockIdx.x);
  __shared__ Byte* sources[sourceCount];
  if (threadIdx.x < sourceCount) {
    sources[threadIdx.x] = args.src + threadIdx.x * args.bytesPerBuffer + offset;
  }
  __syncthreads();
  const purlin::ReduceTRArgs reduceArgs{
    .sources = sources, .dst = args.dst + offset, .bytesRed = bytes,
    .world = cuda::fast_mod_div<int, true>{sourceCount}};
  Element* workspace = nullptr;
  Fascia::reduce<purlin::ReduceResult::unicast>(reduceArgs, workspace);
}

void benchmark(const bench::KernelOptions& options, cudaStream_t stream) {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
  const int maxBlocks = options.maxBlocks == -1 ? 4 * properties.multiProcessorCount : options.maxBlocks;
  if (maxBlocks <= 0) throw std::invalid_argument("maxBlocks must be positive");

  cudaFuncAttributes attributes{};
  CHECK_CUDA(cudaFuncGetAttributes(&attributes, reduceKernel));
  std::fprintf(stderr, "%s, BF16 sum, %d sources, 1 destination, %d threads, UNROLL_FACTOR=%d, WORLD_UNROLL=%d, "
    "%d %s trials\n", properties.name, sourceCount, threads, unrollFactor, worldUnroll,
    trials, bench::mode(options.graphLaunches));
  std::fprintf(stderr, "Registers/thread: %d. Local bytes/thread: %zu.\n",
    attributes.numRegs, attributes.localSizeBytes);

  // Rotate destination slices past L2. Each slice has its own eight input buffers.
  const auto workingSetBytes = 2 * static_cast<size_t>(properties.l2CacheSize);
  const auto arenaBytes = std::max(options.maxBytes, workingSetBytes);
  const auto arenaElements = arenaBytes / sizeof(Element);
  bench::DeviceBuffer<Element> source(bench::checkedMultiply(arenaElements, sourceCount), stream);
  bench::DeviceBuffer<Element> destination(arenaElements, stream);
  bench::DeviceBuffer<Element> reference(options.maxBytes / sizeof(Element), stream);
  const auto seed = bench::broadcastRandomSeed(0, options.seed);
  bench::reportSeed(0, seed);
  bench::fillRandomReduction(source.get(), source.size(), seed, stream);

  std::puts("bytes_per_buffer,sources,destinations,traffic_bytes,blocks,runs,us,effective_GBps,registers_per_thread");
  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](size_t bytes) {
    const auto elements = bytes / sizeof(Element);
    const auto sourceBytes = bench::checkedMultiply(bytes, sourceCount);
    const auto trafficBytes = bench::checkedMultiply(bytes, sourceCount + 1);
    const auto slots = arenaBytes / bytes;
    const auto blocks = static_cast<int>(std::min(cuda::ceil_div(bytes, size_t{threads * alignment}),
      static_cast<size_t>(maxBlocks)));
    const cuda::fast_mod_div<long int> blockCount{blocks};
    auto timing = options;
    timing.runs = std::max(options.runs, static_cast<int>(std::min(slots, cuda::ceil_div(workingSetBytes, bytes))));

    const auto launch = [&](size_t slot) {
      reduceKernel<<<blocks, threads, 0, stream>>>(Args{
        .src = reinterpret_cast<Byte*>(source.get()) + slot * sourceBytes,
        .dst = reinterpret_cast<Byte*>(destination.get()) + slot * bytes,
        .bytesPerBuffer = bytes, .blocks = blockCount});
    };
    bench::computeReductionReference(source.get(), reference.get(), elements, sourceCount, stream);
    CHECK_CUDA(cudaMemsetAsync(destination.get(), 0, bytes, stream));
    launch(0);
    CHECK_CUDA(cudaGetLastError());
    if (bench::matxByteMismatches(destination.get(), reference.get(), bytes, stream) != 0) {
      throw std::runtime_error("Fascia reduction failed validation at " + std::to_string(bytes) + " bytes per buffer");
    }

    std::array<double, trials> times{};
    for (auto& time : times) {
      size_t slot = 0;
      time = bench::measureOperation(stream, MPI_COMM_SELF, timing, [&] {
        launch(slot);
        slot = (slot + 1) % slots;
      });
    }
    std::sort(times.begin(), times.end());
    const auto milliseconds = (times[trials / 2 - 1] + times[trials / 2]) / 2;
    std::printf("%zu,%d,1,%zu,%d,%d,%.6f,%.6f,%d\n", bytes, sourceCount, trafficBytes, blocks,
      timing.runs, milliseconds * 1000, bench::bandwidth(trafficBytes, milliseconds), attributes.numRegs);
    std::fflush(stdout);
  });
}
}

// fascia_reduce [minBytes] [maxBytes] [maxBlocks] [graphLaunches] [runs] [warmup] [seed]
int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));
  try {
    auto options = bench::parseKernelOptions(argc, argv);
    if (argc == 1) options.minBytes = 1024;
    if (options.minBytes % alignment != 0) throw std::invalid_argument("Sizes must be 16-byte aligned");
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    benchmark(options, stream);
    CHECK_CUDA(cudaStreamDestroy(stream));
    MPI_CHECK(MPI_Finalize());
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
