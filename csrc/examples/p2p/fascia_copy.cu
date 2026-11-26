#include <algorithm>
#include <array>
#include <cmath>
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
constexpr int threads = 256;
constexpr int alignment = 16;
constexpr int unrollFactor = 8;
constexpr int worldUnroll = 8;
constexpr int bufferPairs = 1;
constexpr int trials = 4;
constexpr size_t KiB = 1024;
constexpr size_t MiB = KiB * KiB;
using Config = purlin::Configuration<threads, alignment, purlin::UNUSED,
  purlin::UNUSED, unrollFactor, worldUnroll, alignment>;
using Fascia = purlin::Atom<700, Config>;

struct CopyArgs {
  Byte* src;
  Byte* dst;
  size_t bytesPerBuffer;
  cuda::fast_mod_div<long int> blocks;
};

// Keep the original raw-pointer copy as a benchmark baseline.
__device__ __forceinline__
void legacyCopy(Byte* __restrict__ const& dst, const Byte* __restrict__ const& src,
  const size_t& bytes) {
  using Element = purlin::AlignedType<Config::ALIGNMENT_BYTES>::type;
  constexpr int vectorWidth = Config::GMEM_ACCESS_ALIGNMENT_BYTES / sizeof(Element);
  using VT = purlin::AlignedArray<Element, vectorWidth>;
  using IndexT = uint32_t;
  const auto tIdx = threadIdx.x;
  const auto vP = static_cast<IndexT>(bytes / Config::GMEM_ACCESS_ALIGNMENT_BYTES);
  auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
  const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
  const auto threadElems = vP / Config::THREADS;
  const auto trips = threadElems / Config::UNROLL_FACTOR;
  for (int i = 0; i < trips; ++i) {
    VT values[Config::UNROLL_FACTOR];
    IndexT indices[Config::UNROLL_FACTOR];
    purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
      indices[j] = (i * Config::UNROLL_FACTOR + j) * Config::THREADS + tIdx;
    });
    purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
      values[j] = vS[indices[j]];
    });
    purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
      purlin::store(vD + indices[j], values[j]);
    });
  }
  const auto cutoff = trips * Config::UNROLL_FACTOR * Config::THREADS;
  if (vP > cutoff) {
    const auto residue = vP - cutoff;
    vS += cutoff;
    vD += cutoff;
    for (int i = static_cast<int>(tIdx); i < residue; i += Config::THREADS) {
      purlin::store(vD + i, vS[i]);
    }
  }
}

template<bool general>
__launch_bounds__(threads, 1)
__global__ void copyKernel(const __grid_constant__ CopyArgs args) {
  const auto [bytes, offset] = purlin::partition<alignment>(
    args.bytesPerBuffer, args.blocks, blockIdx.x);
  if constexpr (general && bufferPairs == 1) {
    const auto* source = args.src + offset;
    Fascia::copy(args.dst + offset, source, bytes, nullptr);
  } else if constexpr (general) {
    const purlin::SingleCopyArgs src{
      .address = args.src + offset, .stride = args.bytesPerBuffer,
      .n = cuda::fast_mod_div<int>{bufferPairs}};
    const purlin::SingleCopyArgs dst{
      .address = args.dst + offset, .stride = args.bytesPerBuffer,
      .n = cuda::fast_mod_div<int>{bufferPairs}};
    Fascia::copy(dst, src, bytes, nullptr);
  } else {
    #pragma unroll
    for (int buffer = 0; buffer < bufferPairs; ++buffer) {
      const auto bufferOffset = buffer * args.bytesPerBuffer + offset;
      legacyCopy(args.dst + bufferOffset, args.src + bufferOffset, bytes);
    }
  }
}

struct Geomean {
  double logSum = 0;
  int count = 0;

  void add(double speedup) {
    logSum += std::log(speedup);
    ++count;
  }

  void report(const char* label) const {
    if (count > 0) std::fprintf(stderr, "%s %.4fx (%d sizes)\n", label, std::exp(logSum / count), count);
  }
};

double median(std::array<double, trials> samples) {
  std::sort(samples.begin(), samples.end());
  return (samples[trials / 2 - 1] + samples[trials / 2]) / 2;
}

void benchmark(const bench::KernelOptions& options, cudaStream_t stream) {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
  const int maxBlocks = options.maxBlocks == -1 ? 4 * properties.multiProcessorCount : options.maxBlocks;
  if (maxBlocks <= 0) throw std::invalid_argument("maxBlocks must be positive");

  cudaFuncAttributes oldAttributes{};
  cudaFuncAttributes generalAttributes{};
  CHECK_CUDA(cudaFuncGetAttributes(&oldAttributes, copyKernel<false>));
  CHECK_CUDA(cudaFuncGetAttributes(&generalAttributes, copyKernel<true>));
  std::fprintf(stderr, "%s, n=m=%d, %d threads, UNROLL_FACTOR=%d, WORLD_UNROLL=%d, %d %s trials\n",
    properties.name, bufferPairs, threads, unrollFactor, worldUnroll, trials, bench::mode(options.graphLaunches));
  std::fprintf(stderr, "Registers/thread: old=%d, general=%d. Local bytes/thread: old=%zu, general=%zu.\n",
    oldAttributes.numRegs, generalAttributes.numRegs, oldAttributes.localSizeBytes, generalAttributes.localSizeBytes);

  // Rotate through distinct local buffers on successive launches to exceed L2.
  const auto workingSetBytes = 2 * static_cast<size_t>(properties.l2CacheSize);
  const auto arenaBytes = std::max(bench::checkedMultiply(options.maxBytes, bufferPairs), workingSetBytes);
  bench::DeviceBuffer<Byte> source(arenaBytes, stream);
  bench::DeviceBuffer<Byte> destination(arenaBytes, stream);
  const auto seed = bench::broadcastRandomSeed(0, options.seed);
  bench::reportSeed(0, seed);
  bench::fillRandomBytes(source.get(), arenaBytes, seed, stream);

  Geomean latency;
  Geomean bandwidth;
  std::puts("bytes_per_buffer,pairs,traffic_bytes,blocks,runs,old_us,general_us,old_effective_GBps,general_effective_GBps,speedup");
  bench::forEachPowerOfTwoSize(options.minBytes, options.maxBytes, [&](size_t bytes) {
    const auto payload = bench::checkedMultiply(bytes, bufferPairs);
    const auto trafficBytes = bench::checkedMultiply(payload, 2);
    const auto slots = arenaBytes / payload;
    const auto blocks = static_cast<int>(std::min(cuda::ceil_div(bytes, size_t{threads * alignment}),
      static_cast<size_t>(maxBlocks)));
    const cuda::fast_mod_div<long int> blockCount{blocks};
    auto timing = options;
    timing.runs = std::max(options.runs, static_cast<int>(std::min(slots, cuda::ceil_div(workingSetBytes, payload))));

    const auto launch = [&]<bool general>(size_t slot) {
      const auto offset = slot * payload;
      copyKernel<general><<<blocks, threads, 0, stream>>>(CopyArgs{
        .src = source.get() + offset, .dst = destination.get() + offset,
        .bytesPerBuffer = bytes, .blocks = blockCount});
    };
    const auto validate = [&]<bool general>() {
      CHECK_CUDA(cudaMemsetAsync(destination.get(), 0, payload, stream));
      launch.template operator()<general>(0);
      CHECK_CUDA(cudaGetLastError());
      if (bench::matxByteMismatches(destination.get(), source.get(), payload, stream) != 0) {
        throw std::runtime_error(std::string(general ? "General" : "Old") +
          " Fascia copy failed validation at " + std::to_string(bytes) + " bytes per buffer");
      }
    };
    validate.template operator()<false>();
    validate.template operator()<true>();

    const auto measure = [&]<bool general>() {
      size_t slot = 0;
      return bench::measureOperation(stream, MPI_COMM_SELF, timing, [&] {
        launch.template operator()<general>(slot);
        slot = (slot + 1) % slots;
      });
    };
    std::array<double, trials> oldTimes{};
    std::array<double, trials> generalTimes{};
    // Alternate which version runs first to balance warmup and clock changes.
    for (int trial = 0; trial < trials; ++trial) {
      if (trial % 2 == 0) {
        oldTimes[trial] = measure.template operator()<false>();
        generalTimes[trial] = measure.template operator()<true>();
      } else {
        generalTimes[trial] = measure.template operator()<true>();
        oldTimes[trial] = measure.template operator()<false>();
      }
    }

    const double oldMs = median(oldTimes);
    const double generalMs = median(generalTimes);
    const double oldGBps = bench::bandwidth(trafficBytes, oldMs);
    const double generalGBps = bench::bandwidth(trafficBytes, generalMs);
    const double speedup = oldMs / generalMs;
    if (bytes >= KiB && bytes <= 512 * KiB) latency.add(speedup);
    if (bytes >= MiB && bytes <= 128 * MiB) bandwidth.add(generalGBps / oldGBps);
    std::printf("%zu,%d,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", bytes, bufferPairs, trafficBytes, blocks,
      timing.runs, oldMs * 1000, generalMs * 1000, oldGBps, generalGBps, speedup);
    std::fflush(stdout);
  });
  latency.report("Latency geomean speedup (1 KiB to 512 KiB)");
  bandwidth.report("Effective bandwidth geomean speedup (1 MiB to 128 MiB)");
}
}

// fascia_copy [minBytes] [maxBytes] [maxBlocks] [graphLaunches] [runs] [warmup] [seed]
int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));
  try {
    auto options = bench::parseKernelOptions(argc, argv);
    if (argc == 1) options.minBytes = KiB;
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
