#ifndef PURLIN_SUPPORT_BENCHMARK_DATA_CUH
#define PURLIN_SUPPORT_BENCHMARK_DATA_CUH

#include <cstddef>
#include <cstdint>
#include <random>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/std/type_traits>

#include "checks.cuh"

namespace bench {

__host__ __device__ inline uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

template<typename Element, typename Value>
__device__ inline Element reductionConvert(const Value value) {
  if constexpr (cuda::std::is_same_v<Element, __half>) {
    return __float2half_rn(static_cast<float>(value));
  } else if constexpr (cuda::std::is_same_v<Element, __nv_bfloat16>) {
    return __float2bfloat16_rn(static_cast<float>(value));
  } else {
    return static_cast<Element>(value);
  }
}

template<typename Accumulator, typename Element>
__device__ inline Accumulator reductionLoad(const Element value) {
  if constexpr (cuda::std::is_same_v<Element, __half>) {
    return static_cast<Accumulator>(__half2float(value));
  } else if constexpr (cuda::std::is_same_v<Element, __nv_bfloat16>) {
    return static_cast<Accumulator>(__bfloat162float(value));
  } else {
    return static_cast<Accumulator>(value);
  }
}

// Every rank's contribution to a gather collective is its own seeded stream;
// receivers replay peer streams locally to build reference outputs without
// any communication.
__host__ __device__ inline uint32_t gatherSeed(const uint32_t seed, const int sourceRank) {
  return static_cast<uint32_t>(splitmix64((static_cast<uint64_t>(seed) << 32) |
    static_cast<uint64_t>(static_cast<uint32_t>(sourceRank))));
}

// All-to-all chunks are seeded per (source, destination) pair so a receiver
// can replay exactly the chunk addressed to it, independent of peer layouts.
__host__ __device__ inline uint32_t pairSeed(const uint32_t seed,
  const int sourceRank, const int destinationRank) {
  return static_cast<uint32_t>(splitmix64((static_cast<uint64_t>(seed) << 32) |
    (static_cast<uint64_t>(static_cast<uint32_t>(sourceRank)) << 16) |
    static_cast<uint64_t>(static_cast<uint32_t>(destinationRank))));
}

__host__ __device__ inline uint32_t allReduceSeed(
  const uint32_t seed, const int sourceRank) {
  return seed + static_cast<uint32_t>(sourceRank) * 42u;
}

__host__ __device__ inline uint32_t reduceScatterSeed(
  const uint32_t seed, const int sourceRank, const int destinationRank) {
  return (static_cast<uint32_t>(sourceRank) + 1u) *
    (seed + static_cast<uint32_t>(destinationRank) * 42u);
}

__host__ __device__ inline uint32_t randomReductionBits(const uint32_t seed, const size_t index) {
  return static_cast<uint32_t>(splitmix64(
    static_cast<uint64_t>(index) ^ (static_cast<uint64_t>(seed) << 32)) >> 32);
}

__global__ inline void fillRandomBytesKernel(std::byte* destination, const size_t count,
  const uint32_t streamSeed) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    reinterpret_cast<unsigned char*>(destination)[index] =
      static_cast<unsigned char>(randomReductionBits(streamSeed, index) & 0xffu);
  }
}

template<typename Element>
__device__ inline Element randomReductionValue(const uint32_t seed, const size_t index) {
  constexpr float inverseRange = 1.0f / 16777216.0f;
  const float unit = static_cast<float>(randomReductionBits(seed, index) >> 8) * inverseRange;
  return reductionConvert<Element>(unit * 2.0f - 1.0f);
}

template<typename Element>
__global__ void fillRandomReductionKernel(Element* destination, const size_t count,
  const uint32_t seed) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) destination[index] = randomReductionValue<Element>(seed, index);
}

template<typename Element>
__global__ void computeReductionReferenceKernel(const Element* sources,
  Element* reference, const size_t count, const int world) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  using Accumulator = cuda::std::common_type_t<Element, float>;
  Accumulator accumulator = static_cast<Accumulator>(0.0f);
  // Match Purlin's deterministic rank order.
  for (int source = 0; source < world; ++source) {
    accumulator += reductionLoad<Accumulator>(sources[source * count + index]);
  }
  reference[index] = reductionConvert<Element>(accumulator);
}

template<typename Element>
__global__ void fillRandomAllReduceReferenceSourcesKernel(Element* sources,
  const size_t count, const uint32_t seed, const int world) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = count * static_cast<size_t>(world);
  if (index >= total) return;
  const int source = static_cast<int>(index / count);
  sources[index] = randomReductionValue<Element>(allReduceSeed(seed, source), index % count);
}

template<typename Element>
__global__ void fillRandomReduceScatterReferenceSourcesKernel(Element* sources,
  const size_t count, const uint32_t seed, const int world, const int destinationRank) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = count * static_cast<size_t>(world);
  if (index >= total) return;
  const int source = static_cast<int>(index / count);
  sources[index] = randomReductionValue<Element>(
    reduceScatterSeed(seed, source, destinationRank), index % count);
}

inline dim3 validationBlocks(const size_t count) {
  constexpr unsigned threads = 256;
  return dim3(static_cast<unsigned>((count + threads - 1) / threads));
}

template<typename Byte>
inline void fillRandomBytes(Byte* destination, const size_t count,
  const uint32_t streamSeed, cudaStream_t stream) {
  static_assert(sizeof(Byte) == 1);
  if (count == 0) return;
  fillRandomBytesKernel<<<validationBlocks(count), 256, 0, stream>>>(
    reinterpret_cast<std::byte*>(destination), count, streamSeed);
  CHECK_CUDA(cudaGetLastError());
}

// A requested seed of zero draws a fresh one, so unseeded runs stay random
// while a replay can pin the exact data of a previous run.
inline uint32_t broadcastRandomSeed(const int rank, const uint32_t requested = 0) {
  uint32_t seed = requested;
  if (requested == 0 && rank == 0) {
    do { seed = std::random_device{}(); } while (seed == 0);
  }
  MPI_CHECK(MPI_Bcast(&seed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
  return seed;
}

inline void reportSeed(const int rank, const uint32_t seed) {
  // stderr keeps the CSV on stdout machine-readable.
  if (rank == 0) std::fprintf(stderr, "data seed: %u\n", seed);
}

template<typename Element>
inline void fillRandomReduction(Element* destination, const size_t count,
  const uint32_t seed, cudaStream_t stream) {
  if (count == 0) return;
  fillRandomReductionKernel<Element><<<validationBlocks(count), 256, 0, stream>>>(
    destination, count, seed);
  CHECK_CUDA(cudaGetLastError());
}

template<typename Element>
inline void fillRandomAllReduceReferenceSources(Element* sources,
  const size_t count, const uint32_t seed, const int world, cudaStream_t stream) {
  const size_t total = count * static_cast<size_t>(world);
  if (total == 0) return;
  fillRandomAllReduceReferenceSourcesKernel<Element>
    <<<validationBlocks(total), 256, 0, stream>>>(sources, count, seed, world);
  CHECK_CUDA(cudaGetLastError());
}

template<typename Element>
inline void fillRandomReduceScatterReferenceSources(Element* sources,
  const size_t count, const uint32_t seed, const int world,
  const int destinationRank, cudaStream_t stream) {
  const size_t total = count * static_cast<size_t>(world);
  if (total == 0) return;
  fillRandomReduceScatterReferenceSourcesKernel<Element>
    <<<validationBlocks(total), 256, 0, stream>>>(
      sources, count, seed, world, destinationRank);
  CHECK_CUDA(cudaGetLastError());
}

template<typename Element>
inline void computeReductionReference(const Element* sources, Element* reference,
  const size_t count, const int world, cudaStream_t stream) {
  if (count == 0) return;
  computeReductionReferenceKernel<Element><<<validationBlocks(count), 256, 0, stream>>>(
    sources, reference, count, world);
  CHECK_CUDA(cudaGetLastError());
}

template<typename Element>
consteval const char* dataTypeName() {
  if constexpr (cuda::std::is_same_v<Element, float>) return "fp32";
  else if constexpr (cuda::std::is_same_v<Element, double>) return "fp64";
  else if constexpr (cuda::std::is_same_v<Element, __half>) return "fp16";
  else if constexpr (cuda::std::is_same_v<Element, __nv_bfloat16>) return "bf16";
  else if constexpr (cuda::std::is_same_v<Element, __nv_fp8_e4m3>) return "fp8_e4m3";
  else if constexpr (cuda::std::is_same_v<Element, __nv_fp8_e5m2>) return "fp8_e5m2";
  else {
    static_assert(cuda::std::is_same_v<Element, void>, "Unsupported reduction data type");
    return "unsupported";
  }
}

} // namespace bench

#endif
