//
// Created by osayamen on 6/1/26.
//

#ifndef PURLIN_UTIL_CUH
#define PURLIN_UTIL_CUH
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <curanddx.hpp>

#include <purlin/context.cuh>
#include <purlin/constants.cuh>
#include <purlin/math.cuh>

#define NCCL_CHECK(call)                                   \
    do                                                     \
    {                                                      \
        ncclResult_t status = call;                        \
        if (status != ncclSuccess)                         \
        {                                                  \
            fprintf(stderr, "NCCL error at %s:%d : %d\n",  \
            __FILE__, __LINE__, status);                   \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

#if !defined(CHECK_CUDA)
#  define CHECK_CUDA(e)                                      \
do {                                                         \
    cudaError_t code = (e);                                  \
    if (code != cudaSuccess) {                               \
        fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",         \
            __FILE__, __LINE__, #e,                          \
            cudaGetErrorName(code),                          \
            cudaGetErrorString(code));                       \
        fflush(stderr);                                      \
        exit(1);                                             \
    }                                                        \
} while (0);
#endif

static constexpr int SAMPLE_SMEM_ALIGNMENT = 128;
template<typename T>
__host__ __forceinline__
auto splitPointerTable(T** const& base, const size_t& offset, const int& world, cudaStream_t stream) {
  void* mem = nullptr;
  std::vector<T*> p(world);
  std::vector<T*> q(world);
  const auto pB = sizeof(typename decltype(p)::value_type) * p.size();
  CHECK_CUDA(cudaMallocAsync(&mem, pB, stream));
  CHECK_CUDA(cudaMemcpyAsync(p.data(), base, pB, cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  for (int i = 0; i < world; ++i) {
    q[i] = p[i] + offset;
  }
  CHECK_CUDA(cudaMemcpyAsync(mem, q.data(), pB, cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return static_cast<T**>(mem);
}

template <typename T>
__host__ __forceinline__
auto allocateSymMem(const int& world, const size_t& elems, cudaStream_t stream) {
  static_assert(!cuda::std::is_void_v<T>);
  if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
    throw std::runtime_error("nvshmem is not initialized");
  }
  const auto* base = static_cast<T*>(nvshmem_calloc(elems, sizeof(T)));
  void* mem = nullptr;
  std::vector<T*> p(world);
  const auto pB = sizeof(typename decltype(p)::value_type) * p.size();
  CHECK_CUDA(cudaMallocAsync(&mem, pB, stream));
  for (int i = 0; i < world; ++i) {
    p[i] = static_cast<T*>(nvshmem_ptr(base, i));
  }
  CHECK_CUDA(cudaMemcpyAsync(mem, p.data(), pB, cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return static_cast<T**>(mem);
}

template <typename T>
__host__ __forceinline__
void freeSymMem(T** const& p, const int& rank, cudaStream_t stream) {
  static_assert(!cuda::std::is_void_v<T>);
  T* localPtr = nullptr;
  CHECK_CUDA(cudaMemcpyAsync(&localPtr, p + rank, sizeof(T*), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvshmem_free(localPtr);
  CHECK_CUDA(cudaFreeAsync(p, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
}

__host__ __forceinline__
auto makeWorkspace(const int& world, cudaStream_t stream) {
  const auto size = 2 * (purlin::STAGING_BUFFER_SIZE_ + world * purlin::PACKET_BUFFER_SIZE);
  auto base = allocateSymMem<cuda::std::byte>(world, size, stream);
  auto sigBase = allocateSymMem<uint64_t>(world, 2 * world, stream);
  auto varLenBase = allocateSymMem<purlin::LRP16Raw>(world, 2 * world, stream);
  auto varOffsetBase = allocateSymMem<purlin::LRP16Raw>(world, 2 * world, stream);
  return purlin::WorkspaceMemory{
    .stagingLR = splitPointerTable(base, 2 * purlin::STAGING_BUFFER_SIZE_, world, stream),
    .stagingTR = base,
    .signals = sigBase,
    .gatherSignals = splitPointerTable(sigBase, world, world, stream),
    .varLenSignals = varLenBase,
    .varOffsetSignals = varOffsetBase
  };
}

__host__ __forceinline__
auto destroyWorkspace(const purlin::WorkspaceMemory& w, const int& rank, cudaStream_t stream) {
  freeSymMem(w.stagingTR, rank, stream);
  freeSymMem(w.signals, rank, stream);
  freeSymMem(w.varLenSignals, rank, stream);
  freeSymMem(w.varOffsetSignals, rank, stream);
  CHECK_CUDA(cudaFreeAsync(w.stagingLR, stream));
  CHECK_CUDA(cudaFreeAsync(w.gatherSignals, stream));
}

struct Times {
  double t_ms;
  double ep;
};

template<typename T, typename S>
struct Converter {
  __device__ auto operator()(const S &x) const {
    return static_cast<T>(x);
  }
};

template<>
struct Converter<float, __half> {
  __device__ auto operator()(const __half &x) const {
    return __half2float(x);
  }
};

template<>
struct Converter<__half, float> {
  __device__ auto operator()(const float &x) const {
    return __float2half(x);
  }
};

template<>
struct Converter<__nv_bfloat16, float> {
  __device__ auto operator()(const float &x) const {
    return __float2bfloat16(x);
  }
};

template<>
struct Converter<matx::matxFp16, float> {
  __device__ auto operator()(const float &x) const {
    return __float2half(x);
  }
};

template<>
struct Converter<matx::matxBf16, float> {
  __device__ auto operator()(const float &x) const {
    return __float2bfloat16(x);
  }
};

template<>
struct Converter<float, matx::matxFp16> {
  __device__ auto operator()(const matx::matxFp16 &x) const {
    return __half2float(x.x);
  }
};

template<>
struct Converter<float, matx::matxBf16> {
  __device__ auto operator()(const matx::matxBf16 &x) const {
    return __bfloat162float(x.x);
  }
};

template<typename T, int Alignment = 16>
struct VectorTypeDescriptor {
  using VectorWidth = cuda::std::integral_constant<int, Alignment / sizeof(T)>;
  using VectorType = purlin::AlignedArray<T, VectorWidth::value, Alignment>;
};

template<int Arch, bool predicate, typename Element>
__global__ void generateRandUniform(
  Element *__restrict__ out,
  const __grid_constant__ size_t n,
  const __grid_constant__ size_t seed,
  const __grid_constant__ float minv,
  const __grid_constant__ float maxv,
  const __grid_constant__ unsigned long long global_offset = 0ULL
) {
  using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
                       curanddx::SM<Arch>() +
                       curanddx::Thread());
  const auto tid = static_cast<unsigned long long int>(blockIdx.x)
                   * blockDim.x + threadIdx.x;

  constexpr int vF = 4;
  const size_t out_base = static_cast<size_t>(tid) * vF;

  RNG rng(seed, tid, global_offset);

  curanddx::uniform<float> dist(minv, maxv);

  auto v = dist.generate4(rng);

  constexpr Converter<Element, float> storeOp{};

  if constexpr (predicate) {
    if (out_base + (vF - 1) >= n) return;
    // n % 4 == 0
    using VTD = VectorTypeDescriptor<Element, vF * sizeof(Element)>;
    using VT = VTD::VectorType;
    static_assert(VTD::VectorWidth::value == vF);
    auto *__restrict__ vo = reinterpret_cast<VT *>(out);
    VT vt{};
    vt[0] = storeOp(v.x);
    vt[1] = storeOp(v.y);
    vt[2] = storeOp(v.z);
    vt[3] = storeOp(v.w);
    vo[tid] = vt;
  } else {
    if (out_base >= n) return;
    out[out_base + 0] = storeOp(v.x);
    if (out_base + 1 < n) out[out_base + 1] = storeOp(v.y);
    if (out_base + 2 < n) out[out_base + 2] = storeOp(v.z);
    if (out_base + 3 < n) out[out_base + 3] = storeOp(v.w);
  }
}

template<int Arch, typename Element>
__host__ __forceinline__
void randUniform(Element *__restrict__ const&out, const size_t &n, const size_t &seed, const float &minv,
                 const float &maxv, cudaStream_t stream) {
  constexpr uint threads = 1024;
  const auto blocks = static_cast<uint>(cuda::ceil_div(n, threads * 4));
  if (n % 4 == 0) {
    generateRandUniform<Arch, true><<<blocks, threads, 0, stream>>>(out, n, seed, minv, maxv);
  } else {
    generateRandUniform<Arch, false><<<blocks, threads, 0, stream>>>(out, n, seed, minv, maxv);
  }
}

template<typename Element>
consteval const char *element_string() {
  static_assert(
    cuda::std::is_same_v<Element, __half> ||
    cuda::std::is_same_v<Element, __nv_bfloat16> ||
    cuda::std::is_same_v<Element, float> ||
    cuda::std::is_same_v<Element, __nv_fp8_e4m3> ||
    cuda::std::is_same_v<Element, __nv_fp8_e5m2> ||
    "Unsupported Element type"
  );
  if constexpr (cuda::std::is_same_v<Element, float>) return "fp32";
  else if constexpr (cuda::std::is_same_v<Element, __half>) return "fp16";
  else if constexpr (cuda::std::is_same_v<Element, __nv_fp8_e4m3>) return "fp8_E4M3";
  else if constexpr (cuda::std::is_same_v<Element, __nv_fp8_e5m2>) return "fp8_E5M2";
  else return "bf16";
}

// Parse sizes like 4096, 4K, 16M, 1G
__host__ __forceinline__
size_t parseSize(const std::string &s) {
  char unit = 0;
  double val = 0.0;
  if (sscanf(s.c_str(), "%lf%c", &val, &unit) >= 1) {
    size_t mult = 1;
    switch (unit) {
      case 'k':
      case 'K': mult = 1024ull;
        break;
      case 'm':
      case 'M': mult = 1024ull * 1024ull;
        break;
      case 'g':
      case 'G': mult = 1024ull * 1024ull * 1024ull;
        break;
      default: mult = 1;
        break;
    }
    if (unit == 0 || (unit != 'K' && unit != 'k' && unit != 'M' && unit != 'm' && unit != 'G' && unit != 'g')) {
      // no unit, already parsed in val
      return static_cast<size_t>(val);
    }
    return static_cast<size_t>(val * static_cast<double>(mult));
  }
  fprintf(stderr, "Invalid Size\n");
  std::exit(EXIT_FAILURE);
}

struct RunOptions {
  size_t minLocalBytes = 128;
  size_t maxLocalBytes = 128 * 1024 * 1024;
  int warmup = 128;
  int runs = 128;
  int graph_launches = 8;
  int maxSuperBlockSize = 32; // # of blocks in a superblock
  int maxReduceBlocks = 32;
  float rtol = 2e-2;
  float atol = 2e-3;
};

template<int nArch, int threshold>
constexpr auto getSBZ(const int &world, const size_t &maxBytes) {
  // A100
  if (world >= 8) {
    return 8;
  }
  if (maxBytes >= threshold) {
    if constexpr (nArch <= 800) {
      return 32;
    }
    return 64;
  }
  return 16;
}

template<typename Element>
using MXE = cuda::std::conditional_t<cuda::std::is_same_v<Element, __half>, matx::matxFp16,
  cuda::std::conditional_t<cuda::std::is_same_v<Element, __nv_bfloat16>, matx::matxBf16, Element> >;
#endif //PURLIN_UTIL_CUH
