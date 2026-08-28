#ifndef PURLIN_TESTS_COMMON_PURLIN_RUNTIME_CUH
#define PURLIN_TESTS_COMMON_PURLIN_RUNTIME_CUH

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvshmem.h>

#include "checks.cuh"

#include <purlin/core.cuh>

#include "benchmark.cuh"
#include "variable_counts.cuh"

namespace bench {

template<typename T>
inline T** allocateSymmetricPointerTable(const int world, const size_t elements,
  cudaStream_t stream, T** localOut = nullptr) {
  T* local = static_cast<T*>(nvshmem_calloc(elements, sizeof(T)));
  if (local == nullptr) throw std::bad_alloc();
  if (localOut != nullptr) *localOut = local;

  std::vector<T*> pointers(world);
  for (int rank = 0; rank < world; ++rank) pointers[rank] = static_cast<T*>(nvshmem_ptr(local, rank));

  T** devicePointers = nullptr;
  CHECK_CUDA(cudaMallocAsync(&devicePointers, sizeof(T*) * world, stream));
  CHECK_CUDA(cudaMemcpyAsync(devicePointers, pointers.data(), sizeof(T*) * world,
    cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return devicePointers;
}

template<typename T>
inline T** offsetPointerTable(T** base, const size_t offset, const int world,
  cudaStream_t stream) {
  std::vector<T*> pointers(world);
  CHECK_CUDA(cudaMemcpyAsync(pointers.data(), base, sizeof(T*) * world,
    cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  for (T*& pointer : pointers) pointer += offset;

  T** result = nullptr;
  CHECK_CUDA(cudaMallocAsync(&result, sizeof(T*) * world, stream));
  CHECK_CUDA(cudaMemcpyAsync(result, pointers.data(), sizeof(T*) * world,
    cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return result;
}

template<typename T>
inline void freeSymmetricPointerTable(T** pointers, const int rank, cudaStream_t stream) {
  if (pointers == nullptr) return;
  T* local = nullptr;
  CHECK_CUDA(cudaMemcpyAsync(&local, pointers + rank, sizeof(T*),
    cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvshmem_free(local);
  CHECK_CUDA(cudaFreeAsync(pointers, stream));
}

inline purlin::WorkspaceMemory makePurlinWorkspace(const int world, cudaStream_t stream) {
  constexpr size_t stagingBytes = purlin::STAGING_BUFFER_SIZE_;
  const size_t bytes = 2 * (stagingBytes +
    static_cast<size_t>(world) * purlin::PACKET_BUFFER_SIZE);
  cuda::std::byte* localStaging = nullptr;
  auto staging = allocateSymmetricPointerTable<cuda::std::byte>(world, bytes, stream, &localStaging);
  auto signals = allocateSymmetricPointerTable<uint64_t>(world, 3 * world, stream);
  auto lengths = allocateSymmetricPointerTable<purlin::LRP>(world, 2 * world, stream);
  // NVLS multicast mapping of the staging slab; null when unsupported or disabled.
  cuda::std::byte* mcStaging = nullptr;
  if (std::getenv("PURLIN_DISABLE_MULTIMEM") == nullptr) {
    mcStaging = static_cast<cuda::std::byte*>(nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, localStaging));
  }
  const auto mcStagingLR = mcStaging != nullptr &&
    std::getenv("PURLIN_DISABLE_MULTIMEM_LR") == nullptr ?
    mcStaging + 2 * stagingBytes : nullptr;
  return {
    .stagingLR = offsetPointerTable(staging, 2 * stagingBytes, world, stream),
    .stagingTR = staging,
    .signals = signals,
    .gatherSignals = offsetPointerTable(signals, world, world, stream),
    .consumedSignals = offsetPointerTable(signals, 2 * world, world, stream),
    .varLenSignals = lengths,
    .mcStagingTR = mcStaging,
    .mcStagingLR = mcStagingLR,
  };
}

inline void destroyPurlinWorkspace(const purlin::WorkspaceMemory& workspace,
  const int rank, cudaStream_t stream) {
  freeSymmetricPointerTable(workspace.stagingTR, rank, stream);
  freeSymmetricPointerTable(workspace.signals, rank, stream);
  freeSymmetricPointerTable(workspace.varLenSignals, rank, stream);
  CHECK_CUDA(cudaFreeAsync(workspace.stagingLR, stream));
  CHECK_CUDA(cudaFreeAsync(workspace.gatherSignals, stream));
  CHECK_CUDA(cudaFreeAsync(workspace.consumedSignals, stream));
}

class PurlinRuntime {
public:
  PurlinRuntime() {
    nvshmem_init();
    nvshmemInitialized_ = true;
    world = nvshmem_n_pes();
    rank = nvshmem_my_pe();
    device = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    if (world < 2) {
      nvshmem_finalize();
      nvshmemInitialized_ = false;
      throw std::runtime_error("Purlin benchmarks require at least two ranks");
    }

    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProperties, device));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    workspace = makePurlinWorkspace(world, stream);
    workspaceInitialized_ = true;
    // Optional staging-size override for experiments; the allocation stays at
    // STAGING_BUFFER_SIZE_, only the context's working size shrinks.
    size_t stagingTRSize = purlin::STAGING_BUFFER_SIZE_;
    if (const char* override_ = std::getenv("PURLIN_STAGING_TR_SIZE")) {
      stagingTRSize = parseSize(override_);
      if (stagingTRSize > purlin::STAGING_BUFFER_SIZE_) {
        throw std::invalid_argument("PURLIN_STAGING_TR_SIZE exceeds the allocated staging slab");
      }
    }
    context = purlin::initialize(rank, world, workspace, stream, stagingTRSize);
    contextInitialized_ = true;
  }

  PurlinRuntime(const PurlinRuntime&) = delete;
  PurlinRuntime& operator=(const PurlinRuntime&) = delete;

  ~PurlinRuntime() {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    if (contextInitialized_) purlin::finalize(context, stream);
    if (workspaceInitialized_) destroyPurlinWorkspace(workspace, rank, stream);
    if (stream != nullptr) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    if (nvshmemInitialized_) nvshmem_finalize();
  }

  int rank = 0;
  int world = 0;
  int device = 0;
  cudaDeviceProp deviceProperties{};
  cudaStream_t stream = nullptr;
  purlin::WorkspaceMemory workspace{};
  purlin::Context context{};

private:
  bool nvshmemInitialized_ = false;
  bool workspaceInitialized_ = false;
  bool contextInitialized_ = false;
};

inline void validatePurlinOptions(const Options& options) {
  if (options.minBytes % purlin::MAX_ACCESS_ALIGNMENT != 0 ||
      options.maxBytes % purlin::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Sizes must be multiples of " +
      std::to_string(purlin::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
}

inline purlin::VState makePurlinVState(const std::vector<size_t>& sizes,
  const std::vector<size_t>& byteOffsets, const int rank) {
  return {
    .maxBytes = maximumBytes(sizes),
    .totalBytes = totalBytes(sizes),
    .offset = byteOffsets[rank],
    .bytes = sizes[rank],
  };
}

inline purlin::VState makePurlinAllToAllVState(const std::vector<size_t>& sends,
  const std::vector<size_t>& receives, const std::vector<size_t>& sendOffsets,
  const int rank) {
  return {
    .maxOutBytes = maximumBytes(receives),
    .maxBytes = maximumBytes(sends),
    .totalBytes = totalBytes(sends),
    .totalOutBytes = totalBytes(receives),
    .offset = sendOffsets[rank],
    .bytes = sends[rank],
  };
}

} // namespace bench

#endif
