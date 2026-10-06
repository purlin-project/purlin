#ifndef PURLIN_SUPPORT_BENCHMARK_PURLIN_RUNTIME_CUH
#define PURLIN_SUPPORT_BENCHMARK_PURLIN_RUNTIME_CUH

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "checks.cuh"

#include <purlin/core.cuh>
#include <contrib/symm_mem.cuh>

#include "benchmark.cuh"
#include "variable_counts.cuh"

namespace bench {

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
    size_t stagingTRSize = purlin::STAGING_BUFFER_SIZE_;
    if (const char* override_ = std::getenv("PURLIN_STAGING_TR_SIZE")) {
      stagingTRSize = parseSize(override_);
    }
    managed_ = purlin::initialize(rank, world, stream, purlin::NvshmemMemory{}, stagingTRSize);
    context = managed_.context();
  }

  PurlinRuntime(const PurlinRuntime&) = delete;
  PurlinRuntime& operator=(const PurlinRuntime&) = delete;

  ~PurlinRuntime() {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    purlin::finalize(managed_, stream);
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
  purlin::Context context{};

private:
  bool nvshmemInitialized_ = false;
  purlin::ManagedContext<purlin::NvshmemMemory> managed_;
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
