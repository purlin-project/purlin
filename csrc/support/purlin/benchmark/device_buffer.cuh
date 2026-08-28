#ifndef PURLIN_SUPPORT_BENCHMARK_DEVICE_BUFFER_CUH
#define PURLIN_SUPPORT_BENCHMARK_DEVICE_BUFFER_CUH

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "checks.cuh"

namespace bench {

template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer(const size_t count, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) throw std::invalid_argument("DeviceBuffer requires a valid CUDA stream");
    if (count == 0) return;
    CHECK_CUDA(cudaMalloc(&pointer_, count * sizeof(T)));
    count_ = count;
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      count_(std::exchange(other.count_, 0)),
      stream_(std::exchange(other.stream_, nullptr)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      pointer_ = std::exchange(other.pointer_, nullptr);
      count_ = std::exchange(other.count_, 0);
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  ~DeviceBuffer() { release(); }

  T* get() const { return pointer_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }
  explicit operator bool() const { return pointer_ != nullptr; }

private:
  void release() {
    if (pointer_ != nullptr) {
      // Destructors cannot report errors, so log the failure and continue cleanup.
      const cudaError_t status = cudaFree(pointer_);
      if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error while freeing a benchmark buffer: %s\n",
          cudaGetErrorString(status));
      }
      pointer_ = nullptr;
      count_ = 0;
    }
    stream_ = nullptr;
  }

  T* pointer_ = nullptr;
  size_t count_ = 0;
  cudaStream_t stream_ = nullptr;
};

} // namespace bench

#endif
