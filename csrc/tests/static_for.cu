#include <cstdio>
#include <stdexcept>

#include <purlin/static_for.cuh>

namespace {
__host__ __device__ constexpr bool emptyDoesNotInstantiate() {
  purlin::static_for<0>([](auto i) {
    static_assert(decltype(i)::value < 0, "An empty loop must not instantiate its body");
  });
  return true;
}
static_assert(emptyDoesNotInstantiate());

struct Visits {
  int count = 0;
  bool ordered = true;

  __host__ __device__ constexpr Visits() = default;
  Visits(const Visits&) = delete;

  template<typename Index>
  __host__ __device__ __forceinline__
  constexpr void operator()(Index) noexcept {
    ordered = ordered && count == Index::value;
    ++count;
  }
};

__host__ __device__ constexpr bool referenceAndIndexChecks() {
  Visits visits;
  purlin::static_for<4>(visits);
  int count = 0;
  purlin::static_for<size_t{2}>([&](auto i) {
    static_assert(cuda::std::is_same_v<typename decltype(i)::value_type, size_t>);
    purlin::static_for<3>([&](auto) { ++count; });
  });
  return visits.count == 4 && visits.ordered && count == 6;
}
static_assert(referenceAndIndexChecks());
static_assert(noexcept(purlin::static_for<4>(cuda::std::declval<Visits&>())));

__host__ __device__ constexpr int sequence(int seed) {
  purlin::static_for<4>([&](auto i) { seed = seed * 10 + decltype(i)::value; });
  return seed;
}
static_assert(sequence(7) == 70123);

__global__ void checkDevice(int* result, int seed) {
  *result = referenceAndIndexChecks() && emptyDoesNotInstantiate() ? sequence(seed) : -1;
}

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

int main() {
  try {
    if (!referenceAndIndexChecks() || sequence(7) != 70123) {
      throw std::runtime_error("static_for host checks failed");
    }
    int* device = nullptr;
    check(cudaMalloc(&device, sizeof(int)));
    checkDevice<<<1, 1>>>(device, 7);
    check(cudaGetLastError());
    int result = 0;
    check(cudaMemcpy(&result, device, sizeof(int), cudaMemcpyDeviceToHost));
    check(cudaFree(device));
    if (result != 70123) throw std::runtime_error("static_for device checks failed");
    std::puts("static_for host and device checks passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
