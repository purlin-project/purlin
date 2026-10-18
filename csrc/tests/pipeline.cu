#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <purlin/pipeline.cuh>

namespace {
template<int stages, int threads = 64>
struct Config {
  static constexpr int THREADS = threads;
  static constexpr int ELEMS_PER_THREAD = 2;
  static constexpr int PIPE_STAGES = stages;
};

// Synchronous transfers exercise the scheduler without any cp.async machinery.
template<typename C>
struct SynchronousOperation : purlin::PipelineOperation {
  using Value = uint32_t;
  static constexpr int STAGE_ELEMENTS = C::THREADS * C::ELEMS_PER_THREAD;
  const Value* source;
  Value* destination;
  int totalStages;
  unsigned int id;
  int commits = 0;
  int advances = 0;
  int waits = 0;
  int consumed = 0;
  bool valid = true;

  __device__ SynchronousOperation(const Value* src, Value* dst, int stages, unsigned int threadId) :
    source(src), destination(dst), totalStages(stages), id(threadId) {}

  template<typename Stage>
  __device__ auto prepare(Stage stage) const { return stage; }

  __device__ void advance() { ++advances; }

  template<typename Stage, typename ElementIdx>
  __device__ void prefetch(Value* const& slot, Stage stage, ElementIdx element) const {
    *slot = source[stage * STAGE_ELEMENTS + element * C::THREADS + id];
  }

  __device__ void commit() { ++commits; }

  template<int N>
  __device__ void wait() {
    const int remaining = totalStages - waits - 1;
    valid = valid && N == (remaining < C::PIPE_STAGES ? remaining : C::PIPE_STAGES - 1);
    ++waits;
  }

  template<typename Values>
  __device__ void consume(int stage, const Values& values) {
    const int expectedCommits = stage + C::PIPE_STAGES + 1;
    valid = valid && commits == (expectedCommits < totalStages ? expectedCommits : totalStages);
    valid = valid && stage == consumed && advances == stage + 1;
    ++consumed;
    purlin::static_for<C::ELEMS_PER_THREAD>([&](auto element) {
      destination[stage * STAGE_ELEMENTS + element * C::THREADS + id] =
        values[element] ^ 0x5a5a5a5au;
    });
  }
};

static_assert(cuda::std::is_empty_v<purlin::PipelineOperation>);
static_assert(!cuda::std::is_polymorphic_v<purlin::PipelineOperation>);

template<typename C, bool warpPipelines>
__global__ void checkPipeline(const uint32_t* source, uint32_t* destination, int stages, int* errors) {
  extern __shared__ __align__(16) cuda::std::byte workspace[];
  constexpr int groups = warpPipelines ? 2 : 1;
  const auto group = warpPipelines ? threadIdx.x / C::THREADS : 0u;
  const auto id = warpPipelines ? threadIdx.x % C::THREADS : threadIdx.x;
  const int groupElements = stages * C::THREADS * C::ELEMS_PER_THREAD;
  const auto offset = (blockIdx.x * groups + group) * groupElements;
  SynchronousOperation<C> operation{
    source + offset, destination + offset, stages, id};
  if constexpr (warpPipelines) {
    auto* groupWorkspace = workspace +
      group * C::PIPE_STAGES * C::THREADS * C::ELEMS_PER_THREAD * sizeof(uint32_t);
    purlin::Pipeline<C>::run(groupWorkspace, stages, operation, id);
  } else {
    purlin::Pipeline<C>::run(workspace, stages, operation);
  }
  if (!operation.valid || operation.commits != stages || operation.advances != stages ||
      operation.waits != stages || operation.consumed != stages) {
    atomicAdd(errors, 1);
  }
}

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<int depth, bool warpPipelines = false>
void runCase(int stages) {
  using C = Config<depth, warpPipelines ? 32 : 64>;
  constexpr int blocks = 3;
  constexpr int groups = warpPipelines ? 2 : 1;
  const int count = blocks * groups * stages * C::THREADS * C::ELEMS_PER_THREAD;
  const size_t bytes = count * sizeof(uint32_t);
  std::vector<uint32_t> source(count);
  std::vector<uint32_t> result(count);
  for (int i = 0; i < count; ++i) source[i] = 17u * i + 3u;
  uint32_t* deviceSource = nullptr;
  uint32_t* deviceDestination = nullptr;
  int* deviceErrors = nullptr;
  check(cudaMalloc(&deviceSource, bytes));
  check(cudaMalloc(&deviceDestination, bytes));
  check(cudaMalloc(&deviceErrors, sizeof(int)));
  check(cudaMemcpy(deviceSource, source.data(), bytes, cudaMemcpyHostToDevice));
  check(cudaMemset(deviceDestination, 0, bytes));
  check(cudaMemset(deviceErrors, 0, sizeof(int)));
  constexpr int smem = groups * depth * C::THREADS * C::ELEMS_PER_THREAD * sizeof(uint32_t);
  checkPipeline<C, warpPipelines><<<blocks, groups * C::THREADS, smem>>>(
    deviceSource, deviceDestination, stages, deviceErrors);
  check(cudaGetLastError());
  check(cudaMemcpy(result.data(), deviceDestination, bytes, cudaMemcpyDeviceToHost));
  int errors = 0;
  check(cudaMemcpy(&errors, deviceErrors, sizeof(int), cudaMemcpyDeviceToHost));
  check(cudaFree(deviceSource));
  check(cudaFree(deviceDestination));
  check(cudaFree(deviceErrors));
  if (errors) throw std::runtime_error("Pipeline operation hook ordering failed");
  for (int i = 0; i < count; ++i) {
    if (result[i] != (source[i] ^ 0x5a5a5a5au)) {
      throw std::runtime_error("Pipeline produced incorrect data");
    }
  }
}

template<int depth>
void runDepth() {
  runCase<depth>(depth); // Priming followed immediately by draining.
  runCase<depth>(depth + 1);
  runCase<depth>(3 * depth + 2); // Multiple wraps of the shared-memory ring.
  runCase<depth, true>(depth); // Two independent warp pipelines in one block.
  runCase<depth, true>(depth + 1);
  runCase<depth, true>(3 * depth + 2);
}
}

int main() {
  try {
    runDepth<1>();
    runDepth<2>();
    runDepth<4>();
    runDepth<8>();
    std::puts("Pipeline checks passed for block and warp groups, depths 1/2/4/8");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
