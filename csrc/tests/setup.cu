#include <purlin/setup.cuh>
#include <purlin/host.cuh>

#include <algorithm>
#include <memory>
#include <type_traits>

#if defined(_NVSHMEM_H_) || defined(_NVSHMEMX_H_) || defined(NVSHMEM_HOST_H)
#error "Core headers must not depend on NVSHMEM"
#endif

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<typename Error, typename Function>
void rejects(Function&& function) {
  try {
    function();
  } catch (const Error&) {
    return;
  }
  throw std::runtime_error("Expected an exception");
}

enum class Mapping { valid, shortTable, nullPeer, unalignedPeer, multicast };

struct Stats {
  int allocations = 0;
  int releases = 0;
  int providersDestroyed = 0;
  uint64_t* marker = nullptr;
  cudaStream_t stream = nullptr;
  bool completed = true;
};

struct TestMemory {
  struct Allocation {
    std::vector<void*> peers;
    void* multicast = nullptr;
    std::vector<void*> storage;
    std::unique_ptr<int> token = std::make_unique<int>(42);
  };

  Stats* stats;
  Mapping mapping;
  std::unique_ptr<int> token = std::make_unique<int>(42);

  explicit TestMemory(Stats& stats_, Mapping mapping_ = Mapping::valid)
    : stats(&stats_), mapping(mapping_) {}
  TestMemory(TestMemory&&) = default;
  TestMemory(const TestMemory&) = delete;
  ~TestMemory() { if (token) ++stats->providersDestroyed; }

  Allocation allocate_zeroed(size_t bytes, size_t alignment, cudaStream_t stream) {
    Allocation allocation;
    for (int rank = 0; rank < 2; ++rank) {
      void* local = nullptr;
      CHECK_CUDA(cudaMalloc(&local, bytes));
      require(reinterpret_cast<uintptr_t>(local) % alignment == 0, "Invalid test allocation alignment");
      CHECK_CUDA(cudaMemsetAsync(local, 0, bytes, stream));
      allocation.storage.push_back(local);
      allocation.peers.push_back(local);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    ++stats->allocations;
    if (mapping == Mapping::shortTable) allocation.peers.pop_back();
    if (mapping == Mapping::nullPeer) allocation.peers[1] = nullptr;
    if (mapping == Mapping::unalignedPeer) {
      allocation.peers[1] = static_cast<char*>(allocation.peers[1]) + 1;
    }
    if (mapping == Mapping::multicast) allocation.multicast = allocation.peers[0];
    return allocation;
  }

  void deallocate(Allocation& allocation) noexcept {
    stats->completed &= token && allocation.token;
    if (stats->marker != nullptr) {
      stats->completed &= cudaStreamQuery(stats->stream) == cudaSuccess;
      uint64_t value = 0;
      CHECK_CUDA(cudaMemcpy(&value, stats->marker, sizeof(value), cudaMemcpyDeviceToHost));
      stats->completed &= value == 42;
    }
    for (void* local : allocation.storage) CHECK_CUDA(cudaFree(local));
    ++stats->releases;
  }
};

using Managed = purlin::ManagedContext<TestMemory>;
static_assert(purlin::SymmetricMemoryProvider<TestMemory>);
static_assert(std::is_trivially_copyable_v<purlin::Context>);
static_assert(!std::is_copy_constructible_v<Managed>);
static_assert(std::is_nothrow_move_constructible_v<Managed>);

template<typename T>
T* peerPointer(T** table, int rank) {
  T* pointer = nullptr;
  CHECK_CUDA(cudaMemcpy(&pointer, table + rank, sizeof(pointer), cudaMemcpyDeviceToHost));
  return pointer;
}

void checkWorkspace(const purlin::Context& ctx) {
  for (int peer = 0; peer < 2; ++peer) {
    struct Region { uintptr_t begin; size_t bytes; size_t alignment; };
    Region regions[] = {
      {reinterpret_cast<uintptr_t>(peerPointer(ctx.staging, peer)), 2 * ctx.stagingTRSize, purlin::MAX_ACCESS_ALIGNMENT},
      {reinterpret_cast<uintptr_t>(peerPointer(ctx.stagingLR, peer)), 4 * purlin::PACKET_BUFFER_SIZE, purlin::MAX_ACCESS_ALIGNMENT},
      {reinterpret_cast<uintptr_t>(peerPointer(ctx.signals, peer)), 2 * sizeof(uint64_t), alignof(uint64_t)},
      {reinterpret_cast<uintptr_t>(peerPointer(ctx.gatherSignals, peer)), 2 * sizeof(uint64_t), alignof(uint64_t)},
      {reinterpret_cast<uintptr_t>(peerPointer(ctx.consumedSignals, peer)), 2 * sizeof(uint64_t), alignof(uint64_t)},
      {reinterpret_cast<uintptr_t>(peerPointer(ctx.varLenSignals, peer)), 4 * sizeof(purlin::LRP), alignof(purlin::LRP)},
    };
    std::sort(std::begin(regions), std::end(regions), [](auto a, auto b) { return a.begin < b.begin; });
    uintptr_t end = 0;
    for (const auto region : regions) {
      require(region.begin >= end, "Workspace regions overlap");
      require(region.begin % region.alignment == 0, "Workspace region is unaligned");
      end = region.begin + region.bytes;
      std::vector<unsigned char> bytes(region.bytes);
      CHECK_CUDA(cudaMemcpy(bytes.data(), reinterpret_cast<void*>(region.begin), region.bytes,
        cudaMemcpyDeviceToHost));
      require(std::all_of(bytes.begin(), bytes.end(), [](auto byte) { return byte == 0; }),
        "Workspace is not zeroed");
    }
  }
}

__global__ void writeMarker(purlin::Context ctx) {
  const auto start = clock64();
  while (clock64() - start < 10000000ULL) {}
  ctx.signals[0][0] = 42;
}

void checkLifecycle(cudaStream_t stream, size_t stagingBytes) {
  Stats stats;
  auto managed = purlin::initialize(0, 2, stream, TestMemory{stats}, stagingBytes);
  checkWorkspace(managed.context());
  require(managed.context().stagingTRSize == stagingBytes, "Staging size was lost");
  require(managed.context().mcStagingTR == nullptr, "Unexpected multicast mapping");
  Managed moved(std::move(managed));
  rejects<std::logic_error>([&] { managed.context(); });
  const auto& ctx = moved.context();
  const purlin::WorkspaceMemory workspace{
    .stagingLR = ctx.stagingLR, .stagingTR = ctx.staging,
    .signals = ctx.signals, .gatherSignals = ctx.gatherSignals,
    .consumedSignals = ctx.consumedSignals, .varLenSignals = ctx.varLenSignals,
  };
  const auto borrowed = purlin::initialize(0, 2, workspace, stream, stagingBytes);
  purlin::finalize(borrowed, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  require(stats.releases == 0, "Borrowed finalization released the workspace");
  cudaStream_t finalStream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&finalStream, cudaStreamNonBlocking));
  stats.marker = peerPointer(ctx.signals, 0);
  stats.stream = finalStream;
  writeMarker<<<1, 1, 0, finalStream>>>(ctx);
  CHECK_CUDA(cudaGetLastError());
  purlin::finalize(moved, finalStream);
  purlin::finalize(moved, finalStream);
  CHECK_CUDA(cudaStreamDestroy(finalStream));
  rejects<std::logic_error>([&] { moved.context(); });
  require(stats.releases == stats.allocations && stats.completed, "Managed cleanup failed");
  require(stats.providersDestroyed == 1, "Provider lifetime was not retained");
}

void checkValidation(cudaStream_t stream) {
  Stats stats;
  for (const int world : {0, 1, purlin::MAX_RANKS_PER_DOMAIN + 1}) {
    rejects<std::invalid_argument>([&] { purlin::initialize(0, world, stream, TestMemory{stats}); });
  }
  for (const int rank : {-1, 2}) {
    rejects<std::invalid_argument>([&] { purlin::initialize(rank, 2, stream, TestMemory{stats}); });
  }
  for (const size_t size : {size_t{0}, purlin::MIN_CHUNK_SIZE - 1, purlin::MAX_STAGING_SIZE + 1}) {
    rejects<std::invalid_argument>([&] { purlin::initialize(0, 2, stream, TestMemory{stats}, size); });
  }
  require(stats.allocations == 0, "Invalid arguments reached the provider");
  for (const auto mapping : {Mapping::shortTable, Mapping::nullPeer, Mapping::unalignedPeer}) {
    rejects<std::invalid_argument>([&] {
      purlin::initialize(0, 2, stream, TestMemory{stats, mapping}, purlin::MIN_CHUNK_SIZE);
    });
  }
  require(stats.releases == stats.allocations, "Invalid mappings leaked allocations");
}

void checkScopeCleanup(cudaStream_t stream) {
  Stats stats;
  {
    auto managed = purlin::initialize(0, 2, stream, TestMemory{stats}, purlin::MIN_CHUNK_SIZE);
    Managed replacement;
    replacement = std::move(managed);
  }
  require(stats.releases == stats.allocations && stats.providersDestroyed == 1,
    "Scope cleanup failed");
}

void checkMulticast(cudaStream_t stream) {
  Stats stats;
  auto managed = purlin::initialize(0, 2, stream, TestMemory{stats, Mapping::multicast}, purlin::MIN_CHUNK_SIZE);
  const auto& ctx = managed.context();
  const bool enabled = std::getenv("PURLIN_DISABLE_MULTIMEM") == nullptr;
  const bool latencyEnabled = enabled && std::getenv("PURLIN_DISABLE_MULTIMEM_LR") == nullptr;
  require(ctx.mcStagingTR == (enabled ? peerPointer(ctx.staging, 0) : nullptr), "Invalid multicast staging");
  require(ctx.mcStagingLR == (latencyEnabled ? peerPointer(ctx.stagingLR, 0) : nullptr), "Invalid multicast latency staging");
  purlin::finalize(managed, stream);
}
}

int main() {
  try {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkValidation(stream);
    checkLifecycle(stream, purlin::MIN_CHUNK_SIZE);
    checkLifecycle(stream, 2 * purlin::MIN_CHUNK_SIZE + purlin::MAX_ACCESS_ALIGNMENT);
    checkScopeCleanup(stream);
    checkMulticast(stream);
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::puts("Managed and caller-owned setup checks passed");
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
