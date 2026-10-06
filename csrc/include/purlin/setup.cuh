//
// Created by Osayamen on 4/16/26.
//

#ifndef PURLIN_SETUP_CUH
#define PURLIN_SETUP_CUH
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "constants.cuh"
#include "context.cuh"

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
namespace purlin {
  __host__ __forceinline__
  auto initialize(const int& rank, const int& world, const WorkspaceMemory& w, cudaStream_t stream,
    const size_t& stagingTRSize = STAGING_BUFFER_SIZE_) {
    Context ctx{};
    if (world <= 1 || world > MAX_RANKS_PER_DOMAIN) {
      const auto errmsg = "world: " + std::to_string(world) + " is invalid";
      throw std::runtime_error(errmsg);
    }
    if (stagingTRSize > MAX_STAGING_SIZE) {
      const auto errmsg = "stagingSize: " + std::to_string(stagingTRSize) + " exceeds max: " + std::to_string(MAX_STAGING_SIZE);
      throw std::runtime_error(errmsg);
    }
    using ET = cuda::std::remove_pointer_t<decltype(ctx.epochs)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.epochs, sizeof(ET) * purlin::MAX_NUM_CTAS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.epochs, 0, sizeof(ET) * purlin::MAX_NUM_CTAS, stream));
    using PCT = cuda::std::remove_pointer_t<decltype(ctx.putCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.putCounter, sizeof(PCT) * world * MAX_CHUNKS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.putCounter, 0, sizeof(PCT) * world * MAX_CHUNKS, stream));
    using RCT = cuda::std::remove_pointer_t<decltype(ctx.redCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.redCounter, sizeof(RCT) * MAX_CHUNKS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.redCounter, 0, sizeof(RCT) * MAX_CHUNKS, stream));
    using CCT = cuda::std::remove_pointer_t<decltype(ctx.consumedCounter)>;
    CHECK_CUDA(cudaMallocAsync(&ctx.consumedCounter, sizeof(CCT) * world * MAX_CHUNKS, stream));
    CHECK_CUDA(cudaMemsetAsync(ctx.consumedCounter, 0, sizeof(CCT) * world * MAX_CHUNKS, stream));
    CHECK_CUDA(cudaMallocAsync(&ctx.sizes, 2 * sizeof(size_t) * world, stream));
    ctx.staging = w.stagingTR;
    ctx.stagingLR = w.stagingLR;
    ctx.mcStagingTR = w.mcStagingTR;
    ctx.mcStagingLR = w.mcStagingLR;
    ctx.signals = w.signals;
    ctx.gatherSignals = w.gatherSignals;
    ctx.consumedSignals = w.consumedSignals;
    ctx.world = cuda::fast_mod_div<int, true>{world};
    ctx.actualWorld = cuda::fast_mod_div<int>{(world - 1)};
    ctx.world_l = cuda::fast_mod_div<size_t, true>{static_cast<size_t>(world)};
    ctx.rank = rank;
    ctx.stagingTRSize = stagingTRSize;
    ctx.varLenSignals = w.varLenSignals;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return ctx;
  }
  // for Python bindings
  __host__ __forceinline__
  auto initialize(const int& rank, const int& world,
    cuda::std::byte** const& stagingLR,
    cuda::std::byte** const& stagingTR,
    uint64_t** const& signals,
    uint64_t** const& gatherSignals,
    uint64_t** const& consumedSignals,
    LRP** const& varLenSignals,
    const size_t& stagingTRSize,
    cudaStream_t stream) {
    const WorkspaceMemory w{
      .stagingLR = stagingLR,
      .stagingTR = stagingTR,
      .signals = signals,
      .gatherSignals = gatherSignals,
      .consumedSignals = consumedSignals,
      .varLenSignals = varLenSignals
    };
    return initialize(rank, world, w, stream, stagingTRSize);
  }

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

  __host__ __forceinline__
  void finalize(const Context& ctx, cudaStream_t stream) {
    CHECK_CUDA(cudaFreeAsync(ctx.epochs, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.putCounter, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.redCounter, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.consumedCounter, stream));
    CHECK_CUDA(cudaFreeAsync(ctx.sizes, stream));
  }

  template<typename Provider>
  concept SymmetricMemoryProvider = requires(Provider& provider,
    typename Provider::Allocation& allocation, size_t bytes, cudaStream_t stream) {
    { provider.allocate_zeroed(bytes, bytes, stream) } -> std::same_as<typename Provider::Allocation>;
    { provider.deallocate(allocation) } -> std::same_as<void>;
    { allocation.peers.size() } -> std::convertible_to<size_t>;
    { allocation.peers[0] } -> std::convertible_to<void*>;
    { allocation.multicast } -> std::convertible_to<void*>;
  };

  namespace detail {
    template<typename T, typename Peers>
    T** workspacePointerTable(const Peers& peers, size_t offset, cudaStream_t stream) {
      std::vector<T*> pointers(peers.size());
      for (size_t rank = 0; rank < peers.size(); ++rank) {
        void* base = peers[rank];
        pointers[rank] = reinterpret_cast<T*>(static_cast<cuda::std::byte*>(base) + offset);
      }
      T** table = nullptr;
      const size_t bytes = sizeof(T*) * pointers.size();
      CHECK_CUDA(cudaMallocAsync(&table, bytes, stream));
      CHECK_CUDA(cudaMemcpyAsync(table, pointers.data(), bytes, cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
      return table;
    }
  }

  template<SymmetricMemoryProvider Provider>
  class ManagedContext;

  template<SymmetricMemoryProvider Provider>
  void finalize(ManagedContext<Provider>& managed, cudaStream_t stream);

  template<SymmetricMemoryProvider Provider>
  class ManagedContext {
    struct State {
      Provider provider;
      typename Provider::Allocation allocation;
      cudaStream_t stream;
      WorkspaceMemory workspace{};
      Context ctx{};

      State(Provider provider_, size_t bytes, size_t alignment, cudaStream_t stream_)
        : provider(std::move(provider_)),
          allocation(provider.allocate_zeroed(bytes, alignment, stream_)), stream(stream_) {}

      ~State() {
        CHECK_CUDA(cudaStreamSynchronize(stream));
        if (ctx.epochs != nullptr) purlin::finalize(ctx, stream);
        void* tables[] = {workspace.stagingLR, workspace.stagingTR, workspace.signals,
          workspace.gatherSignals, workspace.consumedSignals, workspace.varLenSignals};
        for (void* table : tables) {
          if (table != nullptr) CHECK_CUDA(cudaFreeAsync(table, stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        provider.deallocate(allocation);
      }
    };

    std::unique_ptr<State> state_;
    friend void finalize<Provider>(ManagedContext& managed, cudaStream_t stream);

  public:
    ManagedContext() = default;
    ManagedContext(const ManagedContext&) = delete;
    ManagedContext& operator=(const ManagedContext&) = delete;
    ManagedContext(ManagedContext&&) noexcept = default;
    ManagedContext& operator=(ManagedContext&&) noexcept = default;

    ManagedContext(int rank, int world, cudaStream_t stream, Provider provider,
      size_t stagingTRSize = STAGING_BUFFER_SIZE_) {
      if (world <= 1 || world > MAX_RANKS_PER_DOMAIN || rank < 0 || rank >= world) {
        throw std::invalid_argument("Invalid Purlin rank or world size");
      }
      constexpr size_t alignment = MAX_ACCESS_ALIGNMENT;
      if (stagingTRSize < MIN_CHUNK_SIZE || stagingTRSize > MAX_STAGING_SIZE ||
          stagingTRSize % alignment != 0) {
        throw std::invalid_argument("Invalid Purlin staging size");
      }
      const size_t signalOffset = 2 * (stagingTRSize + world * PACKET_BUFFER_SIZE);
      const size_t lengthOffset = cuda::round_up(signalOffset + 3 * world * sizeof(uint64_t), alignment);
      const size_t bytes = lengthOffset + 2 * world * sizeof(LRP);
      state_ = std::make_unique<State>(std::move(provider), bytes, alignment, stream);
      const auto& allocation = state_->allocation;
      if (allocation.peers.size() != static_cast<size_t>(world)) {
        throw std::invalid_argument("The provider must return one pointer per rank");
      }
      for (size_t peer = 0; peer < allocation.peers.size(); ++peer) {
        void* pointer = allocation.peers[peer];
        if (pointer == nullptr || reinterpret_cast<uintptr_t>(pointer) % alignment != 0) {
          throw std::invalid_argument("The provider returned a null or unaligned peer pointer");
        }
      }
      void* multicast = allocation.multicast;
      if (reinterpret_cast<uintptr_t>(multicast) % alignment != 0) {
        throw std::invalid_argument("The provider returned an unaligned multicast pointer");
      }
      auto& workspace = state_->workspace;
      workspace.stagingTR = detail::workspacePointerTable<cuda::std::byte>(allocation.peers, 0, stream);
      workspace.stagingLR = detail::workspacePointerTable<cuda::std::byte>(allocation.peers, 2 * stagingTRSize, stream);
      workspace.signals = detail::workspacePointerTable<uint64_t>(allocation.peers, signalOffset, stream);
      workspace.gatherSignals = detail::workspacePointerTable<uint64_t>(allocation.peers,
        signalOffset + world * sizeof(uint64_t), stream);
      workspace.consumedSignals = detail::workspacePointerTable<uint64_t>(allocation.peers,
        signalOffset + 2 * world * sizeof(uint64_t), stream);
      workspace.varLenSignals = detail::workspacePointerTable<LRP>(allocation.peers, lengthOffset, stream);
      if (std::getenv("PURLIN_DISABLE_MULTIMEM") == nullptr) {
        workspace.mcStagingTR = static_cast<cuda::std::byte*>(multicast);
      }
      if (workspace.mcStagingTR != nullptr && std::getenv("PURLIN_DISABLE_MULTIMEM_LR") == nullptr) {
        workspace.mcStagingLR = workspace.mcStagingTR + 2 * stagingTRSize;
      }
      state_->ctx = purlin::initialize(rank, world, workspace, stream, stagingTRSize);
    }

    Context& context() & {
      if (!state_) throw std::logic_error("The managed context is empty");
      return state_->ctx;
    }

    const Context& context() const& {
      if (!state_) throw std::logic_error("The managed context is empty");
      return state_->ctx;
    }

    Context& context() && = delete;
    const Context& context() const&& = delete;
  };

  template<SymmetricMemoryProvider Provider>
  auto initialize(int rank, int world, cudaStream_t stream, Provider provider,
    size_t stagingTRSize = STAGING_BUFFER_SIZE_) {
    return ManagedContext<Provider>(rank, world, stream, std::move(provider), stagingTRSize);
  }

  template<SymmetricMemoryProvider Provider>
  void finalize(ManagedContext<Provider>& managed, cudaStream_t stream) {
    if (managed.state_) {
      managed.state_->stream = stream;
      managed.state_.reset();
    }
  }
}
#endif //PURLIN_SETUP_CUH
