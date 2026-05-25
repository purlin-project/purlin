//
// Created by osayamen on 5/25/26.
//

#ifndef SUTURE_SIGNAL_CUH
#define SUTURE_SIGNAL_CUH
namespace suture {
  struct EpochState {
    const uint64_t epoch;
    const uint64_t nextEpoch;
    const uint senseBit;
    const size_t trStagingPrefix;
    const size_t lrStagingPrefix;
  };
  __device__ __forceinline__
  static auto makeEpochState(const Context& ctx, const int& bIdx) {
    const auto epoch = ctx.epochs[bIdx];
    const auto senseBit = static_cast<uint>(epoch % 2);
    return EpochState{
      .epoch = epoch,
      .nextEpoch = epoch + static_cast<uint64_t>(1),
      .senseBit = senseBit,
      .trStagingPrefix = STAGING_BUFFER_SIZE_ * senseBit,
      .lrStagingPrefix = senseBit * ctx.world * suture::PACKET_BUFFER_SIZE,
    };
  }
  __device__ __forceinline__
  static void markEpoch(const Context& ctx, const int bIdx, const uint64_t flag) {
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = flag;
    }
  }
  template<typename SutureAtom, typename CB, typename AB>
  __device__ __forceinline__
  static void markUnusedEpochs(const Context& ctx, const CB collBlocks,
    const AB activeBlocks, const uint64_t flag, const int tid) {
    const auto leftover = suture::MAX_NUM_CTAS - collBlocks;
    auto* __restrict__ epochs = ctx.epochs + collBlocks;
    for (int i = tid; i < leftover; i += (SutureAtom::THREADS * activeBlocks)) {
      epochs[i] = flag;
    }
  }
  template<typename SutureAtom, int activeBlocks, typename CB>
  __device__ __forceinline__
  static void markUnusedEpochs(const Context& ctx, const CB collBlocks,
    const uint64_t flag, const int tid) {
    markUnusedEpochs<SutureAtom>(ctx, collBlocks, activeBlocks, flag, tid);
  }
  __device__ __forceinline__
  static void waitUntilAtLeast(uint64_t* __restrict__ const& signal, const uint64_t flag) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
    auto isHere = sig.load(cuda::memory_order_relaxed) >= flag;
    while (!isHere) {
      isHere = sig.load(cuda::memory_order_relaxed) >= flag;
    }
    cuda::std::ignore = sig.load(cuda::memory_order_acquire);
  }
  template<typename SutureAtom>
  __device__ __forceinline__
  static void waitPeerArrivals(uint64_t* __restrict__ const& signalBase,
    const int world, const uint64_t flag, const int tid = static_cast<int>(threadIdx.x)) {
    for (int peer = tid; peer < world; peer += SutureAtom::THREADS) {
      waitUntilAtLeast(signalBase + peer, flag);
    }
  }
  template<typename SutureAtom>
  __device__ __forceinline__
  static void waitPointerList(uint64_t** __restrict__ const& signals,
    const int world, const uint64_t flag, const int tid = static_cast<int>(threadIdx.x)) {
    for (int peer = tid; peer < world; peer += SutureAtom::THREADS) {
      waitUntilAtLeast(signals[peer], flag);
    }
  }
  __device__ __forceinline__
  static void signalOne(uint64_t* __restrict__ const& signal, const uint64_t flag) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
    sig.store(flag, cuda::std::memory_order_release);
  }
  __device__ __forceinline__
  static void signalAllPeers(uint64_t** __restrict__ const& signals,
    const int rank, const int world, const uint64_t flag, const int laneId) {
    for (int peer = laneId; peer < world; peer += WARP_SIZE) {
      signalOne(signals[peer] + rank, flag);
    }
  }
  __device__ __forceinline__
  static void signalPointerList(uint64_t** __restrict__ const& signals,
    const int world, const uint64_t flag, const int laneId) {
    for (int peer = laneId; peer < world; peer += WARP_SIZE) {
      signalOne(signals[peer], flag);
    }
  }
}
#endif //SUTURE_SIGNAL_CUH
