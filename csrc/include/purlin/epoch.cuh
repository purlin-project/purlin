//
// Created by osayamen on 5/25/26.
//

#ifndef PURLIN_SIGNAL_CUH
#define PURLIN_SIGNAL_CUH
namespace purlin {
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
      .trStagingPrefix = ctx.stagingTRSize * senseBit,
      .lrStagingPrefix = senseBit * ctx.world * purlin::PACKET_BUFFER_SIZE,
    };
  }
  __device__ __forceinline__
  static void markEpoch(const Context& ctx, const int bIdx, const uint64_t flag) {
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = flag;
    }
  }
  // Chunked collectives advance the epoch by their per-call flag count. Forcing the
  // advance odd keeps the staging sense bit alternating even for even chunk counts:
  // same-half reuse on consecutive calls is not covered by the signal chain (a
  // skewed peer's gather may still be reading the half the next call's put rewrites).
  // Waits compare with >=, so the skipped flag value is never observed.
  __device__ __forceinline__
  static uint64_t chunkedNextEpoch(const uint64_t& epoch, const uint64_t& advance) {
    return epoch + (advance | 1);
  }
  template<typename PurlinAtom, typename CB, typename AB>
  __device__ __forceinline__
  static void markUnusedEpochs(const Context& ctx, const CB collBlocks,
    const AB activeBlocks, const uint64_t flag, const int tid) {
    const auto leftover = purlin::MAX_NUM_CTAS - collBlocks;
    auto* __restrict__ epochs = ctx.epochs + collBlocks;
    for (int i = tid; i < leftover; i += (PurlinAtom::THREADS * activeBlocks)) {
      epochs[i] = flag;
    }
  }
  template<typename PurlinAtom, int activeBlocks, typename CB>
  __device__ __forceinline__
  static void markUnusedEpochs(const Context& ctx, const CB collBlocks,
    const uint64_t flag, const int tid) {
    markUnusedEpochs<PurlinAtom>(ctx, collBlocks, activeBlocks, flag, tid);
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
  template<typename PurlinAtom>
  __device__ __forceinline__
  static void waitPeerArrivals(uint64_t* __restrict__ const& signalBase,
    const int world, const uint64_t flag, const int tid = static_cast<int>(threadIdx.x)) {
    for (int peer = tid; peer < world; peer += PurlinAtom::THREADS) {
      waitUntilAtLeast(signalBase + peer, flag);
    }
  }
  template<typename PurlinAtom>
  __device__ __forceinline__
  static void waitPointerList(uint64_t** __restrict__ const& signals,
    const int world, const uint64_t flag, const int tid = static_cast<int>(threadIdx.x)) {
    for (int peer = tid; peer < world; peer += PurlinAtom::THREADS) {
      waitUntilAtLeast(signals[peer], flag);
    }
  }
  template<typename T>
  __device__ __forceinline__
  static void signalOne(T* __restrict__ const& signal, const T& v) {
    cuda::atomic_ref<T, cuda::thread_scope_system> sig{*signal};
    sig.store(v, cuda::std::memory_order_release);
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
  // Ring-staging backpressure: once every block of a consumer set has drained a
  // chunk, the last arrival publishes the chunk's flag to the staging owner's
  // consumed signal. The acq_rel counter chain orders every block's reads before
  // the release store, so the producer may rewrite the slot upon observing it.
  __device__ __forceinline__
  static void signalConsumed(uint32_t* __restrict__ const& counter,
    uint64_t* __restrict__ const& signal, const int& blockSetSize, const uint64_t& flag) {
    __syncthreads(); // this block's chunk reads are complete
    if (threadIdx.x / WARP_SIZE == 0) {
      const auto laneId = static_cast<int>(threadIdx.x % WARP_SIZE);
      int shouldNotify = blockSetSize == 1 ? 1 : 0;
      if (blockSetSize > 1 && !laneId) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*counter};
        shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
        if (shouldNotify) {
          s.store(0, cuda::memory_order_relaxed);
        }
      }
      __syncwarp();
      shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
      if (shouldNotify && !laneId) {
        signalOne(signal, flag);
      }
      __syncwarp();
    }
  }
}
#endif //PURLIN_SIGNAL_CUH
