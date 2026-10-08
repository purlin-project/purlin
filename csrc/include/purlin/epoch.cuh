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
  // forcing the update value to be odd is necessary to ensure the epoch % 2 always alternates
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
    // Epoch signals reside in global memory and synchronize across devices.
    uint64_t value;
    do {
      asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
        : "=l"(value) : "l"(signal) : "memory");
    } while (value < flag);
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
      : "=l"(value) : "l"(signal) : "memory");
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
  __device__ __forceinline__
  static void signalOne(uint64_t* __restrict__ const& signal, const uint64_t& v) {
    asm volatile("st.release.sys.global.b64 [%0], %1;"
      :: "l"(signal), "l"(v) : "memory");
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
  __device__ __forceinline__
  static bool lastArrival(uint32_t* __restrict__ const& counter,
    const int blockSetSize) {
    int last = blockSetSize == 1 ? 1 : 0;
    // below lets the compiler avoid emitting some warp collective instructions
    const auto laneId = cuda::ptx::get_sreg_laneid();
    if (blockSetSize > 1 && !laneId) {
      uint32_t arrivals;
      asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;"
        : "=r"(arrivals) : "l"(counter), "r"(1) : "memory");
      last = arrivals + 1 == blockSetSize;
      if (last) {
        asm volatile("st.relaxed.gpu.global.b32 [%0], %1;"
          :: "l"(counter), "r"(0) : "memory");
      }
    }
    __syncwarp();
    return __shfl_sync(0xffffffff, last, 0);
  }
  __device__ __forceinline__
  static void signalConsumed(uint32_t* __restrict__ const& counter,
    uint64_t* __restrict__ const& signal, const int& blockSetSize, const uint64_t& flag) {
    __syncthreads(); // this block's chunk reads are complete
    if (threadIdx.x / WARP_SIZE == 0) {
      const auto laneId = cuda::ptx::get_sreg_laneid();
      if (lastArrival(counter, blockSetSize) && !laneId) {
        signalOne(signal, flag);
      }
      __syncwarp();
    }
  }
}
#endif //PURLIN_SIGNAL_CUH
