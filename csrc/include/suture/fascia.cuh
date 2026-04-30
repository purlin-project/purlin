//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_FASCIA_CUH
#define SUTURE_FASCIA_CUH
#include "base.cuh"
#include "copy.cuh"
#include "sync.cuh"

template<typename Cfg_>
struct suture::Atom<700, Cfg_> {
  using Config = Cfg_;
  static constexpr int SMEM_SIZE = 0;
  static constexpr int THREADS = Config::THREADS;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config::GMEM_ACCESS_ALIGNMENT_BYTES;

  __device__ __forceinline__
  static void putAsync(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    const cuda::std::byte* __restrict__ const& /*workspace is not needed*/) {
    using CopyElement = AlignedType<Config::ALIGNMENT_BYTES>::type;
    using OpCfg = fascia::PeerOpConfig<
      Config,
      ST, // store op
      CopyElement,
      size_t
    >;
    fascia::putOp<OpCfg>(src, dst, bytes);
  }

  __device__ __forceinline__
  static void getAsync(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes /*in bytes*/,
    const cuda::std::byte* __restrict__ const&) {
    putAsync(dst, src, bytes, nullptr);
  }

  template<typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const&) {
    using RedOp = ArrayInplaceSum<700>;
    if (redArgs.putBlock) {
      // transfer
      // 0. sync with others.
      syncRelaxed(redArgs.remoteSync, redArgs.localSync,redArgs.flag);
      // 1. Do put
      putAsync(redArgs.redPut, redArgs.srcPut, redArgs.bytesPut, nullptr);
      // 2. Notify peer
      __syncthreads();
      if (!threadIdx.x) {
        const cuda::atomic_ref<uint, cuda::thread_scope_device> s{*redArgs.sigCounter};
        if (s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == redArgs.superBlockSize) {
          s.store(0, cuda::memory_order_relaxed);
          auto* __restrict__ signal = redArgs.putSignals;
          const cuda::atomic_ref<uint64_t, cuda::thread_scope_system> rS{*(signal)};
          rS.store(redArgs.flag, cuda::memory_order_release);
        }
      }
      __syncwarp();
    }
    for (int i = static_cast<int>(threadIdx.x) + 1; i < redArgs.world; i += Config::THREADS) {
      const auto peer = (i + redArgs.rank) % redArgs.world;
      auto* __restrict__ signal = redArgs.signals + peer;
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> s{*signal};
      auto isHere = s.load(cuda::memory_order_relaxed) == redArgs.flag;
      while (!isHere) {
        isHere = s.load(cuda::memory_order_relaxed) == redArgs.flag;
      }
      cuda::std::ignore = s.load(cuda::memory_order_acquire);
    }
    __syncthreads();
    // call fascia reduce
    fascia::reduce<Config, RedOp, Element>(redArgs);
  }

  template<typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceLRArgs& redArgs, Element* __restrict__ const&) {
    using RedOp = ArrayInplaceSum<700>;
    // low latency
    fascia::reduce<Config, RedOp, Element>(redArgs);
  }

  __device__ __forceinline__
  static void flush() {}

  __device__ __forceinline__
  static void fence() {
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
  }
};
#endif //SUTURE_FASCIA_CUH