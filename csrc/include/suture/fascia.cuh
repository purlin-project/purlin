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