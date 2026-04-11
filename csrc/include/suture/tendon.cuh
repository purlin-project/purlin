//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_TENDON_CUH
#define SUTURE_TENDON_CUH
#include <cuda/atomic>

#include "atom.cuh"
#include "base.cuh"
#include "copy.cuh"
#include "regime.cuh"
namespace suture::tendon {
  // nArch is implicitly 800 in tendon
  constexpr int nArch = 800;
  template<suture::Regime regime>
  struct Reduce {};

  template<>
  struct Reduce<Regime::throughput> {

  };
  template<>
  struct Reduce<Regime::latency> {

  };
  template<typename AtomConfig_>
  struct PipelineConfig {
    using AtomConfig = AtomConfig_;
    static constexpr int UNROLL_FACTOR = AtomConfig::UNROLL_FACTOR;
    static constexpr int THREADS = AtomConfig::THREADS;
    static constexpr int ALIGNMENT_BYTES = AtomConfig::ALIGNMENT_BYTES;
    static constexpr int ELEMS_PER_THREAD = AtomConfig::ELEMS_PER_THREAD;
    static constexpr int PIPE_STAGES = AtomConfig::PIPE_STAGES;
    static constexpr int STAGE_BYTES = THREADS * ELEMS_PER_THREAD * ALIGNMENT_BYTES;
    static constexpr int PIPELINE_BYTES = STAGE_BYTES * PIPE_STAGES;
  };
}

// GMEM (local) -> GMEM(remote)
template<typename Config_>
struct suture::Atom<800, Config_> {
  static_assert(tendon::nArch == 800);
  using Config = tendon::PipelineConfig<Config_>;
  static constexpr int SMEM_SIZE = Config::PIPELINE_BYTES;
  using MaxAlignmentBytes = cuda::std::integral_constant<int, 16>;

  __device__ __forceinline__
  static void putAsync(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    //assert(__isShared(workspace));
    using AT = AlignedType<Config::ALIGNMENT_BYTES>::type;
    if (bytes < Config::PIPELINE_BYTES) {
      // use unrolled direct loads as pipelining is not possible
      using OpCfg = fascia::PeerOpConfig<
        Config,
        ST,
        AT,
        uint32_t
      >;
      fascia::peerOp<OpCfg>(src, dst, bytes);
      return;
    }
    constexpr int VectorWidth = Config::ALIGNMENT_BYTES / sizeof(AT);
    using VT = cutlass::AlignedArray<AT, VectorWidth, Config::ALIGNMENT_BYTES>;
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const int stages = static_cast<int>(bytes / Config::STAGE_BYTES);
    cuda::static_for<Config::PIPE_STAGES>([&vW, &vS](auto i) {
      cuda::static_for<Config::ELEMS_PER_THREAD>([&i, &vW, &vS](auto j) {
        const int slot = ((i * Config::ELEMS_PER_THREAD + j) * Config::THREADS) + threadIdx.x;
        // async gmem -> smem
        cpAsync<Config::ALIGNMENT_BYTES>(vW + slot, vS + slot);
      });
      cpAsyncCommit();
    });
    VT reginald[Config::ELEMS_PER_THREAD];
    for (int i = Config::PIPE_STAGES; i < stages; ++i) {
      cpAsyncWait<Config::PIPE_STAGES - 1>();
      const int stage_out = i - Config::PIPE_STAGES;
      const int cs = stage_out % Config::PIPE_STAGES;
      cuda::static_for<Config::ELEMS_PER_THREAD>([&i, &cs, &vW, &reginald, &vS](auto j) {
        const int csW = (cs * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        const long int slot = (static_cast<size_t>(i) * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // smem -> rmem
        reginald[j] = vW[csW];
        // async gmem -> smem prefetch
        cpAsync<Config::ALIGNMENT_BYTES>(vW + csW, vS + slot);
      });
      cuda::static_for<Config::ELEMS_PER_THREAD>([&stage_out, &reginald, &vD](auto j) {
        const long int slot = (stage_out * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // rmem -> gmem
        vD[slot] = reginald[j];
      });
      // commit async transfers from this stage
      cpAsyncCommit();
    }
    // tail
    cuda::static_for<Config::PIPE_STAGES>([&vW, &reginald, &vS, &vD, &stages](auto i) {
      const int stage = (stages - Config::PIPE_STAGES) + i;
      const int cs = stage % Config::PIPE_STAGES;
      cpAsyncWait<Config::PIPE_STAGES - 1 - i>();
      cuda::static_for<Config::ELEMS_PER_THREAD>([&i, &cs, &vW, &reginald, &vS, &stages](auto j) {
        const int csW = (cs * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // smem -> rmem
        reginald[j] = vW[csW];
      });
      cuda::static_for<Config::ELEMS_PER_THREAD>([&stage, &reginald, &vD](auto j) {
        const long int slot = (stage * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // rmem -> gmem
        vD[slot] = reginald[j];
      });
    });
    // residue
    const auto cutoff = stages * static_cast<size_t>(Config::STAGE_BYTES);
    if (bytes > cutoff) {
      const auto cutoffElems = cutoff / Config::ALIGNMENT_BYTES;
      const auto residue = static_cast<int>((bytes - cutoff) / Config::ALIGNMENT_BYTES); // elements not bytes
      vS += cutoffElems;
      vD += cutoffElems;
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += Config::THREADS) {
        suture::store(vD + i, vS[i]);
      }
    }
  }

  __device__ __forceinline__
  static void getAsync(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& workspace,
    const size_t& bytes) {
    putAsync(dst, src, bytes, workspace);
  }

  __device__ __forceinline__
  static void flush() {}

  __device__ __forceinline__
  static void fence() {
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
  }
};
#endif //SUTURE_TENDON_CUH