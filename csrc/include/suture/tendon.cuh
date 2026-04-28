//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_TENDON_CUH
#define SUTURE_TENDON_CUH
#include <cuda/atomic>
#include <cuda/cmath>

#include "atom.cuh"
#include "base.cuh"
#include "copy.cuh"
#include "math.cuh"
#include "sync.cuh"

namespace suture::tendon {
  // nArch is implicitly 800 in tendon
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
    // reduction config
    static constexpr int PRODUCER_WARPS = THREADS - (2 * WARP_SIZE);
    static constexpr int PRODUCER_THREADS = PRODUCER_WARPS * WARP_SIZE;
    static constexpr int CONSUMER_THREADS = THREADS - PRODUCER_THREADS;
    static constexpr int CONSUMER_WARPS = CONSUMER_THREADS / WARP_SIZE;
    static constexpr int CONS_ELEMS_PER_THREAD = ELEMS_PER_THREAD / CONSUMER_WARPS;
    static constexpr int STAGES_PER_WARP = PIPE_STAGES / PRODUCER_WARPS;
    static constexpr int RED_STAGE_BYTES = PRODUCER_THREADS * ELEMS_PER_THREAD * ALIGNMENT_BYTES;
    static constexpr int RED_PIPELINE_BYTES = RED_STAGE_BYTES * PIPE_STAGES;
    static constexpr int RED_SMEM_BYTES = RED_PIPELINE_BYTES + (CONSUMER_WARPS * sizeof(uint32_t) * PIPE_STAGES);
  };

  template<typename Cfg>
  __device__ __forceinline__
  void redProducer(const ReduceTRArgs& redArgs,
    cuda::std::byte* __restrict__ const& workspace,
    uint32_t* __restrict__ const& flags,
    const int& totalStages, const int& prodId, const int& tId) {
    static_assert(Cfg::PIPE_STAGES % Cfg::PRODUCER_WARPS == 0);
    static_assert(Cfg::CONSUMER_WARPS < WARP_SIZE);
    const int laneId = tId % WARP_SIZE;
    using AT = AlignedType<Cfg::ALIGNMENT_BYTES>::type;
    constexpr int VectorWidth = Cfg::ALIGNMENT_BYTES / sizeof(AT);
    using VT = cutlass::AlignedArray<AT, VectorWidth, Cfg::ALIGNMENT_BYTES>;
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(redArgs.srcRed);
    const int producerStages = totalStages / Cfg::PRODUCER_WARPS +
      (prodId < (totalStages % Cfg::PRODUCER_WARPS));
    // prime pipeline
    cuda::static_for<Cfg::STAGES_PER_WARP>([&](auto i) {
      const int stage = prodId + i * Cfg::PRODUCER_WARPS;
      const auto peer = stage % redArgs.world;
      const auto peerSlot = stage / redArgs.world;
      cuda::static_for<Cfg::ELEMS_PER_THREAD>([&](auto j) {
        const int stagingSlot = (stage * Cfg::ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
        const int dataSlot = (peerSlot * Cfg::ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
        const auto dataOffset = static_cast<size_t>(peer) * redArgs.totalBytes + dataSlot;
        // async gmem -> smem
        cpAsync(vW + stagingSlot, vS + dataOffset);
      });
      cpAsyncCommit();
    });
    // steady state
    for (int i = Cfg::STAGES_PER_WARP; i < producerStages; ++i) {
      const auto dataStage = prodId + i * Cfg::PRODUCER_WARPS;
      const int stage = dataStage % Cfg::TOTAL_PIPE_STAGES;
      cpAsyncWait<Cfg::STAGES_PER_WARP - 1>();
      __syncwarp();
      if (laneId < Cfg::CONSUMER_WARPS) {
        auto* __restrict__ fP = flags + (stage * Cfg::CONSUMER_WARPS + laneId);
        const cuda::atomic_ref<uint32_t, cuda::thread_scope_block> flag{*fP};
        flag.store(full, cuda::memory_order_release);
        auto isStageEmpty = flag.load(cuda::memory_order_relaxed) == empty;
        while (!isStageEmpty) {
          isStageEmpty = flag.load(cuda::memory_order_relaxed) == empty;
        }
        cuda::std::ignore = flag.load(cuda::memory_order_acquire);
      }
      __syncwarp();
      // refill
      {
        const auto peer = dataStage % redArgs.world;
        const auto peerSlot = static_cast<size_t>(dataStage / redArgs.world);
        cuda::static_for<Cfg::ELEMS_PER_THREAD>([&](auto j) {
          const int stagingSlot = (stage * Cfg::ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
          const size_t dataSlot = (peerSlot * Cfg::ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
          const auto dataOffset = static_cast<size_t>(peer) * redArgs.totalBytes + dataSlot;
          // async gmem -> smem
          cpAsync(vW + stagingSlot, vS + dataOffset);
        });
        cpAsyncCommit();
      }
    }
    // tail flush
    const auto firstTailStage = (prodId + producerStages * Cfg::PRODUCER_WARPS) % Cfg::TOTAL_PIPE_STAGES;
    cuda::static_for<Cfg::PIPE_STAGES>([&](auto i) {
      constexpr int remaining = Cfg::PIPE_STAGES - 1 - i;
      const auto stage = (firstTailStage + i * Cfg::PRODUCER_WARPS) % Cfg::TOTAL_PIPE_STAGES;
      cpAsyncWait<remaining>();
      __syncwarp();
      if (laneId < Cfg::CONSUMER_WARPS) {
        auto* __restrict__ fP = flags + (stage * Cfg::CONSUMER_WARPS + laneId);
        const cuda::atomic_ref<uint32_t, cuda::thread_scope_block> flag{*fP};
        flag.store(full, cuda::memory_order_release);
      }
    });
  }

  template<typename Cfg, typename RedOp, typename Element>
  __device__ __forceinline__
  void redConsumer(const ReduceTRArgs& redArgs,
    const cuda::std::byte* __restrict__ const& workspace,
    uint32_t* __restrict__ const& flags,
    const int& totalStages, const int& consId, const int& tId) {
    static_assert(Cfg::CONSUMER_THREADS % WARP_SIZE == 0);
    static_assert(Cfg::ELEMS_PER_THREAD % Cfg::CONSUMER_WARPS == 0);
    const auto laneId = tId % WARP_SIZE;
    static_assert(Cfg::ALIGNMENT_BYTES % sizeof(Element) == 0);
    using VE = cuda::std::conditional_t<
      (Cfg::ALIGNMENT_BYTES > sizeof(Element)), Element, typename Element2<Element>::type>;
    using AccumType = cuda::std::conditional_t<
      (Cfg::ALIGNMENT_BYTES > sizeof(Element)), ReduceAccumType<Element>, Element2<ReduceAccumType<Element>>>;
    constexpr int vectorWidth = Cfg::ALIGNMENT_BYTES / sizeof(VE);
    using AVT = cutlass::AlignedArray<AccumType, vectorWidth>;
    using VER = DataToRawType<VE>::type;
    using VT = cutlass::AlignedArray<VER, vectorWidth>;
    using DVT = cutlass::AlignedArray<VE, vectorWidth>;
    const auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(redArgs.src);
    auto* __restrict__ vD = reinterpret_cast<DVT*>(redArgs.dst);
    VT stash[Cfg::CONS_ELEMS_PER_THREAD];
    AVT accumulators[Cfg::CONS_ELEMS_PER_THREAD];
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VE, AccumType> storeConv{};
    constexpr RedOp op{};
    cuda::static_for<Cfg::CONS_ELEMS_PER_THREAD>([&](auto i) {
      cuda::static_for<vectorWidth>([&](auto j) {
        clear(accumulators[i][j]);
      });
    });
    int ticker = 0;
    int chunkIdx = 0;
    constexpr InplaceZero<AccumType> clear{};
    for (int globalStage = 0; globalStage < totalStages; ++globalStage) {
      ticker += 1;
      const auto stage = globalStage % Cfg::PIPE_STAGES;
      // 0. wait until buffer is filled
      if (laneId == 0) {
        auto* __restrict__ fP = flags + (stage * Cfg::CONSUMER_WARPS + consId);
        const cuda::atomic_ref<uint32_t, cuda::thread_scope_block> flag{*fP};
        auto isStageFull = flag.load(cuda::memory_order_relaxed) == full;
        while (!isStageFull) {
          isStageFull = flag.load(cuda::memory_order_relaxed) == full;
        }
        cuda::std::ignore = flag.load(cuda::memory_order_acquire);
      }
      __syncwarp();
      // 1. drain smem buffer to registers
      cuda::static_for<Cfg::CONS_ELEMS_PER_THREAD>([&](auto i) {
        const int offset = stage * Cfg::STAGE_BYTES + (i * Cfg::CONSUMER_THREADS + tId);
        stash[i] = vW[offset];
      });
      __syncwarp();
      // 2. eagerly notify that buffer is free
      if (laneId == 0) {
        auto* __restrict__ fP = flags + (stage * Cfg::CONSUMER_WARPS + consId);
        const cuda::atomic_ref<uint32_t, cuda::thread_scope_block> flag{*fP};
        flag.store(empty, cuda::memory_order_release);
      }
      __syncwarp(); // <- unnecessary
      // 3. reduce in-place to accumulators
      cuda::static_for<Cfg::CONS_ELEMS_PER_THREAD>([&](auto i) {
        AVT val{};
        cuda::static_for<val.size()>([&](auto j) {
          val[j] = loadConv(stash[i][j]);
        });
        op(accumulators[i], val); // convert to accumulator type
      });
      // 4. if applicable, store to local gmem directly
      if (ticker == redArgs.world) {
        ticker = 0;
        // store to gmem
        cuda::static_for<Cfg::CONS_ELEMS_PER_THREAD>([&](auto i) {
          VT resultRaw{};
          cuda::static_for<resultRaw.size()>([&](auto j) {
            resultRaw[j] = storeConv(accumulators[i][j]);
          });
          const size_t offset = (static_cast<size_t>(chunkIdx) * Cfg::STAGE_BYTES) * (i * Cfg::CONSUMER_THREADS + tId);
          vD[offset] = resultRaw;
        });
        chunkIdx++;
        // clear
        cuda::static_for<Cfg::CONS_ELEMS_PER_THREAD>([&](auto i) {
          cuda::static_for<vectorWidth>([&](auto j) {
            clear(accumulators[i][j]);
          });
        });
      }
    }
  }
}

// GMEM (local) -> GMEM(remote)
template<typename Config_>
struct suture::Atom<800, Config_> {
  using Config = tendon::PipelineConfig<Config_>;
  static constexpr int SMEM_SIZE = cute::max(Config::PIPELINE_BYTES, Config::RED_SMEM_BYTES);
  static constexpr int THREADS = Config::THREADS;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config_::GMEM_ACCESS_ALIGNMENT_BYTES;
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
      fascia::putOp<OpCfg>(src, dst, bytes);
      return;
    }
    constexpr int VectorWidth = Config::ALIGNMENT_BYTES / sizeof(AT);
    using VT = cutlass::AlignedArray<AT, VectorWidth, Config::ALIGNMENT_BYTES>;
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const int stages = static_cast<int>(bytes / Config::STAGE_BYTES);
    // priming
    cuda::static_for<Config::PIPE_STAGES>([&vW, &vS](auto i) {
      cuda::static_for<Config::ELEMS_PER_THREAD>([&i, &vW, &vS](auto j) {
        const int slot = ((i * Config::ELEMS_PER_THREAD + j) * Config::THREADS) + threadIdx.x;
        // async gmem -> smem
        cpAsync(vW + slot, vS + slot);
      });
      cpAsyncCommit();
    });
    VT reginald[Config::ELEMS_PER_THREAD];
    // steady state
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
        cpAsync(vW + csW, vS + slot);
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

  // throughput-regime
  template<typename Element>
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
    // assert(__isShared(typedWorkspace));
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    using RedOp = ArrayInplaceSum<800>;
    // throughput regime
    auto* __restrict__ flags = reinterpret_cast<uint32_t*>(workspace + Config::RED_PIPELINE_BYTES);
    const auto warpId = threadIdx.x / WARP_SIZE;
    constexpr auto flagsLength = Config::PIPE_STAGES * Config::CONSUMER_WARPS;
    for (int i = threadIdx.x; i < flagsLength; i += Config::THREADS) {
      flags[threadIdx.x] = empty;
    }
    __syncthreads();
    if (redArgs.putBlock) {
      // transfer
      // 0. sync with others.
      syncRelaxed(
        redArgs.syncRemoteOffset,
        redArgs.syncLocalOffset,
        redArgs.flag,
        redArgs.arrivals);
      // 1. Do put
      putAsync(redArgs.redPut, redArgs.srcPut, redArgs.bytesPut, workspace);
      // 2. Notify peer
      __syncthreads();
      if (!threadIdx.x) {
        const cuda::atomic_ref<uint, cuda::thread_scope_device> s{*redArgs.sigCounter};
        if (s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == redArgs.superBlockSize) {
          s.store(0, cuda::memory_order_relaxed);
          auto* __restrict__ signal = redArgs.putSignals;
          const cuda::atomic_ref<uint32_t, cuda::thread_scope_system> rS{*(signal)};
          rS.store(redArgs.flag, cuda::memory_order_release);
        }
      }
      __syncwarp();
    }
    for (int i = static_cast<int>(threadIdx.x) + 1; i < redArgs.world; i += Config::THREADS) {
      const auto peer = (i + redArgs.rank) % redArgs.world;
      auto* __restrict__ signal = redArgs.signals + peer;
      cuda::atomic_ref<uint32_t, cuda::thread_scope_system> s{*signal};
      auto isHere = s.load(cuda::memory_order_relaxed) == redArgs.flag;
      while (!isHere) {
        isHere = s.load(cuda::memory_order_relaxed) == redArgs.flag;
      }
      cuda::std::ignore = s.load(cuda::memory_order_acquire);
    }
    __syncthreads();
    if (redArgs.bytesRed < Config::RED_PIPELINE_BYTES) {
      fascia::reduce<Config_, RedOp, Element>(redArgs);
      return;
    }
    // 2. Do warp-specialized reduction
    const int totalStages = static_cast<int>(redArgs.bytesRed / static_cast<size_t>(Config::RED_STAGE_BYTES));
    if (warpId < Config::PRODUCER_WARPS) {
      tendon::redProducer<Config>(redArgs, workspace, flags, totalStages, warpId, threadIdx.x);
    }
    else {
      tendon::redConsumer<Config, RedOp, Element>(redArgs, workspace, flags,
        warpId - Config::PRODUCER_WARPS, threadIdx.x - Config::PRODUCER_THREADS);
    }
    // residue
    const auto cutoff = static_cast<size_t>(totalStages) * Config::RED_STAGE_BYTES;
    if (redArgs.bytesRed > cutoff) {
      const auto* __restrict__ src = redArgs.src + cutoff;
      const auto* __restrict__ srcRed = redArgs.srcRed + cutoff;
      auto* __restrict__ dst = redArgs.dst + cutoff;
      const auto bytesRed = redArgs.bytesRed - cutoff;
      fascia::reduce<Config, RedOp, Element>(redArgs, src, srcRed, dst, bytesRed);
    }
  }

  // latency-regime
  template<typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceLRArgs& redArgs, Element* __restrict__ const&) {
    using RedOp = ArrayInplaceSum<800>;
    fascia::reduce<Config, RedOp, Element>(redArgs);
  }

  __device__ __forceinline__
  static void flush() {}

  __device__ __forceinline__
  static void fence() {
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
  }
};
#endif //SUTURE_TENDON_CUH