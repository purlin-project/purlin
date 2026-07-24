//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_TENDON_CUH
#define PURLIN_TENDON_CUH
#include <cuda/cmath>

#include "atom.cuh"
#include "base.cuh"
#include "math.cuh"

namespace purlin {
  template <typename Element>
  __device__ __forceinline__
  void cpAsync(Element* __restrict__ const& smem_ptr, const Element* __restrict__ const& gmem_ptr) {
    constexpr int Size = alignof(Element);
    static_assert(sizeof(Element) == alignof(Element));
    static_assert(Size == 4 || Size == 8 || Size == 16, "cp.async only supports Size in {4, 8, 16}");
    uint32_t sp = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
      "cp.async.cg.shared.global [%0], [%1], %2;\n"
      :
      : "r"(sp), "l"(gmem_ptr), "n"(Size)
      : "memory"
    );
  }
  // cp.async.wait_group N: wait until at most N groups remain outstanding
  template<int N>
  __device__ __forceinline__
  void cpAsyncWait() {
    if constexpr (N == 0) {
      asm volatile("cp.async.wait_all;\n" ::: "memory");
    }
    else {
      static_assert(N >= 0, "cp.async.wait_group argument must be >= 0");
      asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
    }
  }

  // cp.async.commit_group: close the current group on this thread's ring
  __device__ __forceinline__
  void cpAsyncCommit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
}

namespace purlin::tendon {
  template<typename AtomConfig_>
  struct PipelineConfig {
    using AtomConfig = AtomConfig_;
    static constexpr int UNROLL_FACTOR = AtomConfig::UNROLL_FACTOR;
    static constexpr int THREADS = AtomConfig::THREADS;
    static constexpr int WARPS = THREADS / WARP_SIZE;
    static constexpr int ALIGNMENT_BYTES = AtomConfig::ALIGNMENT_BYTES;
    static constexpr int ELEMS_PER_THREAD = AtomConfig::ELEMS_PER_THREAD;
    static constexpr int PIPE_STAGES = AtomConfig::PIPE_STAGES;
    static constexpr int STAGE_BYTES = THREADS * ELEMS_PER_THREAD * ALIGNMENT_BYTES;
    static constexpr int PIPELINE_BYTES = STAGE_BYTES * PIPE_STAGES;
    static constexpr int PIPELINE_SMEM_BYTES = PIPELINE_BYTES;
  };
}

// GMEM (local) -> GMEM(remote)
template<typename Config_>
struct purlin::Atom<800, Config_> {
  using BaseConfig = Config_;
  using Config = tendon::PipelineConfig<Config_>;
  static constexpr Regime REGIME = BaseConfig::REGIME;
  static constexpr int COPY_PIPELINE_BYTES = Config::PIPELINE_BYTES;
  static constexpr int RED_PIPELINE_BYTES = COPY_PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_SMEM_BYTES = Config::PIPELINE_SMEM_BYTES;
  static constexpr int RED_PIPELINE_SMEM_BYTES = COPY_PIPELINE_SMEM_BYTES;
  static constexpr int RED_SMEM_SIZE = COLLECTIVE_STATE_BYTES + (REGIME == Regime::throughput ? RED_PIPELINE_SMEM_BYTES : 0);
  static constexpr int COPY_SMEM_SIZE = COLLECTIVE_STATE_BYTES + (REGIME == Regime::throughput ?COPY_PIPELINE_SMEM_BYTES : 0);
  static constexpr int THREADS = Config::THREADS;
  static constexpr int WARPS = Config::WARPS;
  static constexpr int STAGE_BYTES = Config::STAGE_BYTES;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config_::GMEM_ACCESS_ALIGNMENT_BYTES;
  __device__ __forceinline__
  static void copy(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    //assert(__isShared(workspace));
    using AT = AlignedType<Config::ALIGNMENT_BYTES>::type;
    if (bytes < Config::PIPELINE_BYTES) {
      // use unrolled direct loads as pipelining is not possible
      using OpCfg = fascia::PeerOpConfig<
        Config_,
        ST,
        AT,
        uint32_t
      >;
      fascia::copyOp<OpCfg>(src, dst, bytes);
      return;
    }
    constexpr int VectorWidth = Config::ALIGNMENT_BYTES / sizeof(AT);
    using VT = AlignedArray<AT, VectorWidth, Config::ALIGNMENT_BYTES>;
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
      const int stage_out = i - Config::PIPE_STAGES;
      const int cs = i % Config::PIPE_STAGES;
      cpAsyncWait<Config::PIPE_STAGES - 1>();
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int csW = (cs * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        const long int slot = (static_cast<size_t>(i) * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // smem -> rmem
        reginald[j] = vW[csW];
        // async gmem -> smem prefetch
        cpAsync(vW + csW, vS + slot);
      });
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const long int slot = (stage_out * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // rmem -> gmem
        vD[slot] = reginald[j];
      });
      // commit async transfers from this stage
      cpAsyncCommit();
    }
    // tail
    cuda::static_for<Config::PIPE_STAGES>([&](auto i) {
      const int stage = (stages - Config::PIPE_STAGES) + i;
      const int cs = stage % Config::PIPE_STAGES;
      cpAsyncWait<Config::PIPE_STAGES - 1 - i>();
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int csW = (cs * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // smem -> rmem
        reginald[j] = vW[csW];
      });
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
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
        purlin::store(vD + i, vS[i]);
      }
    }
  }

  template<typename RedOp = ArrayInplaceSum<800>, typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
    // assert(__isShared(typedWorkspace));
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    // throughput regime
    const auto roundedBytes = cuda::round_down(redArgs.bytesRed, Config::STAGE_BYTES);
    const auto stagesPerPeer = static_cast<int>(roundedBytes / Config::STAGE_BYTES);
    const auto totalStages = stagesPerPeer * redArgs.world;
    if (redArgs.bytesRed < Config::STAGE_BYTES || totalStages < Config::PIPE_STAGES) {
      fascia::reduce<Config_, RedOp, Element>(redArgs);
      return;
    }
    using VE = cuda::std::conditional_t<
    (Config::ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Config::ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<ReduceAccumType<Element>>::type,
    ReduceAccumType<Element>>;
    constexpr int vectorWidth = Config::ALIGNMENT_BYTES / sizeof(VE);
    using AVT = AlignedArray<AccumType, vectorWidth>;
    using VER = DataToRawType<VE>::type;
    using VT = AlignedArray<VER, vectorWidth>;
    static_assert(cuda::std::is_trivially_copyable_v<VT>);
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    auto* __restrict__ vD = reinterpret_cast<VT*>(redArgs.dst);
    VT reginald[Config::ELEMS_PER_THREAD];
    AVT accumulators[Config::ELEMS_PER_THREAD];
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VER, AccumType> storeConv{};
    constexpr RedOp op{};
    constexpr InplaceZero<AccumType> clear{};
    constexpr int stageElems = Config::STAGE_BYTES / sizeof(VT);
    const auto worldTrips = redArgs.world / BaseConfig::WORLD_UNROLL;
    const auto cutoff = worldTrips * BaseConfig::WORLD_UNROLL;
    cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
      cuda::static_for<vectorWidth>([&](auto j) {
        clear(accumulators[i][j]);
      });
    });
    int ticker = 0;
    int chunkIdx = 0;
    // priming
    cuda::static_for<Config::PIPE_STAGES>([&](auto i) {
      constexpr int globalStage = i;
      const int dataPeer = globalStage % redArgs.world;
      const auto peerSlot = globalStage / redArgs.world;
      const auto* __restrict__ vSp = reinterpret_cast<const VT*>(redArgs.sources[dataPeer]);
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int slot = ((globalStage * Config::ELEMS_PER_THREAD + j) * Config::THREADS) + threadIdx.x;
        const auto dataSlot = (static_cast<size_t>(peerSlot) * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // async gmem -> smem
        cpAsync(vW + slot, vSp + dataSlot);
      });
      cpAsyncCommit();
    });
    // steady state
    for (int globalStage = Config::PIPE_STAGES; globalStage < totalStages; ++globalStage) {
      ticker++;
      const int stage = globalStage % Config::PIPE_STAGES;
      const int dataPeer = globalStage % redArgs.world;
      const auto peerSlot = globalStage / redArgs.world;
      const auto* __restrict__ vSp = reinterpret_cast<const VT*>(redArgs.sources[dataPeer]);
      cpAsyncWait<Config::PIPE_STAGES - 1>();
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int slot = (stage * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        const auto dataSlot = (static_cast<size_t>(peerSlot) * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // smem -> rmem
        reginald[j] = vW[slot];
        // async gmem -> smem prefetch
        cpAsync(vW + slot, vSp + dataSlot);
      });
      // commit async transfers from this stage
      cpAsyncCommit();
      // reduce
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        AVT val{};
        cuda::static_for<val.size()>([&](auto j) {
          val[j] = loadConv(reginald[i][j]);
        });
        op(accumulators[i], val);
      });
      // check if we need to store results
      if (ticker == redArgs.world) {
        ticker = 0;
        cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          VT resultRaw{};
          cuda::static_for<resultRaw.size()>([&](auto j) {
            resultRaw[j] = storeConv(accumulators[i][j]);
          });
          const size_t offset = (static_cast<size_t>(chunkIdx) * stageElems) + (i * Config::THREADS + threadIdx.x);
          vD[offset] = resultRaw;
        });
        chunkIdx++;
        // clear
        cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          cuda::static_for<vectorWidth>([&](auto j) {
            clear(accumulators[i][j]);
          });
        });
      }
    }
    // tail
    cuda::static_for<Config::PIPE_STAGES>([&](auto remaining) {
      ticker++;
      const int globalStage = (totalStages - Config::PIPE_STAGES) + remaining;
      const int stage = globalStage % Config::PIPE_STAGES;
      cpAsyncWait<Config::PIPE_STAGES - 1 - remaining>();
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int slot = (stage * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // smem -> rmem
        reginald[j] = vW[slot];
      });
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        AVT val{};
        cuda::static_for<val.size()>([&](auto j) {
          val[j] = loadConv(reginald[i][j]);
        });
        op(accumulators[i], val);
      });
      if (ticker == redArgs.world) {
        ticker = 0;
        cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          VT resultRaw{};
          cuda::static_for<resultRaw.size()>([&](auto j) {
            resultRaw[j] = storeConv(accumulators[i][j]);
          });
          const size_t offset = (static_cast<size_t>(chunkIdx) * stageElems) + (i * Config::THREADS + threadIdx.x);
          vD[offset] = resultRaw;
        });
        chunkIdx++;
        // clear
        cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          cuda::static_for<vectorWidth>([&](auto j) {
            clear(accumulators[i][j]);
          });
        });
      }
    });

    // residue
    if (redArgs.bytesRed > roundedBytes) {
      const auto dataCutoff = roundedBytes;
      auto* __restrict__ dst = redArgs.dst + dataCutoff;
      const auto bytesRed = redArgs.bytesRed - dataCutoff;
      fascia::reduce<Config_, RedOp, Element>(redArgs, dst, bytesRed, dataCutoff);
    }
  }

  // latency-regime
  template<DataLayout inputLayout, typename RedOp = ArrayInplaceSum<800>, typename Element>
  __device__ __forceinline__
  static void reduce(const LRArgs& redArgs, Element* __restrict__ const&) {
    fascia::reduce<Config_, RedOp, Element, inputLayout>(redArgs);
  }
};
#endif //PURLIN_TENDON_CUH
