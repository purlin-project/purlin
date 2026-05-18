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

namespace suture::tendon {
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
struct suture::Atom<800, Config_> {
  using BaseConfig = Config_;
  using Config = tendon::PipelineConfig<Config_>;
  static constexpr Regime REGIME = BaseConfig::REGIME;
  static constexpr int COLL_STATE_BYTES = 2 * MAX_RANKS_PER_DOMAIN * sizeof(cuda::std::byte*);
  static constexpr int COPY_PIPELINE_BYTES = Config::PIPELINE_BYTES;
  static constexpr int RED_PIPELINE_BYTES = COPY_PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_SMEM_BYTES = Config::PIPELINE_SMEM_BYTES;
  static constexpr int RED_PIPELINE_SMEM_BYTES = COPY_PIPELINE_SMEM_BYTES;
  static constexpr int RED_SMEM_SIZE = COLL_STATE_BYTES + (REGIME == Regime::throughput ? RED_PIPELINE_SMEM_BYTES : 0);
  static constexpr int COPY_SMEM_SIZE = COLL_STATE_BYTES + (REGIME == Regime::throughput ?COPY_PIPELINE_SMEM_BYTES : 0);
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
        Config_,
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
        suture::store(vD + i, vS[i]);
      }
    }
  }

  __device__ __forceinline__
  static void put(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    putAsync(dst, src, bytes, workspace);
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
    (Config::ALIGNMENT_BYTES > sizeof(Element)), typename Element2<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Config::ALIGNMENT_BYTES > sizeof(Element)), typename Element2<ReduceAccumType<Element>>::type, ReduceAccumType<Element>>;
    constexpr int vectorWidth = Config::ALIGNMENT_BYTES / sizeof(VE);
    using AVT = cutlass::AlignedArray<AccumType, vectorWidth>;
    using VER = DataToRawType<VE>::type;
    using VT = cutlass::AlignedArray<VER, vectorWidth>;
    static_assert(cuda::std::is_trivially_copyable_v<VT>);
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    auto* __restrict__ vD = reinterpret_cast<VT*>(redArgs.dst);
    VT reginald[Config::ELEMS_PER_THREAD];
    AVT accumulators[Config::ELEMS_PER_THREAD];
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VE, AccumType> storeConv{};
    constexpr RedOp op{};
    constexpr InplaceZero<AccumType> clear{};
    constexpr int stageElems = Config::STAGE_BYTES / sizeof(VT);
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
        op(accumulators[i], val); // convert to accumulator type
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
      const auto cutoff = roundedBytes;
      auto* __restrict__ dst = redArgs.dst + cutoff;
      const auto bytesRed = redArgs.bytesRed - cutoff;
      fascia::reduce<Config_, RedOp, Element>(redArgs, dst, bytesRed, cutoff);
    }
  }

  // latency-regime
  template<InputLayout iLayout, typename Element>
  __device__ __forceinline__
  static void reduce(const LRArgs& redArgs, Element* __restrict__ const&) {
    using RedOp = ArrayInplaceSum<800>;
    fascia::reduce<Config_, RedOp, Element, iLayout>(redArgs);
  }
};
#endif //SUTURE_TENDON_CUH
