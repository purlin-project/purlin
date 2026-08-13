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

  enum class PipelineCommit {
    beforeConsume,
    afterConsume
  };

  template<typename Config, typename Value_>
  struct CopyPipelineOp {
    using Value = Value_;
    static constexpr auto COMMIT_ORDER = PipelineCommit::afterConsume;

    const Value* __restrict__ source;
    Value* __restrict__ destination;

    template<typename Stage>
    __device__ __forceinline__
    auto prepare(const Stage stage) const {
      return stage;
    }

    __device__ __forceinline__
    void advance() const {}

    template<typename Stage, typename ElementIdx>
    __device__ __forceinline__
    void prefetch(Value* __restrict__ const& smem,
      const Stage stage, const ElementIdx elementIdx) const {
      const auto dataSlot = (static_cast<size_t>(stage) * Config::ELEMS_PER_THREAD + elementIdx) *
        Config::THREADS + threadIdx.x;
      cpAsync(smem, source + dataSlot);
    }

    template<typename Values>
    __device__ __forceinline__
    void consume(const int completedStage, const Values& values) const {
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        const auto dataSlot = (static_cast<size_t>(completedStage) * Config::ELEMS_PER_THREAD + i) *
          Config::THREADS + threadIdx.x;
        destination[dataSlot] = values[i];
      });
    }
  };

  template<typename Config, typename RedOp, typename Element>
  struct ReducePipelineOp {
    using VE = cuda::std::conditional_t<
      (Config::ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Config::ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<ReduceAccumType<Element>>::type,
      ReduceAccumType<Element>>;
    static constexpr int VECTOR_WIDTH = Config::ALIGNMENT_BYTES / sizeof(VE);
    using Accumulator = AlignedArray<AccumType, VECTOR_WIDTH>;
    using VERaw = DataToRawType<VE>::type;
    using Value = AlignedArray<VERaw, VECTOR_WIDTH>;
    static constexpr auto COMMIT_ORDER = PipelineCommit::beforeConsume;
    static constexpr int STAGE_ELEMENTS = Config::STAGE_BYTES / sizeof(Value);
    static_assert(cuda::std::is_trivially_copyable_v<Value>);

    struct InputStage {
      const Value* source;
      size_t slot;
    };

    const ReduceTRArgs& redArgs;
    Value* __restrict__ destination;
    Accumulator accumulators[Config::ELEMS_PER_THREAD];
    int ticker;
    int chunkIdx;

    __device__ __forceinline__
    ReducePipelineOp(const ReduceTRArgs& redArgs_, Value* __restrict__ const& destination_) :
      redArgs(redArgs_), destination(destination_), ticker(0), chunkIdx(0) {}

    __device__ __forceinline__
    void clearAccumulators() {
      constexpr InplaceZero<AccumType> clear{};
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        cuda::static_for<VECTOR_WIDTH>([&](auto j) {
          clear(accumulators[i][j]);
        });
      });
    }

    template<typename Stage>
    __device__ __forceinline__
    InputStage prepare(const Stage globalStage_) const {
      const auto globalStage = static_cast<int>(globalStage_);
      const auto dataPeer = globalStage % redArgs.world;
      const auto peerSlot = globalStage / redArgs.world;
      return InputStage{
        .source = reinterpret_cast<const Value*>(redArgs.sources[dataPeer]),
        .slot = static_cast<size_t>(peerSlot)
      };
    }

    __device__ __forceinline__
    void advance() {
      ++ticker;
    }

    template<typename ElementIdx>
    __device__ __forceinline__
    void prefetch(Value* __restrict__ const& smem,
      const InputStage& input, const ElementIdx elementIdx) const {
      const auto dataSlot = (input.slot * Config::ELEMS_PER_THREAD + elementIdx) *
        Config::THREADS + threadIdx.x;
      cpAsync(smem, input.source + dataSlot);
    }

    template<typename Values>
    __device__ __forceinline__
    void consume(const int, const Values& values) {
      constexpr Converter<AccumType, VE> loadConv{};
      constexpr Converter<VERaw, AccumType> storeConv{};
      constexpr RedOp op{};
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        Accumulator value{};
        cuda::static_for<VECTOR_WIDTH>([&](auto j) {
          value[j] = loadConv(values[i][j]);
        });
        op(accumulators[i], value);
      });

      if (ticker == redArgs.world) {
        ticker = 0;
        cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          Value result{};
          cuda::static_for<VECTOR_WIDTH>([&](auto j) {
            result[j] = storeConv(accumulators[i][j]);
          });
          const auto offset = static_cast<size_t>(chunkIdx) * STAGE_ELEMENTS +
            (i * Config::THREADS + threadIdx.x);
          destination[offset] = result;
        });
        ++chunkIdx;
        clearAccumulators();
      }
    }
  };

  template<typename Config, typename Operation>
  __device__ __forceinline__
  void runPipeline(cuda::std::byte* __restrict__ const& workspace,
    const int totalStages, Operation& operation) {
    using Value = typename Operation::Value;
    constexpr int stageElements = Config::ELEMS_PER_THREAD * Config::THREADS;
    auto* __restrict__ pipeline = reinterpret_cast<Value*>(workspace);

    // priming
    cuda::static_for<Config::PIPE_STAGES>([&](auto globalStage) {
      const auto input = operation.prepare(globalStage);
      auto* __restrict__ stage = pipeline + globalStage * stageElements;
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        const auto slot = i * Config::THREADS + threadIdx.x;
        operation.prefetch(stage + slot, input, i);
      });
      cpAsyncCommit();
    });

    Value values[Config::ELEMS_PER_THREAD];
    // steady state
    for (int globalStage = Config::PIPE_STAGES; globalStage < totalStages; ++globalStage) {
      operation.advance();
      const auto completedStage = globalStage - Config::PIPE_STAGES;
      const auto circularStage = globalStage % Config::PIPE_STAGES;
      const auto input = operation.prepare(globalStage);
      auto* __restrict__ stage = pipeline + circularStage * stageElements;
      cpAsyncWait<Config::PIPE_STAGES - 1>();
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        const auto slot = i * Config::THREADS + threadIdx.x;
        values[i] = stage[slot];
        operation.prefetch(stage + slot, input, i);
      });
      if constexpr (Operation::COMMIT_ORDER == PipelineCommit::beforeConsume) {
        cpAsyncCommit();
      }
      operation.consume(completedStage, values);
      if constexpr (Operation::COMMIT_ORDER == PipelineCommit::afterConsume) {
        cpAsyncCommit();
      }
    }

    // tail
    cuda::static_for<Config::PIPE_STAGES>([&](auto remaining) {
      operation.advance();
      const auto completedStage = totalStages - Config::PIPE_STAGES + remaining;
      const auto circularStage = completedStage % Config::PIPE_STAGES;
      auto* __restrict__ stage = pipeline + circularStage * stageElements;
      cpAsyncWait<Config::PIPE_STAGES - 1 - remaining>();
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        const auto slot = i * Config::THREADS + threadIdx.x;
        values[i] = stage[slot];
      });
      operation.consume(completedStage, values);
    });
  }
}

// GMEM (local) -> GMEM(remote)
template<typename Config_>
struct purlin::Atom<800, Config_> {
  static_assert(Config_::DATAPATH == Datapath::unicast, "the multimem datapath requires sm90 or newer");
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
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const int stages = static_cast<int>(bytes / Config::STAGE_BYTES);
    tendon::CopyPipelineOp<Config, VT> operation{vS, vD};
    tendon::runPipeline<Config>(workspace, stages, operation);
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

  template<ReduceResult result = ReduceResult::multicast,
    typename RedOp = ArrayInplaceSum<800>, typename Element>
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
    using Operation = tendon::ReducePipelineOp<Config, RedOp, Element>;
    Operation operation{redArgs, reinterpret_cast<typename Operation::Value*>(redArgs.dst)};
    operation.clearAccumulators();
    tendon::runPipeline<Config>(workspace, totalStages, operation);

    // residue
    if (redArgs.bytesRed > roundedBytes) {
      const auto dataCutoff = roundedBytes;
      auto* __restrict__ dst = redArgs.dst + dataCutoff;
      const auto bytesRed = redArgs.bytesRed - dataCutoff;
      fascia::reduce<Config_, RedOp, Element>(redArgs, dst, bytesRed, dataCutoff);
    }
  }

  // latency-regime
  template<DataLayout inputLayout, bool partitioned = false,
    typename RedOp = ArrayInplaceSum<800>, typename Element>
  __device__ __forceinline__
  static void reduce(const LRArgs& redArgs, Element* __restrict__ const&) {
    fascia::reduce<Config_, RedOp, Element, inputLayout, partitioned>(redArgs);
  }
};
#endif //PURLIN_TENDON_CUH
