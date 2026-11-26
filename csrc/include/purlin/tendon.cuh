//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_TENDON_CUH
#define PURLIN_TENDON_CUH
#include "static_for.cuh"
#include <cuda/cmath>

#include "atom.cuh"
#include "base.cuh"
#include "fascia.cuh"
#include "math.cuh"
#include "pipeline.cuh"

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

  // cp.async.commit_group: close the current group on this thread
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

  template<typename Config, typename Value>
  struct PipelineWorkspace {
    using Values = AlignedArray<Value, Config::ELEMS_PER_THREAD, alignof(Value)>;
    Value* __restrict__ data;

    __device__ __forceinline__
    void prefetch(int row, const Value* source) const {
      auto* entries = threadRow(row);
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        cpAsync(entries + i * Config::THREADS, source + i * Config::THREADS);
      });
    }

    __device__ __forceinline__
    Values refill(int row, const Value* source) const {
      auto* entries = threadRow(row);
      Values values;
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        const auto offset = i * Config::THREADS;
        values[i] = entries[offset];
        cpAsync(entries + offset, source + offset);
      });
      return values;
    }

    __device__ __forceinline__
    Values read(int row) const {
      const auto* entries = threadRow(row);
      Values values;
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        values[i] = entries[i * Config::THREADS];
      });
      return values;
    }

  private:
    __device__ __forceinline__
    Value* threadRow(int row) const {
      return data + row * Config::ELEMS_PER_THREAD * Config::THREADS + threadIdx.x;
    }
  };

  template<typename Config, typename Value, typename Src, typename Dest>
  struct CopyPipelineOp : PipelineOperation {
    const Src& source;
    const Dest& destination;
    PipelineWorkspace<Config, Value> workspace;

    __device__ __forceinline__
    CopyPipelineOp(const Src& src, const Dest& dst, Value* shared) :
      source(src), destination(dst), workspace{shared} {}

    __device__ __forceinline__
    void advance() const {}

    __device__ __forceinline__
    void prefetch(int row, int stage) const {
      workspace.prefetch(row, stageAddress<const Value>(source, stage));
    }

    __device__ __forceinline__
    auto refill(int row, int stage) const {
      return workspace.refill(row, stageAddress<const Value>(source, stage));
    }

    __device__ __forceinline__
    auto read(int row) const { return workspace.read(row); }

    __device__ __forceinline__
    void commit() const { cpAsyncCommit(); }

    template<int N>
    __device__ __forceinline__
    void wait() const { cpAsyncWait<N>(); }

    template<typename Values>
    __device__ __forceinline__
    void consume(int completedStage, const Values& values) const {
      auto* output = stageAddress<Value>(destination, completedStage);
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        output[i * Config::THREADS] = values[i];
      });
    }

  private:
    template<typename StageValue, typename Args>
    __device__ __forceinline__
    StageValue* stageAddress(const Args& args, int stage) const {
      const auto index = stage % args.n;
      const auto tile = static_cast<size_t>(stage / args.n);
      auto* base = args.advance(index);
      return reinterpret_cast<StageValue*>(base + tile * Config::STAGE_BYTES) + threadIdx.x;
    }
  };

  template<typename Config, typename RedOp, typename Element>
  struct ReducePipelineOp : PipelineOperation {
    using VE = cuda::std::conditional_t<
      (Config::ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Config::ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<ReduceAccumType<Element>>::type,
      ReduceAccumType<Element>>;
    static constexpr int VECTOR_WIDTH = Config::ALIGNMENT_BYTES / sizeof(VE);
    using Accumulator = AlignedArray<AccumType, VECTOR_WIDTH>;
    using VERaw = DataToRawType<VE>::type;
    using Value = AlignedArray<VERaw, VECTOR_WIDTH>;
    static constexpr int STAGE_ELEMENTS = Config::STAGE_BYTES / sizeof(Value);
    static_assert(cuda::std::is_trivially_copyable_v<Value>);

    const ReduceTRArgs& redArgs;
    Value* __restrict__ destination;
    PipelineWorkspace<Config, Value> workspace;
    Accumulator accumulators[Config::ELEMS_PER_THREAD];
    int ticker;
    int chunkIdx;

    __device__ __forceinline__
    ReducePipelineOp(const ReduceTRArgs& args, Value* dst, Value* shared) :
      redArgs(args), destination(dst), workspace{shared}, ticker(0), chunkIdx(0) {}

    __device__ __forceinline__
    void clearAccumulators() {
      constexpr typename RedOp::template Identity<AccumType> clear{};
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        purlin::static_for<VECTOR_WIDTH>([&](auto j) {
          clear(accumulators[i][j]);
        });
      });
    }

    __device__ __forceinline__
    void advance() {
      ++ticker;
    }

    __device__ __forceinline__
    void prefetch(int row, int stage) const {
      workspace.prefetch(row, stageSource(stage));
    }

    __device__ __forceinline__
    auto refill(int row, int stage) const {
      return workspace.refill(row, stageSource(stage));
    }

    __device__ __forceinline__
    auto read(int row) const { return workspace.read(row); }

    __device__ __forceinline__
    void commit() const {
      cpAsyncCommit();
    }

    template<int N>
    __device__ __forceinline__
    void wait() const {
      cpAsyncWait<N>();
    }

    template<typename Values>
    __device__ __forceinline__
    void consume(const int, const Values& values) {
      constexpr Converter<AccumType, VE> loadConv{};
      constexpr Converter<VERaw, AccumType> storeConv{};
      constexpr RedOp op{};
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        Accumulator value{};
        purlin::static_for<VECTOR_WIDTH>([&](auto j) {
          value[j] = loadConv(values[i][j]);
        });
        op(accumulators[i], value);
      });

      if (ticker == redArgs.world) {
        ticker = 0;
        purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          Value result{};
          purlin::static_for<VECTOR_WIDTH>([&](auto j) {
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

  private:
    __device__ __forceinline__
    const Value* stageSource(int stage) const {
      const auto peer = stage % redArgs.world;
      const auto tile = static_cast<size_t>(stage / redArgs.world);
      const auto* source = reinterpret_cast<const Value*>(redArgs.sources[peer] + redArgs.residualOffset);
      return source + tile * Config::ELEMS_PER_THREAD * Config::THREADS + threadIdx.x;
    }
  };
}

template<typename Config_>
struct purlin::Atom<800, Config_> {
  static_assert(Config_::MEMTYPE == MemType::unicast);
  using BaseConfig = Config_;
  using Config = tendon::PipelineConfig<Config_>;
  static constexpr int NARCH = 800;
  static constexpr int COPY_PIPELINE_BYTES = Config::PIPELINE_BYTES;
  static constexpr int RED_PIPELINE_BYTES = COPY_PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_SMEM_BYTES = Config::PIPELINE_SMEM_BYTES;
  static constexpr int RED_PIPELINE_SMEM_BYTES = COPY_PIPELINE_SMEM_BYTES;
  static constexpr int THREADS = Config::THREADS;
  static constexpr int WARPS = Config::WARPS;
  static constexpr int STAGE_BYTES = Config::STAGE_BYTES;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config_::GMEM_ACCESS_ALIGNMENT_BYTES;

  template<typename Dest, typename Src, typename Workspace>
  __device__ __forceinline__
  static void copy(const Dest& dst, const Src& src, const size_t& bytes, const Workspace& workspace)
    requires ((cuda::std::is_base_of_v<CopyArgs, Dest> || cuda::std::is_convertible_v<const Dest&, cuda::std::byte*>)
      && (cuda::std::is_base_of_v<CopyArgs, Src> || cuda::std::is_convertible_v<const Src&, const cuda::std::byte*>)
      && cuda::std::is_same_v<Workspace, cuda::std::byte*>) {
    const auto& dstArgs = asCopyArgs<cuda::std::byte*>(dst);
    const auto& srcArgs = asCopyArgs<const cuda::std::byte*>(src);
    using AT = AlignedType<Config::ALIGNMENT_BYTES>::type;
    constexpr int VectorWidth = Config::ALIGNMENT_BYTES / sizeof(AT);
    using VT = AlignedArray<AT, VectorWidth, Config::ALIGNMENT_BYTES>;
    const auto roundedBytes = cuda::round_down(bytes, Config::STAGE_BYTES);
    const auto stagesPerSource = roundedBytes / Config::STAGE_BYTES;
    const auto totalStages = stagesPerSource * srcArgs.n;
    if (totalStages < Config::PIPE_STAGES) {
      Atom<700, Config_>::copy(dstArgs, srcArgs, bytes, workspace);
      return;
    }
    using Operation = tendon::CopyPipelineOp<Config, VT, cuda::std::remove_cvref_t<decltype(srcArgs)>,
      cuda::std::remove_cvref_t<decltype(dstArgs)>>;
    Operation operation{srcArgs, dstArgs, reinterpret_cast<VT*>(workspace)};
    Pipeline<Config>::run(static_cast<int>(totalStages), operation);
    if (bytes > roundedBytes) {
      Atom<700, Config_>::copy(dstArgs.withOffset(roundedBytes), srcArgs.withOffset(roundedBytes),
        bytes - roundedBytes, workspace);
    }
  }

  template<
    ReduceResult result,
    ReduceOp ro = ReduceOp::add,
    typename RedOp = typename LoweredReduceOp<ro, NARCH>::type,
    typename Element
  >
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
    static_assert(result == ReduceResult::unicast);
    // assert(__isShared(typedWorkspace));
    // throughput regime
    const auto roundedBytes = cuda::round_down(redArgs.bytesRed, Config::STAGE_BYTES);
    const auto stagesPerPeer = static_cast<int>(roundedBytes / Config::STAGE_BYTES);
    const auto totalStages = stagesPerPeer * redArgs.world;
    if (totalStages < Config::PIPE_STAGES) {
      Atom<700, Config_>::template reduce<result, ro, RedOp>(redArgs, typedWorkspace);
      return;
    }
    using Operation = tendon::ReducePipelineOp<Config, RedOp, Element>;
    using Value = typename Operation::Value;
    Operation operation{redArgs, reinterpret_cast<Value*>(redArgs.dst), reinterpret_cast<Value*>(typedWorkspace)};
    operation.clearAccumulators();
    Pipeline<Config>::run(totalStages, operation);

    if (redArgs.bytesRed > roundedBytes) {
      Atom<700, Config_>::template reduce<result, ro, RedOp>(redArgs.withOffset(roundedBytes), typedWorkspace);
    }
  }
};
#endif //PURLIN_TENDON_CUH
