/******************************************************************************
* Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_LIGAMENT_CUH
#define PURLIN_LIGAMENT_CUH
#include "static_for.cuh"
#include <cuda/ptx>
#include <cuda/barrier>

#include "base.cuh"
#include "fascia.cuh"
#include "constants.cuh"
#include "pipeline.cuh"

namespace purlin {
  template<typename Element, ReduceOp ro>
  struct MultimemLdReduce {
  };

  template<>
  struct MultimemLdReduce<__nv_bfloat16, ReduceOp::add> {
    __device__ __forceinline__
    static uint4 loadReduce(const cuda::std::byte* __restrict__ const& mc) {
      uint4 v;
      asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(mc) : "memory");
      return v;
    }
  };

  template<>
  struct MultimemLdReduce<__half, ReduceOp::add> {
    __device__ __forceinline__
    static uint4 loadReduce(const cuda::std::byte* __restrict__ const& mc) {
      uint4 v;
      asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(mc) : "memory");
      return v;
    }
  };

  template<>
  struct MultimemLdReduce<float, ReduceOp::add> {
    __device__ __forceinline__
    static uint4 loadReduce(const cuda::std::byte* __restrict__ const& mc) {
      uint4 v;
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(mc) : "memory");
      return v;
    }
  };

  template<>
  struct MultimemLdReduce<__nv_bfloat16, ReduceOp::max> {
    __device__ __forceinline__
    static uint4 loadReduce(const cuda::std::byte* __restrict__ const& mc) {
      uint4 v;
      asm("multimem.ld_reduce.relaxed.sys.global.max.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(mc) : "memory");
      return v;
    }
  };

  template<>
  struct MultimemLdReduce<__half, ReduceOp::max> {
    __device__ __forceinline__
    static uint4 loadReduce(const cuda::std::byte* __restrict__ const& mc) {
      uint4 v;
      asm("multimem.ld_reduce.relaxed.sys.global.max.v4.f16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(mc) : "memory");
      return v;
    }
  };

  template<typename Element>
  struct MultimemStore {
  };

  template<>
  struct MultimemStore<__nv_bfloat16> {
    __device__ __forceinline__
    static void store(cuda::std::byte* __restrict__ const& mc, const uint4& v) {
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1, %2, %3, %4};"
        :: "l"(mc), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
  };

  template<>
  struct MultimemStore<__half> {
    __device__ __forceinline__
    static void store(cuda::std::byte* __restrict__ const& mc, const uint4& v) {
      asm volatile("multimem.st.relaxed.sys.global.v4.f16x2 [%0], {%1, %2, %3, %4};"
        :: "l"(mc), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
  };

  template<>
  struct MultimemStore<float> {
    __device__ __forceinline__
    static void store(cuda::std::byte* __restrict__ const& mc, const uint4& v) {
      asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(mc), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
  };
}

namespace purlin::ligament {
  template<typename Config, typename Element, ReduceResult result,
    ReduceOp ro = ReduceOp::add>
  __device__ __forceinline__
  static void multimemReduce(const ReduceTRArgs& redArgs,
    const int& tid = static_cast<int>(threadIdx.x)) {
    constexpr size_t accessBytes = Config::ALIGNMENT_BYTES;
    static_assert(accessBytes == 16, "multimem is implemented only for 16-byte accesses currently");
    constexpr int threads = Config::THREADS;
    constexpr int depth = Config::MM_DEPTH;
    auto* __restrict__ const mcBase = redArgs.mcSource + redArgs.residualOffset;
    auto* __restrict__ const vDst = reinterpret_cast<uint4*>(redArgs.dst);
    const auto accesses = redArgs.bytesRed / accessBytes;
    const auto trips = accesses / (threads * depth);
    for (size_t i = 0; i < trips; ++i) {
      const auto tripBase = (i * depth) * threads + tid;
      uint4 values[depth];
      purlin::static_for<depth>([&](auto j) {
        values[j] = MultimemLdReduce<Element, ro>::loadReduce(
          mcBase + (tripBase + j * threads) * accessBytes);
      });
      purlin::static_for<depth>([&](auto j) {
        if constexpr (result == ReduceResult::multicast) {
          MultimemStore<Element>::store(
            mcBase + (tripBase + j * threads) * accessBytes, values[j]);
        }
        else {
          vDst[tripBase + j * threads] = values[j];
        }
      });
    }
    const auto cutoff = trips * threads * depth;
    for (size_t idx = cutoff + tid; idx < accesses; idx += threads) {
      const auto value = MultimemLdReduce<Element, ro>::loadReduce(mcBase + idx * accessBytes);
      if constexpr (result == ReduceResult::multicast) {
        MultimemStore<Element>::store(mcBase + idx * accessBytes, value);
      }
      else {
        vDst[idx] = value;
      }
    }
  }

  template<typename AtomConfig_>
  struct PipelineConfig {
    using AtomConfig = AtomConfig_;
    static_assert(AtomConfig::THREADS % WARP_SIZE == 0);
    static constexpr int UNROLL_FACTOR = AtomConfig::UNROLL_FACTOR;
    static constexpr int THREADS = AtomConfig::THREADS;
    static constexpr int WARPS = THREADS / WARP_SIZE;
    static constexpr int ALIGNMENT_BYTES = AtomConfig::ALIGNMENT_BYTES;
    static constexpr int PIPE_STAGES = AtomConfig::PIPE_STAGES;
    static constexpr int ELEMS_PER_THREAD = AtomConfig::ELEMS_PER_THREAD * WARPS;
    static constexpr int STAGE_BYTES = WARP_SIZE * ELEMS_PER_THREAD * ALIGNMENT_BYTES;
    static constexpr int STAGE_ELEMS = STAGE_BYTES / ALIGNMENT_BYTES;
    static constexpr int PIPELINE_BYTES = STAGE_BYTES * PIPE_STAGES;
    static constexpr int PIPE_STAGES_PER_WARP = PIPE_STAGES / WARPS;
    // experimental TMA copy reserves one block-scoped CUDA barrier per pipeline stage.
    //static constexpr int PIPELINE_SMEM_BYTES = PIPELINE_BYTES + PIPE_STAGES * sizeof(cuda::barrier<cuda::thread_scope_block>);
    static constexpr int PIPELINE_SMEM_BYTES = PIPELINE_BYTES;
  };

  template<typename Config>
  struct WarpPipelineConfig {
    static constexpr int THREADS = WARP_SIZE;
    static constexpr int PIPE_STAGES = Config::PIPE_STAGES_PER_WARP;
    static constexpr int ELEMS_PER_THREAD = Config::ELEMS_PER_THREAD;
  };

  template<typename Config, typename Value_>
  struct CopyPipelineOp : PipelineOperation {
    using Value = Value_;
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

    const cuda::std::byte* __restrict__ source;
    Value* __restrict__ destination;
    cuda::std::byte* __restrict__ workspace;
    Barrier* __restrict__ barriers;
    const unsigned int warpId;
    const unsigned int laneId;
    int pendingStage = 0;
    int completedStage = -1;

    __device__ __forceinline__
    CopyPipelineOp(const cuda::std::byte* __restrict__ src,
      Value* __restrict__ dst, cuda::std::byte* __restrict__ warpWorkspace,
      Barrier* __restrict__ warpBarriers, unsigned int warp, unsigned int lane) :
      source(src), destination(dst), workspace(warpWorkspace), barriers(warpBarriers),
      warpId(warp), laneId(lane) {}

    template<typename Stage>
    __device__ __forceinline__
    auto prepare(Stage stage) const { return stage; }

    __device__ __forceinline__
    void advance() { ++completedStage; }

    template<typename Stage, typename ElementIdx>
    __device__ __forceinline__
    void prefetch(Value* __restrict__ const&, Stage stage, ElementIdx) {
      // TMA replaces a whole stage. Record the request once; commit issues it
      // after the scheduler has read every value from the reused stage.
      if constexpr (ElementIdx::value == 0) pendingStage = stage;
    }

    __device__ __forceinline__
    void commit() const {
      // Prime has no readers. A reused stage needs every lane to finish reading.
      if (pendingStage >= Config::PIPE_STAGES_PER_WARP) __syncwarp();
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        const int slot = pendingStage % Config::PIPE_STAGES_PER_WARP;
        auto& barrier = barriers[slot * Config::WARPS];
        const auto globalStage = warpId + static_cast<size_t>(pendingStage) * Config::WARPS;
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          workspace + slot * Config::STAGE_BYTES,
          source + globalStage * Config::STAGE_BYTES,
          Config::STAGE_BYTES,
          cuda::device::barrier_native_handle(barrier));
        cuda::device::barrier_expect_tx(barrier, Config::STAGE_BYTES);
      }
    }

    template<int N>
    __device__ __forceinline__
    void wait() const {
      const int slot = completedStage % Config::PIPE_STAGES_PER_WARP;
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        barriers[slot * Config::WARPS].arrive_and_wait();
      }
      __syncwarp();
    }

    template<typename Values>
    __device__ __forceinline__
    void consume(int stage, const Values& values) const {
      const auto globalStage = warpId + static_cast<size_t>(stage) * Config::WARPS;
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
        destination[globalStage * Config::STAGE_ELEMS + i * WARP_SIZE + laneId] = values[i];
      });
    }
  };

  // Unused TMA-based copy path. Atom<900> instead delegates copy to Atom<800>::copy, which is faster.
  template<typename Config, typename BaseConfig>
  __device__ __forceinline__
  static void copy(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    if (bytes < Config::PIPELINE_BYTES) {
      Atom<700, BaseConfig>::copy(dst, src, bytes, workspace);
      return;
    }
    static_assert(Config::PIPE_STAGES % Config::WARPS == 0);
    using AT = AlignedType<Config::ALIGNMENT_BYTES>::type;
    constexpr int VectorWidth = Config::ALIGNMENT_BYTES / sizeof(AT);
    using VT = AlignedArray<AT, VectorWidth, Config::ALIGNMENT_BYTES>;
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const int totalStages = static_cast<int>(bytes / Config::STAGE_BYTES);
    const auto warpId = cuda::ptx::get_sreg_warpid();
    const auto laneId = cuda::ptx::get_sreg_laneid();
    const auto stages = totalStages / Config::WARPS + (warpId < totalStages % Config::WARPS);
    auto* __restrict__ barriers = reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>
    (workspace + Config::PIPELINE_BYTES);
    for (int i = static_cast<int>(laneId); i < Config::PIPE_STAGES_PER_WARP; i += WARP_SIZE) {
      const auto stage = warpId + i * Config::WARPS;
      auto& barrier = *(barriers + stage);
      cuda::ptx::mbarrier_inval(cuda::device::barrier_native_handle(barrier));
      init(barriers + stage, 1);
    }
    __syncwarp();
    // Each warp has a contiguous ring, indexed by laneId within the pipeline.
    constexpr int warpPipelineBytes = Config::PIPE_STAGES_PER_WARP * Config::STAGE_BYTES;
    auto* __restrict__ warpWorkspace = workspace + warpId * warpPipelineBytes;
    CopyPipelineOp<Config, VT> operation{src, vD, warpWorkspace, barriers + warpId, warpId, laneId};
    Pipeline<WarpPipelineConfig<Config>>::run(warpWorkspace, stages, operation, laneId);
    const auto cutoff = totalStages * Config::STAGE_BYTES;
    if (bytes > cutoff) {
      Atom<700, BaseConfig>::copy(dst + cutoff, src + cutoff, bytes - cutoff, workspace);
    }
  }
}

template<typename Config_>
struct purlin::Atom<900, Config_> {
  using BaseConfig = Config_;
  using Config = ligament::PipelineConfig<Config_>;
  static constexpr int NARCH = 900;
  using BaseAtom = Atom<800,
    Configuration<
        BaseConfig::THREADS,
        BaseConfig::ALIGNMENT_BYTES,
        BaseConfig::PIPE_STAGES,
        BaseConfig::ELEMS_PER_THREAD,
        BaseConfig::UNROLL_FACTOR,
        BaseConfig::WORLD_UNROLL,
        BaseConfig::GMEM_ACCESS_ALIGNMENT_BYTES
    >
  >;
  static constexpr int RED_PIPELINE_BYTES = BaseAtom::RED_PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_BYTES = Config::PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_SMEM_BYTES = Config::PIPELINE_SMEM_BYTES;
  static constexpr int RED_PIPELINE_SMEM_BYTES =
    BaseConfig::MEMTYPE == MemType::multimem ? 0 : BaseAtom::RED_PIPELINE_SMEM_BYTES;
  static constexpr int THREADS = Config::THREADS;
  static constexpr int WARPS = Config::WARPS;
  static constexpr int STAGE_BYTES = Config::STAGE_BYTES;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config_::GMEM_ACCESS_ALIGNMENT_BYTES;

  __device__ __forceinline__
  static void copy(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    BaseAtom::copy(dst, src, bytes, workspace);
  }

  template<ReduceResult result, ReduceOp ro = ReduceOp::add,
    typename RedOp = typename LoweredReduceOp<ro, NARCH>::type, typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
    if constexpr (BaseConfig::MEMTYPE == MemType::multimem) {
      static_assert(multimemReducible<NARCH, Element, ro>());
      ligament::multimemReduce<BaseConfig, Element, result, ro>(redArgs);
    }
    else {
      BaseAtom::template reduce<ReduceResult::unicast, ro, RedOp>(redArgs, typedWorkspace);
    }
  }
};
#endif //PURLIN_LIGAMENT_CUH
