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

namespace purlin {
  // Each specialization implements an element and reduction-operation pair
  // supported by the PTX multimem instructions. multimemReducible() prevents
  // unsupported pairs from reaching these templates. Every form transfers
  // 16 bytes per instruction. The 16-bit addition forms accumulate in f32,
  // which is the highest precision provided by the switch.
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

  // Writing a reduction result back through multicast does not depend on the
  // reduction operation, but the multimem store instruction does depend on the
  // element type.
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
    const auto* __restrict__ const mcBase = redArgs.mcSource + redArgs.residualOffset;
    auto* __restrict__ const mcOut = redArgs.mcResult + redArgs.residualOffset;
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
            mcOut + (tripBase + j * threads) * accessBytes, values[j]);
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
        MultimemStore<Element>::store(mcOut + idx * accessBytes, value);
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
    // The experimental TMA copy below would also reserve one block-scoped CUDA
    // barrier per pipeline stage. The active copy path does not need that
    // additional shared memory.
    static constexpr int PIPELINE_SMEM_BYTES = PIPELINE_BYTES;
  };
  // Experimental TMA-based copy path. The Atom below currently delegates copy
  // operations to BaseAtom instead of calling this function.
  template<typename Config, typename BaseConfig>
  __device__ __forceinline__
  static void copy(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    if (bytes < Config::PIPELINE_BYTES) {
      // Too small to pipeline: use the generic load/store copy.
      Atom<700, BaseConfig>::copy(dst, src, bytes, workspace);
      return;
    }
    static_assert(Config::PIPE_STAGES % Config::WARPS == 0);
    using AT = AlignedType<Config::ALIGNMENT_BYTES>::type;
    constexpr int VectorWidth = Config::ALIGNMENT_BYTES / sizeof(AT);
    using VT = AlignedArray<AT, VectorWidth, Config::ALIGNMENT_BYTES>;
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const int totalStages = static_cast<int>(bytes / Config::STAGE_BYTES);
    const auto warpId = threadIdx.x / WARP_SIZE;
    const auto laneId = threadIdx.x % WARP_SIZE;
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
    // Prime every pipeline stage with its first asynchronous copy.
    purlin::static_for<Config::PIPE_STAGES_PER_WARP>([&](auto i) {
      const auto stage = warpId + i * Config::WARPS;
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        auto& barrier = *(barriers + stage);
        const auto* __restrict__ sP = src + stage * Config::STAGE_BYTES;
        auto* __restrict dP = workspace + stage * Config::STAGE_BYTES;
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          dP,
          sP,
          Config::STAGE_BYTES,
          cuda::device::barrier_native_handle(barrier));
        cuda::device::barrier_expect_tx(barrier, Config::STAGE_BYTES);
      }
    });
    VT reginald[Config::ELEMS_PER_THREAD];
    // In the steady state, drain one stage while refilling the slot it vacates.
    for (int i = Config::PIPE_STAGES_PER_WARP; i < stages; ++i) {
      const int globalStage = warpId + i * Config::WARPS;
      const auto outStage = warpId + (i - Config::PIPE_STAGES_PER_WARP) * Config::WARPS;
      const int stage = globalStage % Config::PIPE_STAGES;
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        auto* __restrict__ barrier = barriers + stage;
        barrier->arrive_and_wait();
      }
      __syncwarp();
      // Move the completed stage from shared memory into registers.
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int offset = (Config::STAGE_ELEMS * stage) + (j * WARP_SIZE + laneId);
        reginald[j] = vW[offset];
      });
      __syncwarp();
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        auto& barrier = *(barriers + stage);
        const auto* __restrict__ sP = src + globalStage * Config::STAGE_BYTES;
        auto* __restrict dP = workspace + stage * Config::STAGE_BYTES;
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          dP,
          sP,
          Config::STAGE_BYTES,
          cuda::device::barrier_native_handle(barrier));
        cuda::device::barrier_expect_tx(barrier, Config::STAGE_BYTES);
      }
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        // Write the registered values to the peer's global memory.
        const auto offset = (Config::STAGE_ELEMS * static_cast<size_t>(outStage)) + (j * WARP_SIZE + laneId);
        vD[offset] = reginald[j];
      });
    }
    // Drain the stages that remain after the final refill.
    const auto tailStartSlot = stages - Config::PIPE_STAGES_PER_WARP;
    purlin::static_for<Config::PIPE_STAGES_PER_WARP>([&](auto i) {
      const auto globalStage = warpId + (tailStartSlot + i) * Config::WARPS;
      const auto stage = globalStage % Config::PIPE_STAGES;
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        auto* __restrict__ barrier = barriers + stage;
        barrier->arrive_and_wait();
      }
      __syncwarp();
      // Move this remaining stage from shared memory into registers.
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int offset = (Config::STAGE_ELEMS * stage) + (j * WARP_SIZE + laneId);
        reginald[j] = vW[offset];
      });
      purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        // Write the registered values to the peer's global memory.
        const auto offset = (Config::STAGE_ELEMS * static_cast<size_t>(globalStage)) + (j * WARP_SIZE + laneId);
        vD[offset] = reginald[j];
      });
    });
    const auto cutoff = totalStages * Config::STAGE_BYTES;
    if (bytes > cutoff) {
      // The bulk copies cover whole stages; the generic copy takes the tail.
      Atom<700, BaseConfig>::copy(dst + cutoff, src + cutoff, bytes - cutoff, workspace);
    }
  }
}

// Hopper Atom: copy data from local global memory to a peer's global memory.
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
  // A multimem reduction travels through the switch and stays in registers, so
  // it needs no shared-memory reduction pipeline. The unicast path retains the
  // shared-memory pipeline provided by BaseAtom.
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
      static_assert(multimemReducible<NARCH, Element, ro>(),
        "the multimem datapath has no mapping for this element/op pair");
      ligament::multimemReduce<BaseConfig, Element, result, ro>(redArgs);
    }
    else {
      BaseAtom::template reduce<ReduceResult::unicast, ro, RedOp>(redArgs, typedWorkspace);
    }
  }
};
#endif //PURLIN_LIGAMENT_CUH
