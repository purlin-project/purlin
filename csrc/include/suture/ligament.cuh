/******************************************************************************
* Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_LIGAMENT_CUH
#define SUTURE_LIGAMENT_CUH
#include <cuda/ptx>
#include <cuda/barrier>

#include "base.cuh"
#include "constants.cuh"
#include "copy.cuh"

namespace suture::ligament {
  template<typename AtomConfig_>
  struct PipelineConfig {
    using AtomConfig = AtomConfig_;
    static_assert(AtomConfig::THREADS % WARP_SIZE == 0);
    static constexpr int UNROLL_FACTOR = AtomConfig::UNROLL_FACTOR;
    static constexpr int THREADS = AtomConfig::THREADS;
    static constexpr int WARPS = THREADS / WARP_SIZE;
    static constexpr int ALIGNMENT_BYTES = AtomConfig::ALIGNMENT_BYTES;
    static constexpr int PIPE_STAGES = AtomConfig::PIPE_STAGES;
    static constexpr int STAGE_BYTES = AtomConfig::STAGE_BYTES;
    static constexpr int STAGE_ELEMS = STAGE_BYTES / ALIGNMENT_BYTES;
    static_assert(STAGE_BYTES % (WARP_SIZE * ALIGNMENT_BYTES) == 0);
    static constexpr int ELEMS_PER_THREAD = STAGE_BYTES / (WARP_SIZE * ALIGNMENT_BYTES);
    static constexpr int TOTAL_PIPE_STAGES = PIPE_STAGES * WARPS;
    static constexpr int PIPELINE_BYTES = STAGE_BYTES * TOTAL_PIPE_STAGES;
    static constexpr int PIPELINE_SMEM_BYTES = PIPELINE_BYTES +
      TOTAL_PIPE_STAGES * sizeof(cuda::barrier<cuda::thread_scope_block>);
  };
}

// GMEM (local) -> GMEM(remote)
template<typename Config_>
struct suture::Atom<900, Config_> {
  using BaseConfig = Config_;
  using Config = ligament::PipelineConfig<Config_>;
  using RedAtom = Atom<800,
    Configuration<
        800,
        BaseConfig::THREADS,
        BaseConfig::ALIGNMENT_BYTES,
        Config::TOTAL_PIPE_STAGES,
        BaseConfig::ELEMS_PER_THREAD,
        BaseConfig::UNROLL_FACTOR,
        UNUSED,
        BaseConfig::WORLD_UNROLL,
        BaseConfig::GMEM_ACCESS_ALIGNMENT_BYTES
    >
  >;
  static constexpr int COLL_STATE_BYTES = 2 * MAX_RANKS_PER_DOMAIN * sizeof(cuda::std::byte*);
  static constexpr int RED_PIPELINE_BYTES = RedAtom::RED_PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_BYTES = Config::PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_SMEM_BYTES = Config::PIPELINE_SMEM_BYTES;
  static constexpr int RED_PIPELINE_SMEM_BYTES = RedAtom::RED_PIPELINE_SMEM_BYTES;
  static constexpr int RED_SMEM_SIZE = RED_PIPELINE_SMEM_BYTES + COLL_STATE_BYTES;
  static constexpr int COPY_SMEM_SIZE = COPY_PIPELINE_SMEM_BYTES + COLL_STATE_BYTES;
  static constexpr int THREADS = Config::THREADS;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config_::GMEM_ACCESS_ALIGNMENT_BYTES;

  __device__ __forceinline__
  static void putAsync(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    if (bytes < COPY_PIPELINE_BYTES) {
      using CopyElement = AlignedType<Config::ALIGNMENT_BYTES>::type;
      using OpCfg = fascia::PeerOpConfig<
        BaseConfig,
        ST, // store op
        CopyElement,
        uint32_t
      >;
      // via LSU: GMEM (local) -> RMEM -> GMEM (remote)
      fascia::putOp<OpCfg>(src, dst, bytes);
      return;
    }
    using AT = AlignedType<Config::ALIGNMENT_BYTES>::type;
    constexpr int VectorWidth = Config::ALIGNMENT_BYTES / sizeof(AT);
    using VT = cutlass::AlignedArray<AT, VectorWidth, Config::ALIGNMENT_BYTES>;
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const int totalStages = static_cast<int>(bytes / Config::STAGE_BYTES);
    const auto warpId = threadIdx.x / WARP_SIZE;
    const auto laneId = threadIdx.x % WARP_SIZE;
    const auto stages = totalStages / Config::WARPS + (warpId < totalStages % Config::WARPS);
    auto* __restrict__ barriers = reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>
    (workspace + COPY_PIPELINE_BYTES);
    for (int i = static_cast<int>(threadIdx.x); i < Config::TOTAL_PIPE_STAGES; i += Config::THREADS) {
      // initialize mbarrier objects
      init(barriers + i, 1);
    }
    __syncthreads();
    // priming
    cuda::static_for<Config::PIPE_STAGES>([&](auto i) {
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
    // steady state
    for (int i = Config::PIPE_STAGES; i < stages; ++i) {
      const int globalStage = warpId + i * Config::WARPS;
      const auto outStage = warpId + (i - Config::PIPE_STAGES) * Config::WARPS;
      const int stage = globalStage % Config::TOTAL_PIPE_STAGES;
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        auto* __restrict__ barrier = barriers + stage;
        barrier->arrive_and_wait();
      }
      __syncwarp();
      // drain from smem to rmem
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
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
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        // rmem -> gmem
        const auto offset = (Config::STAGE_ELEMS * static_cast<size_t>(outStage)) + (j * WARP_SIZE + laneId);
        vD[offset] = reginald[j];
      });
    }
    // tail
    const auto tailStartSlot = stages - Config::PIPE_STAGES;
    cuda::static_for<Config::PIPE_STAGES>([&](auto i) {
      const auto globalStage = warpId + (tailStartSlot + i) * Config::WARPS;
      const auto stage = globalStage % Config::TOTAL_PIPE_STAGES;
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        auto* __restrict__ barrier = barriers + stage;
        barrier->arrive_and_wait();
      }
      __syncwarp();
      // drain from smem to rmem
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int offset = (Config::STAGE_ELEMS * stage) + (j * WARP_SIZE + laneId);
        reginald[j] = vW[offset];
      });
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        // rmem -> gmem
        const auto offset = (Config::STAGE_ELEMS * static_cast<size_t>(globalStage)) + (j * WARP_SIZE + laneId);
        vD[offset] = reginald[j];
      });
    });
    const auto cutoff = totalStages * Config::STAGE_BYTES;
    if (bytes > cutoff) {
      constexpr auto residueUnrollFactor = 2;
      using CopyElement = AlignedType<Config::ALIGNMENT_BYTES>::type;
      const auto leftover = bytes - cutoff;
      using OpCfg = fascia::PeerOpConfig<
        BaseConfig,
        ST, // store op
        CopyElement,
        uint32_t,
        residueUnrollFactor,
        Config::THREADS
      >;
      // via LSU: GMEM (local) -> RMEM -> GMEM (remote)
      fascia::putOp<OpCfg>(src + cutoff, dst + cutoff, leftover);
    }
  }

  __device__ __forceinline__
  static void put(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    putAsync(dst, src, bytes, workspace);
  }

  // latency-regime
  template<typename Element>
  __device__ __forceinline__
  static void reduce(const LRArgs& redArgs, Element* __restrict__ const&) {
    using RedOp = ArrayInplaceSum<900>;
    fascia::reduce<Config, RedOp, Element>(redArgs);
  }

  template<typename RedOp = ArrayInplaceSum<900>, typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
    RedAtom::template reduce<RedOp>(redArgs, typedWorkspace);
  }
};
#endif //SUTURE_LIGAMENT_CUH
