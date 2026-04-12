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
  // nArch is implicitly 900 in ligament
  constexpr int nArch = 900;
  template<typename AtomConfig_>
  struct PipelineConfig {
    using AtomConfig = AtomConfig_;
    static_assert(AtomConfig::THREADS % WARP_SIZE == 0);
    static constexpr int UNROLL_FACTOR = AtomConfig::UNROLL_FACTOR;
    static constexpr int THREADS = AtomConfig::THREADS;
    static constexpr int PRODUCER_THREADS = AtomConfig::THREADS - WARP_SIZE;
    static_assert(PRODUCER_THREADS % WARP_SIZE == 0);
    static constexpr int ALIGNMENT_BYTES = AtomConfig::ALIGNMENT_BYTES;
    static constexpr int ELEMS_PER_THREAD = AtomConfig::ELEMS_PER_THREAD;
    static constexpr int PIPE_STAGES = AtomConfig::PIPE_STAGES;
    static constexpr int PRODUCER_WARPS = PRODUCER_THREADS / WARP_SIZE;
    static constexpr int TOTAL_PIPE_STAGES = PRODUCER_WARPS * PIPE_STAGES;
    static constexpr int STAGE_BYTES = WARP_SIZE * ALIGNMENT_BYTES * ELEMS_PER_THREAD;
    static constexpr int STAGE_ELEMS = WARP_SIZE * ELEMS_PER_THREAD;
    static constexpr int TOTAL_STAGE_BYTES = STAGE_BYTES * PRODUCER_WARPS;
    static constexpr int PIPELINE_BYTES = TOTAL_STAGE_BYTES * PIPE_STAGES; // bytes in flight at steady state
    static constexpr int SMEM_BYTES = PIPELINE_BYTES + TOTAL_PIPE_STAGES * sizeof(uint);
  };
  template<typename Cfg>
  __device__ __forceinline__
  void putProducer(const int& totalStages,
    uint32_t* __restrict__ const& flags,
    cuda::std::byte* __restrict__ const& stagingBuffers,
    const cuda::std::byte* __restrict__ const& src) {
    static_assert(!cuda::std::is_void_v<Cfg>);
    using AT = AlignedType<Cfg::ALIGNMENT_BYTES>::type;
    constexpr int VectorWidth = Cfg::ALIGNMENT_BYTES / sizeof(AT);
    using VT = cutlass::AlignedArray<AT, VectorWidth, Cfg::ALIGNMENT_BYTES>;
    auto* __restrict__ vW = reinterpret_cast<VT*>(stagingBuffers);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const int prodId = static_cast<int>(threadIdx.x) / WARP_SIZE;
    const int laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
    const int producerStages = totalStages / Cfg::PRODUCER_WARPS + (prodId < (totalStages % Cfg::PRODUCER_WARPS));
    // assert(producerStages >= Cfg::PIPE_STAGES)
    // Stage 1: pipeline priming
    cuda::static_for<Cfg::PIPE_STAGES>([&](auto i) {
      const int stage = prodId + i * Cfg::PRODUCER_WARPS;
      cuda::static_for<Cfg::ELEMS_PER_THREAD>([&](auto j) {
        const int slot = (stage * Cfg::ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
        // async gmem -> smem
        cpAsync(vW + slot, vS + slot);
      });
      cpAsyncCommit();
    });
    // Stage 2: steady state
    for (int i = Cfg::PIPE_STAGES; i < producerStages; ++i) {
      const auto dataStage = prodId + i * Cfg::PRODUCER_WARPS;
      const int stage = dataStage % Cfg::TOTAL_PIPE_STAGES;
      // 1. Wait for _only_ the oldest outstanding cp.async group
      // (the one committed NUM_STAGES_PER_PRODUCER iterations ago) to complete.
      cpAsyncWait<Cfg::PIPE_STAGES - 1>();
      __syncwarp(); // <- lightweight synchronization: benefit of warp granularity rather than CTA's
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) { // <- elect one thread from the warp
        const cuda::atomic_ref<uint32_t, cuda::thread_scope_block> flag{*(flags + stage)};
        // 2. Notify consumer: stage s is ready
        flag.store(full, cuda::memory_order_release);
        // 3. Back-pressure: spin until consumer has drained stage s
        auto isStageEmpty = flag.load(cuda::memory_order_relaxed) == empty;
        while (!isStageEmpty) {
          isStageEmpty = flag.load(cuda::memory_order_relaxed) == empty;
        }
        cuda::std::ignore = flag.load(cuda::memory_order_acquire);
      }
      __syncwarp();
      // 4. Refill stage s
      cuda::static_for<Cfg::ELEMS_PER_THREAD>([&](auto j) {
        const int slot = (stage * Cfg::ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
        const size_t dataSlot = (static_cast<size_t>(dataStage) * Cfg::ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
        // async gmem -> smem
        cpAsync(vW + slot, vS + dataSlot);
      });
      cpAsyncCommit();
    }
    // Stage 3: tail flush
    const auto firstTailStage = (prodId + producerStages * Cfg::PRODUCER_WARPS) % Cfg::TOTAL_PIPE_STAGES;
    cuda::static_for<Cfg::PIPE_STAGES>([&](auto i) {
      constexpr int remaining = Cfg::PIPE_STAGES - 1 - i;
      const auto stage = (firstTailStage + i * Cfg::PRODUCER_WARPS) % Cfg::TOTAL_PIPE_STAGES;
      cpAsyncWait<remaining>();
      __syncwarp();
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        const cuda::atomic_ref<uint32_t, cuda::thread_scope_block> flag{*(flags + stage)};
        flag.store(full, cuda::memory_order_release);
      }
    });
  }
  // multi-threaded
  template<typename Cfg>
  __device__ __forceinline__
  void putConsumer(const int& totalStages,
    uint32_t* __restrict__ const& flags,
    const cuda::std::byte* __restrict__ const& stagingBuffers,
    cuda::std::byte* __restrict__ const& dst) {
    static_assert(!cuda::std::is_void_v<Cfg>);
    const int laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
    // Static stage assignment per lane — only lanes [0, NUM_STAGES) are active.
    const bool active = Cfg::TOTAL_PIPE_STAGES == 1 ?
    cuda::ptx::elect_sync(0xFFFFFFFF) : laneId < Cfg::TOTAL_PIPE_STAGES;
    const int stageId = laneId;
    const auto fullRounds = totalStages / Cfg::TOTAL_PIPE_STAGES;
    const auto residue = totalStages % Cfg::TOTAL_PIPE_STAGES;
    auto* __restrict__ stagingBuffer = stagingBuffers + stageId * Cfg::STAGE_BYTES;
    const cuda::atomic_ref<uint32_t, cuda::thread_scope_block> flag{*(flags + stageId)};

    for (int round = 0; round < fullRounds; ++round) {
      const size_t offset = (static_cast<size_t>(round) * Cfg::TOTAL_PIPE_STAGES + stageId) * Cfg::STAGE_BYTES;
      if (active) {
        // Spin until the producer signals this stage is full
        auto isStageFull = flag.load(cuda::memory_order_relaxed) == full;
        while (!isStageFull) {
          isStageFull = flag.load(cuda::memory_order_relaxed) == full;
        }
        cuda::std::ignore = flag.load(cuda::memory_order_acquire);
        // TMA store: smem → remote HBM
        // Each active lane issues independently for its own disjoint stage.
        cuda::ptx::cp_async_bulk(
        cuda::ptx::space_global, cuda::ptx::space_shared,
        dst + offset, stagingBuffer, Cfg::STAGE_BYTES);
        cuda::ptx::cp_async_bulk_commit_group();

        // Wait until TMA engine has finished reading smem.
        // Each lane tracks its own bulk-async group ring independently.
        cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());

        // signal producer it may refill
        flag.store(empty, cuda::memory_order_release);
      }
      __syncwarp();
    }
    if (Cfg::TOTAL_PIPE_STAGES > 1 && residue) {
      const size_t offset = (fullRounds * Cfg::TOTAL_PIPE_STAGES + stageId) * Cfg::STAGE_BYTES;
      if (laneId < residue) {
        auto isStageFull = flag.load(cuda::memory_order_relaxed) == full;
        while (!isStageFull) {
          isStageFull = flag.load(cuda::memory_order_relaxed) == full;
        }
        cuda::std::ignore = flag.load(cuda::memory_order_acquire);
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global, cuda::ptx::space_shared,
          dst + offset, stagingBuffer, Cfg::STAGE_BYTES);
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
        flag.store(empty, cuda::memory_order_release);
      }
      __syncwarp();
    }
  }

  template<typename Cfg>
  __device__ __forceinline__
  void putProducerTT() {

  }
  template<typename Cfg>
  __device__ __forceinline__
  void putConsumerTT() {

  }

}

// GMEM (local) -> GMEM(remote)
template<typename Config_>
struct suture::Atom<900, Config_> {
  using Config = ligament::PipelineConfig<Config_>;
  static constexpr int SMEM_SIZE = Config::SMEM_BYTES;
  static_assert(ligament::nArch == 900);
  using MaxAlignmentBytes = cuda::std::integral_constant<int, 16>;

  __device__ __forceinline__
  static void putAsync(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    // TODO: Simplifying assumption, relax it
    static_assert(Config::TOTAL_PIPE_STAGES <= WARP_SIZE);
    //assert(__isShared(workspace));
    // 1. if less than threshold, do direct GMEM -> GMEM
    if (bytes < Config::PIPELINE_BYTES) {
      using CopyElement = AlignedType<Config::ALIGNMENT_BYTES>::type;
      using OpCfg = fascia::PeerOpConfig<
        Config,
        ST, // store op
        CopyElement,
        uint32_t
      >;
      // via LSU: GMEM (local) -> RMEM -> GMEM (remote)
      fascia::peerOp<OpCfg>(src, dst, bytes);
      return;
    }
    constexpr auto numWarps = Config::THREADS / WARP_SIZE;
    auto* __restrict__ flags = reinterpret_cast<uint32_t*>(workspace + Config::PIPELINE_BYTES);
    if (threadIdx.x < Config::TOTAL_PIPE_STAGES) {
      flags[threadIdx.x] = 0U;
    }
    __syncthreads();
    const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;

    const int totalStages = bytes / Config::STAGE_BYTES; // assert(totalStages >= Cfg::TOTAL_PIPE_STAGES)
    if (warpId + 1 == numWarps) {
      // last warp is consumer
      ligament::putConsumer<Config>(totalStages, flags, workspace, dst);
      return;
    }
    // producer
    ligament::putProducer<Config>(totalStages, flags, workspace, src);
    // residue
    const auto cutoff = totalStages * Config::STAGE_BYTES;
    if (bytes > cutoff) {
      // high value increases register pressure, low value reduces ILP
      constexpr auto residueUnrollFactor = cute::min(2, Config::ELEMS_PER_THREAD);
      using CopyElement = AlignedType<Config::ALIGNMENT_BYTES>::type;
      const auto leftover = bytes - cutoff;
      using OpCfg = fascia::PeerOpConfig<
        Config,
        ST, // store op
        CopyElement,
        uint32_t,
        residueUnrollFactor,
        Config::PRODUCER_THREADS
      >;
      // via LSU: GMEM (local) -> RMEM -> GMEM (remote)
      fascia::peerOp<OpCfg>(src + cutoff, dst + cutoff, leftover);
    }
  }


  __device__ __forceinline__
  static void putAsyncTT(cuda::std::byte* __restrict__ const& dst,
   const cuda::std::byte* __restrict__ const& src,
   const size_t& bytes,
   cuda::std::byte* __restrict__ const& workspace) {
  }

  __device__ __forceinline__
  static void flush() {
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>());
  }

  __device__ __forceinline__
  static void fence() {
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
  }
};
#endif //SUTURE_LIGAMENT_CUH