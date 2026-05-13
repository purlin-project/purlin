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
    static constexpr int ALL_PIPE_STAGES = PIPE_STAGES * WARPS;
    static constexpr int PIPELINE_BYTES = STAGE_BYTES * ALL_PIPE_STAGES;
    static constexpr int PIPELINE_SMEM_BYTES = PIPELINE_BYTES +
      ALL_PIPE_STAGES * sizeof(cuda::barrier<cuda::thread_scope_block>);
    // producer-consumer config (should be deprecated)
    static constexpr int PRODUCER_THREADS = THREADS / 2;
    static constexpr int CONSUMER_THREADS = THREADS - PRODUCER_THREADS;
    static constexpr int PRODUCER_WARPS = PRODUCER_THREADS / WARP_SIZE;
    static constexpr int CONSUMER_WARPS = CONSUMER_THREADS / WARP_SIZE;
    static constexpr int TOTAL_PIPE_STAGES = PRODUCER_WARPS * PIPE_STAGES;
    static constexpr int TOTAL_STAGE_BYTES = STAGE_BYTES * PRODUCER_WARPS;
    static constexpr int PIPELINE_BYTES1 = TOTAL_STAGE_BYTES * PIPE_STAGES; // bytes in flight at steady state
    static constexpr int PIPELINE_SMEM_BYTES1 = PIPELINE_BYTES1 + TOTAL_PIPE_STAGES * sizeof(uint32_t);
    static_assert((STAGE_BYTES / ALIGNMENT_BYTES) % WARP_SIZE == 0);
    static constexpr int STAGE_ELEMS_PER_THREAD = (STAGE_BYTES / ALIGNMENT_BYTES) / WARP_SIZE;
    // reduction config (should be deprecated)
    static constexpr int RED_PRODUCER_WARPS = 1;
    static constexpr int RED_PRODUCER_THREADS = RED_PRODUCER_WARPS * WARP_SIZE;
    static constexpr int RED_CONSUMER_WARPS = (AtomConfig::THREADS / WARP_SIZE) - 1;
    static constexpr int RED_CONSUMER_THREADS = RED_CONSUMER_WARPS * WARP_SIZE;
    static constexpr int CONS_ELEMS_PER_THREAD = (STAGE_BYTES / ALIGNMENT_BYTES) / RED_CONSUMER_THREADS;
  };
  template<typename Cfg>
  __device__ __forceinline__
  void redProducer(const ReduceTRArgs& redArgs,
    const int& totalStages,
    cuda::barrier<cuda::thread_scope_block> (&ready)[Cfg::TOTAL_PIPE_STAGES],
    cuda::barrier<cuda::thread_scope_block> (&filled)[Cfg::TOTAL_PIPE_STAGES],
    cuda::std::byte* __restrict__ const& stagingBuffers) {
    static_assert(Cfg::RED_PRODUCER_WARPS == 1);
    static_assert(Cfg::TOTAL_PIPE_STAGES <= WARP_SIZE);
    const int laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
    const bool active = Cfg::TOTAL_PIPE_STAGES == 1 ?
    cuda::ptx::elect_sync(0xFFFFFFFF) : laneId < Cfg::TOTAL_PIPE_STAGES;
    auto* __restrict__ const stagingBuffer = stagingBuffers + laneId * Cfg::STAGE_BYTES;

    // priming
    if (active) {
      const int globalStage = laneId;
      const int dataPeer = globalStage % redArgs.world;
      const auto peerSlot = globalStage / redArgs.world;
      const auto* __restrict__ vSp = redArgs.sources[dataPeer] + peerSlot * Cfg::STAGE_BYTES;
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          stagingBuffer,
          vSp,
          Cfg::STAGE_BYTES,
          cuda::device::barrier_native_handle(filled[laneId]));
      cuda::std::ignore = cuda::device::barrier_arrive_tx(filled[laneId], 1, Cfg::STAGE_BYTES);
    }
    __syncwarp();
    const auto uSI = cuda::round_down(totalStages, Cfg::TOTAL_PIPE_STAGES); // Uniform Steady state Iterations (USC)
    // steady state
    for (int i = Cfg::TOTAL_PIPE_STAGES; i < uSI; i += Cfg::TOTAL_PIPE_STAGES) {
      if (active) {
        const int globalStage = i + laneId;
        const int dataPeer = globalStage % redArgs.world;
        const auto peerSlot = globalStage / redArgs.world;
        const auto* __restrict__ vSp = redArgs.sources[dataPeer] + peerSlot * Cfg::STAGE_BYTES;
        // wait for smem to be ready
        ready[laneId].arrive_and_wait();
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          stagingBuffer,
          vSp,
          Cfg::STAGE_BYTES,
          cuda::device::barrier_native_handle(filled[laneId]));
        cuda::std::ignore = cuda::device::barrier_arrive_tx(filled[laneId], 1, Cfg::STAGE_BYTES);
      }
      __syncwarp();
    }
    // residue
    if (totalStages > uSI) {
      const auto residue = totalStages - uSI;
      if (laneId < residue) {
        const int globalStage = uSI + laneId;
        const int dataPeer = globalStage % redArgs.world;
        const auto peerSlot = globalStage / redArgs.world;
        const auto* __restrict__ vSp = redArgs.sources[dataPeer] + peerSlot * Cfg::STAGE_BYTES;
        ready[laneId].arrive_and_wait();
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          stagingBuffer,
          vSp,
          Cfg::STAGE_BYTES,
          cuda::device::barrier_native_handle(filled[laneId]));
        cuda::std::ignore = cuda::device::barrier_arrive_tx(filled[laneId], 1, Cfg::STAGE_BYTES);
      }
      __syncwarp();
    }
  }

  template<typename Config, typename Element, typename RedOp>
  __device__ __forceinline__
  void redConsumer(const ReduceTRArgs& redArgs,
    const int& totalStages,
    cuda::barrier<cuda::thread_scope_block> (&ready)[Config::TOTAL_PIPE_STAGES],
    cuda::barrier<cuda::thread_scope_block> (&filled)[Config::TOTAL_PIPE_STAGES],
    const cuda::std::byte* __restrict__ const& workspace, const int& tId) {
    using VE = cuda::std::conditional_t<
    (Config::ALIGNMENT_BYTES > sizeof(Element)), typename Element2<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Config::ALIGNMENT_BYTES > sizeof(Element)), typename Element2<ReduceAccumType<Element>>::type, ReduceAccumType<Element>>;
    constexpr int vectorWidth = Config::ALIGNMENT_BYTES / sizeof(VE);
    using AVT = cutlass::AlignedArray<AccumType, vectorWidth>;
    using VER = DataToRawType<VE>::type;
    using VT = cutlass::AlignedArray<VER, vectorWidth>;
    static_assert(cuda::std::is_trivially_copyable_v<VT>);
    const auto* __restrict__ vW = reinterpret_cast<const VT*>(workspace);
    auto* __restrict__ vD = reinterpret_cast<VT*>(redArgs.dst);
    VT stash[Config::CONS_ELEMS_PER_THREAD];
    AVT accumulators[Config::CONS_ELEMS_PER_THREAD];
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VE, AccumType> storeConv{};
    constexpr RedOp op{};
    constexpr InplaceZero<AccumType> clear{};
    constexpr int stageElems = Config::STAGE_BYTES / sizeof(VT);
    cuda::static_for<Config::CONS_ELEMS_PER_THREAD>([&](auto i) {
      cuda::static_for<vectorWidth>([&](auto j) {
        clear(accumulators[i][j]);
      });
    });
    const auto laneId = tId % WARP_SIZE;
    int ticker = 0;
    int chunkIdx = 0;
    // priming
    #pragma unroll 2
    for (int globalStage = 0; globalStage < totalStages; ++globalStage) {
      ticker += 1;
      const auto stage = globalStage % Config::TOTAL_PIPE_STAGES;
      if (laneId == 0) {
        filled[stage].arrive_and_wait();
      }
      __syncwarp();
      // 1. drain smem buffer to registers
      cuda::static_for<Config::CONS_ELEMS_PER_THREAD>([&](auto i) {
        const int offset = stage * stageElems + (i * Config::RED_CONSUMER_THREADS + tId);
        stash[i] = vW[offset];
      });
      __syncwarp();
      if (laneId == 0) {
        ready[stage].arrive();
      }
      // 3. reduce in-place to accumulators
      cuda::static_for<Config::CONS_ELEMS_PER_THREAD>([&](auto i) {
        AVT val{};
        cuda::static_for<val.size()>([&](auto j) {
          val[j] = loadConv(stash[i][j]);
        });
        op(accumulators[i], val); // convert to accumulator type
      });
      if (ticker == redArgs.world) {
        ticker = 0;
        // store to gmem
        cuda::static_for<Config::CONS_ELEMS_PER_THREAD>([&](auto i) {
          VT resultRaw{};
          cuda::static_for<resultRaw.size()>([&](auto j) {
            resultRaw[j] = storeConv(accumulators[i][j]);
          });
          const size_t offset = (static_cast<size_t>(chunkIdx) * stageElems) + (i * Config::RED_CONSUMER_THREADS + tId);
          vD[offset] = resultRaw;
        });
        chunkIdx++;
        // clear
        cuda::static_for<Config::CONS_ELEMS_PER_THREAD>([&](auto i) {
          cuda::static_for<vectorWidth>([&](auto j) {
            clear(accumulators[i][j]);
          });
        });
      }
    }
  }

  template<typename Cfg>
  __device__ __forceinline__
  void putProducer(const int& totalStages,
    uint32_t* __restrict__ const& flags,
    cuda::std::byte* __restrict__ const& stagingBuffers,
    const cuda::std::byte* __restrict__ const& src, const int& prodId) {
    static_assert(!cuda::std::is_void_v<Cfg>);
    using AT = AlignedType<Cfg::ALIGNMENT_BYTES>::type;
    constexpr int VectorWidth = Cfg::ALIGNMENT_BYTES / sizeof(AT);
    using VT = cutlass::AlignedArray<AT, VectorWidth, Cfg::ALIGNMENT_BYTES>;
    auto* __restrict__ vW = reinterpret_cast<VT*>(stagingBuffers);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const int laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
    const int producerStages = totalStages / Cfg::PRODUCER_WARPS + (prodId < (totalStages % Cfg::PRODUCER_WARPS));
    // assert(producerStages >= Cfg::PIPE_STAGES)
    // Stage 1: pipeline priming
    cuda::static_for<Cfg::PIPE_STAGES>([&](auto i) {
      const int stage = prodId + i * Cfg::PRODUCER_WARPS;
      cuda::static_for<Cfg::STAGE_ELEMS_PER_THREAD>([&](auto j) {
        const int slot = (stage * Cfg::STAGE_ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
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
      cuda::static_for<Cfg::STAGE_ELEMS_PER_THREAD>([&](auto j) {
        const int slot = (stage * Cfg::STAGE_ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
        const size_t dataSlot = (static_cast<size_t>(dataStage) * Cfg::STAGE_ELEMS_PER_THREAD + j) * WARP_SIZE + laneId;
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
  template<typename Cfg, TransferType tt = TransferType::asynchronous>
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
      }
    }
    if constexpr (tt == TransferType::synchronous) {
      if (active) {
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>());
      }
    }
    __syncwarp();
  }

  template<typename Cfg, TransferType tt = TransferType::asynchronous>
  __device__ __forceinline__
  void putConsumer2(const int& totalStages,
    uint32_t* __restrict__ const& flags,
    const cuda::std::byte* __restrict__ const& stagingBuffers,
    cuda::std::byte* __restrict__ const& dst,
    const int& consumerId) {
    const auto consumerStages = totalStages / Cfg::CONSUMER_WARPS + (consumerId < totalStages % Cfg::CONSUMER_WARPS);
    // Static stage assignment per lane — only lanes [0, NUM_STAGES) are active.
    for (int i = 0; i < consumerStages; ++i) {
      const auto globalStage = consumerId + i * Cfg::CONSUMER_WARPS;
      const auto stage = globalStage % Cfg::TOTAL_PIPE_STAGES;
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        const cuda::atomic_ref<uint32_t, cuda::thread_scope_block> flag{*(flags + stage)};
        const auto* __restrict__ stagingBuffer = stagingBuffers + stage * Cfg::STAGE_BYTES;
        const auto offset = static_cast<size_t>(globalStage) * Cfg::STAGE_BYTES;
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
    if constexpr (tt == TransferType::synchronous) {
      if (cuda::ptx::elect_sync(0xFFFFFFFF)) {
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>());
      }
    }
    __syncwarp();
  }

  template<typename Config, typename BaseConfig, TransferType tt = TransferType::asynchronous>
  __device__ __forceinline__
  void put(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    static_assert(Config::THREADS % (2 * WARP_SIZE) == 0);
    //assert(__isShared(workspace));
    // 1. if less than threshold, do direct GMEM -> GMEM
    if (bytes < Config::PIPELINE_BYTES) {
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
    auto* __restrict__ flags = reinterpret_cast<uint32_t*>(workspace + Config::PIPELINE_BYTES);
    if (threadIdx.x < Config::TOTAL_PIPE_STAGES) {
      flags[threadIdx.x] = 0U;
    }
    __syncthreads();
    const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;

    const int totalStages = bytes / Config::STAGE_BYTES; // assert(totalStages >= Cfg::TOTAL_PIPE_STAGES)
    if (warpId < Config::CONSUMER_WARPS) {
      // last warp is consumer
      ligament::putConsumer2<Config, tt>(totalStages, flags, workspace, dst, warpId);
      return;
    }
    // producer
    ligament::putProducer<Config>(totalStages, flags, workspace, src, warpId - Config::CONSUMER_WARPS);
    // residue
    const auto cutoff = totalStages * Config::STAGE_BYTES;
    if (bytes > cutoff) {
      // high value increases register pressure, low value reduces ILP
      constexpr auto residueUnrollFactor = 2;
      using CopyElement = AlignedType<Config::ALIGNMENT_BYTES>::type;
      const auto leftover = bytes - cutoff;
      using OpCfg = fascia::PeerOpConfig<
        BaseConfig,
        ST, // store op
        CopyElement,
        uint32_t,
        residueUnrollFactor,
        Config::PRODUCER_THREADS
      >;
      // via LSU: GMEM (local) -> RMEM -> GMEM (remote)
      fascia::putOp<OpCfg>(src + cutoff, dst + cutoff, leftover);
    }
  }

  template<typename Cfg>
  __device__ __forceinline__
  void putProducerTT(const int& totalStages,
    cuda::barrier<cuda::thread_scope_block> (&ready)[Cfg::TOTAL_PIPE_STAGES],
    cuda::barrier<cuda::thread_scope_block> (&filled)[Cfg::TOTAL_PIPE_STAGES],
    cuda::std::byte* __restrict__ const& stagingBuffers,
    const cuda::std::byte* __restrict__ const& src) {
    static_assert(Cfg::THREADS == 2 * WARP_SIZE);
    static_assert(Cfg::PRODUCER_WARPS == 1);
    static_assert(Cfg::TOTAL_PIPE_STAGES == Cfg::PIPE_STAGES);
    static_assert(Cfg::TOTAL_PIPE_STAGES <= WARP_SIZE);
    const int laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
    const bool active = Cfg::TOTAL_PIPE_STAGES == 1 ?
    cuda::ptx::elect_sync(0xFFFFFFFF) : laneId < Cfg::TOTAL_PIPE_STAGES;
    auto* __restrict__ const stagingBuffer = stagingBuffers + laneId * Cfg::STAGE_BYTES;

    // priming
    if (active) {
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          stagingBuffer,
          src + laneId * Cfg::STAGE_BYTES,
          Cfg::STAGE_BYTES,
          cuda::device::barrier_native_handle(filled[laneId]) // TMA engine will decrement tx count on this barrier
      );
      // arrive: satisfies arrival count (1) + sets tx count (STAGE_BYTES)
      cuda::std::ignore = cuda::device::barrier_arrive_tx(filled[laneId], 1, Cfg::STAGE_BYTES);
    }
    __syncwarp();
    const auto uSI = cuda::round_down(totalStages, Cfg::TOTAL_PIPE_STAGES); // Uniform Steady state Iterations (USC)
    // steady state
    for (int i = Cfg::TOTAL_PIPE_STAGES; i < uSI; i += Cfg::TOTAL_PIPE_STAGES) {
      if (active) {
        // wait for smem to be ready
        ready[laneId].arrive_and_wait();
        const auto globalStage = i + laneId;
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          stagingBuffer,
          src + globalStage * Cfg::STAGE_BYTES,
          Cfg::STAGE_BYTES,
          cuda::device::barrier_native_handle(filled[laneId])
          );
        cuda::std::ignore = cuda::device::barrier_arrive_tx(filled[laneId], 1, Cfg::STAGE_BYTES);
      }
      __syncwarp();
    }
    // residue
    if (totalStages > uSI) {
      const auto residue = totalStages - uSI;
      if (laneId < residue) {
        ready[laneId].arrive_and_wait();
        const auto globalStage = uSI + laneId;
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          stagingBuffer,
          src + globalStage * Cfg::STAGE_BYTES,
          Cfg::STAGE_BYTES,
          cuda::device::barrier_native_handle(filled[laneId])
          );
        cuda::std::ignore = cuda::device::barrier_arrive_tx(filled[laneId], 1, Cfg::STAGE_BYTES);
      }
      __syncwarp();
    }
  }
  template<typename Cfg>
  __device__ __forceinline__
  void putConsumerTT(const int& totalStages,
    cuda::barrier<cuda::thread_scope_block> (&ready)[Cfg::TOTAL_PIPE_STAGES],
    cuda::barrier<cuda::thread_scope_block> (&filled)[Cfg::TOTAL_PIPE_STAGES],
    const cuda::std::byte* __restrict__ const& stagingBuffers,
    cuda::std::byte* __restrict__ const& dst) {
    const int laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
    const bool active = Cfg::TOTAL_PIPE_STAGES == 1 ?
    cuda::ptx::elect_sync(0xFFFFFFFF) : laneId < Cfg::TOTAL_PIPE_STAGES;
    const int stageId = laneId;
    auto* __restrict__ stagingBuffer = stagingBuffers + stageId * Cfg::STAGE_BYTES;

    const auto fullRounds = totalStages / Cfg::TOTAL_PIPE_STAGES;
    const auto residue = totalStages % Cfg::TOTAL_PIPE_STAGES;

    for (int round = 0; round < fullRounds; ++round) {
      const size_t offset = (static_cast<size_t>(round) * Cfg::TOTAL_PIPE_STAGES + stageId) * Cfg::STAGE_BYTES;
      if (active) {
        filled[laneId].arrive_and_wait();
        // TMA store: smem[s] → remote HBM (NVLink)
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global, cuda::ptx::space_shared,
          dst + offset, stagingBuffer, Cfg::STAGE_BYTES);
        cuda::ptx::cp_async_bulk_commit_group();
        // Wait until TMA engine has finished reading smem[s].
        cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
        // Signal producer: smem is free to overwrite.
        ready[laneId].arrive();
      }
      __syncwarp();
    }

    if (Cfg::TOTAL_PIPE_STAGES > 1 && residue) {
      const size_t offset = (static_cast<size_t>(fullRounds) * Cfg::TOTAL_PIPE_STAGES + stageId) * Cfg::STAGE_BYTES;
      if (laneId < residue) {
        filled[laneId].arrive_and_wait();
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global, cuda::ptx::space_shared,
          dst + offset, stagingBuffer, Cfg::STAGE_BYTES);
      }
      __syncwarp();
    }
  }

  template<int selectedWarp>
  __device__ __forceinline__
  bool isElected(const int& warpId) {
    const auto uniform_warp_id = __shfl_sync(0xFFFFFFFF, warpId, 0);
    return (uniform_warp_id == selectedWarp && cuda::ptx::elect_sync(0xFFFFFFFF));
  }

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
    for (int i = static_cast<int>(threadIdx.x); i < Config::ALL_PIPE_STAGES; i += Config::THREADS) {
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
      const int stage = globalStage % Config::ALL_PIPE_STAGES;
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
      const auto stage = globalStage % Config::ALL_PIPE_STAGES;
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

  template<typename RedOp = ArrayInplaceSum<900>, typename Element>
  __device__ __forceinline__
  static void reduce0(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
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
      (Config::ALIGNMENT_BYTES > sizeof(Element)), typename Element2<ReduceAccumType<Element>>::type,
        ReduceAccumType<Element>>;
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
    const int warpId = static_cast<int>(threadIdx.x / WARP_SIZE);
    auto* __restrict__ barriers = reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>
    (workspace + COPY_PIPELINE_BYTES);
    for (int i = static_cast<int>(threadIdx.x); i < Config::PIPE_STAGES; i += Config::THREADS) {
      // initialize mbarrier objects
      init(barriers + i, 1);
    }
    __syncthreads();
    // priming
    cuda::static_for<Config::PIPE_STAGES>([&](auto i) {
      constexpr auto selectedWarp = i % Config::WARPS;
      if (ligament::isElected<selectedWarp>(warpId)) {
        constexpr int stage = i;
        const int dataPeer = stage % redArgs.world;
        const auto peerSlot = stage / redArgs.world;
        auto& barrier = *(barriers + stage);
        const auto* __restrict__ sP = redArgs.sources[dataPeer] + peerSlot * Config::STAGE_BYTES;
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
    // steady state
    for (int globalStage = Config::PIPE_STAGES; globalStage < totalStages; ++globalStage) {
      ticker++;
      const int stage = globalStage % Config::PIPE_STAGES;
      if (ligament::isElected<0>(warpId)) {
        auto* __restrict__ barrier = barriers + stage;
        barrier->arrive_and_wait();
      }
      __syncthreads();
      cuda::static_for<Config::ELEMS_PER_THREAD>([&](auto j) {
        const int slot = (stage * Config::ELEMS_PER_THREAD + j) * Config::THREADS + threadIdx.x;
        // smem -> rmem
        reginald[j] = vW[slot];
      });
      __syncthreads();
      if (ligament::isElected<0>(warpId)) {
        const int dataPeer = globalStage % redArgs.world;
        const auto peerSlot = globalStage / redArgs.world;
        auto& barrier = *(barriers + stage);
        const auto* __restrict__ sP = redArgs.sources[dataPeer] + peerSlot * Config::STAGE_BYTES;
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
      if (ligament::isElected<0>(warpId)) {
        auto* __restrict__ barrier = barriers + stage;
        barrier->arrive_and_wait();
      }
      __syncthreads();
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

  __device__ __forceinline__
  static void putAsync1(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace) {
    ligament::put<Config, BaseConfig, TransferType::asynchronous>(dst, src, bytes, workspace);
  }

  template<typename Element>
  __device__ __forceinline__
  static void reduce1(const ReduceTRArgs& redArgs, Element* __restrict__ const& typedWorkspace) {
    static_assert((Config::STAGE_BYTES / Config::ALIGNMENT_BYTES) % Config::RED_CONSUMER_THREADS == 0);
    // assert(__isShared(typedWorkspace));
    using RedOp = ArrayInplaceSum<900>;
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    const auto roundedBytes = cuda::round_down(redArgs.bytesRed, Config::STAGE_BYTES);
    const auto stagesPerPeer = static_cast<int>(roundedBytes / Config::STAGE_BYTES);
    const auto totalStages = stagesPerPeer * redArgs.world;
    if (redArgs.bytesRed < Config::STAGE_BYTES || totalStages < Config::TOTAL_PIPE_STAGES) {
      fascia::reduce<Config_, RedOp, Element>(redArgs);
      return;
    }
    const auto warpId = threadIdx.x / WARP_SIZE;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> ready[Config::TOTAL_PIPE_STAGES];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> filled[Config::TOTAL_PIPE_STAGES];
    if (threadIdx.x < Config::TOTAL_PIPE_STAGES) {
      init(ready + threadIdx.x, Config::WARPS);
      init(filled + threadIdx.x, Config::WARPS);
    }
    __syncthreads();
    if (warpId == 0) {
      // producer
      ligament::redProducer<Config>(redArgs, totalStages, ready, filled, workspace);
    }
    else {
      // consumer
      ligament::redConsumer<Config, Element, RedOp>(redArgs, totalStages, ready, filled, workspace,
      threadIdx.x - Config::RED_PRODUCER_THREADS);
    }
    // residue
    if (redArgs.bytesRed > roundedBytes) {
      const auto cutoff = roundedBytes;
      auto* __restrict__ dst = redArgs.dst + cutoff;
      const auto bytesRed = redArgs.bytesRed - cutoff;
      fascia::reduce<Config_, RedOp, Element>(redArgs, dst, bytesRed, cutoff);
    }
  }

  __device__ __forceinline__
  static void putAsyncTT(cuda::std::byte* __restrict__ const& dst,
   const cuda::std::byte* __restrict__ const& src,
   const size_t& bytes,
   cuda::std::byte* __restrict__ const& workspace) {
    if (bytes < Config::PIPELINE_BYTES) {
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
    const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> ready[Config::TOTAL_PIPE_STAGES];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> filled[Config::TOTAL_PIPE_STAGES];

    if (threadIdx.x < Config::TOTAL_PIPE_STAGES) {
      init(ready + threadIdx.x, 2);
      init(filled + threadIdx.x, 2);
    }
    __syncthreads();
    const size_t totalStages = bytes / Config::STAGE_BYTES; // assert(totalStages >= Cfg::TOTAL_PIPE_STAGES)
    if (warpId == 1) {
      ligament::putConsumerTT<Config>(totalStages, ready, filled, workspace, dst);
    }
    else {
      ligament::putProducerTT<Config>(totalStages, ready, filled, workspace, src);
    }
    const auto cutoff = totalStages * Config::STAGE_BYTES;
    if (bytes > cutoff) {
      // high value increases register pressure, low value reduces ILP
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
};
#endif //SUTURE_LIGAMENT_CUH
