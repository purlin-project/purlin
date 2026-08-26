//
// Created by osayamen on 7/19/26.
//
#ifndef PURLIN_HOST_CODESIGN_CUH
#define PURLIN_HOST_CODESIGN_CUH

namespace purlin::host {
  static constexpr int FALLBACK = 0;

  // Remote readers needed to hold ~80% of one fabric direction:
  // ceil(0.8 x linkBW / per-SM read-issue ceiling).
  // Consumer counts in the throughput bands must not drop below these.
  // Hopper: 450 / ~46 GB/s per SM (measured) -> 8.
  // Blackwell: 900 / 48 (measured) -> 16;
  // Ampere: 300 / 32 -> 8.
  static constexpr int MIN_SATURATION_READERS_SM80 = 8;
  static constexpr int MIN_SATURATION_READERS_SM90 = 8;
  static constexpr int MIN_SATURATION_READERS_SM100 = 16;

  template<int World>
  consteval int getWorldUnroll() {
    if constexpr (World == 8) {
      return 8;
    }
    if constexpr (World == 4) {
      return 4;
    }
    return 2;
  }

  struct CodesignPolicyBase {
    static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
    static constexpr int LR_THREADS = 512;
    static constexpr int THREADS = 128;
    static constexpr int PIPE_STAGES = 8;
    static constexpr int STAGE_EXTENT = 2;
    static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
    // Cyclic staging pays a drain round trip per slot, so it can prefer larger
    // chunks than the resident chunked band; 0 falls back to CHUNK_SIZE.
    static constexpr size_t CYCLIC_CHUNK_SIZE = 0;
    // Copy-pipeline depth for the chunked (and cyclic) bands; 0 = PIPE_STAGES.
    // Deephalf pairs it with a halved consumer count in the large bands while
    // the small-size bands keep the shallow pipeline's faster fill.
    static constexpr int CHUNKED_PIPE_STAGES = 0;
    // Consumer cap for the allGather ALT band; AUTO = MAX_CONSUMER_BLOCKS.
    static constexpr int ALT_CONSUMER_BLOCKS = AUTO;
    static constexpr size_t NON_CHUNKED_MAX_BYTES = 0;
    static constexpr size_t CHUNK_SIZE_MID = 0;
    static constexpr size_t CHUNK_SIZE_LARGE = 0;
    static constexpr size_t MID_CHUNK_MIN_BYTES = static_cast<size_t>(-1);
    static constexpr size_t LARGE_CHUNK_MIN_BYTES = static_cast<size_t>(-1);
    static constexpr int MM_DEPTH = AUTO;
    static constexpr int PACED_MM_DEPTH = 1;
    // Copy-pipeline depth for the multimem band's put stage; 0 = PIPE_STAGES.
    // The mm reducers pipeline in registers, so only the puts feel this knob.
    static constexpr int MM_PIPE_STAGES = 0;
    static constexpr int MM_CONSUMER_BLOCKS = AUTO;
    // Upper size bound for the reduceScatter multimem band; 0 disables it.
    static constexpr size_t MM_MAX_BYTES = static_cast<size_t>(-1);
    static constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
    static constexpr int CHUNKED_PUT_BLOCKS = 32;
    static constexpr int LOCAL_PUT_BLOCKS = UNUSED;
    static constexpr int GATHER_BLOCKS = UNUSED;
    static constexpr int MAX_CONSUMER_BLOCKS = 32;
    static constexpr int ALT_THREADS = 0;
    static constexpr size_t ALT_MIN_BYTES = 0;
    static constexpr size_t ALT_MAX_BYTES = 0;
    static constexpr size_t LR_PARTITION_MIN_BYTES = 64UL * 1024UL;
    static constexpr size_t LR_PARTITION_MAX_BYTES = 512UL * 1024UL;
    // Full-buffer LR uses the multicast packet broadcast only at or below this size.
    static constexpr size_t LR_MM_MAX_BYTES = static_cast<size_t>(-1);
    static constexpr size_t LR_PARTITION_SMALL_MAX_BYTES = 64UL * 1024UL;
    static constexpr size_t LR_WIDE_MIN_BYTES = 256UL * 1024UL;
    static constexpr size_t LR_DIRECT_MAX_BYTES = 16UL * 1024UL;
    static constexpr int LR_PARTITION_SMALL_THREADS = 256;
    static constexpr int LR_WIDE_THREADS = 1024;
    static constexpr int LR_DIRECT_BLOCKS_PER_PEER = 4;
    static constexpr int LR_PARTITION_SMALL_BLOCKS_PER_PEER = 4;
    static constexpr int LR_PARTITION_BLOCKS_PER_PEER = 3;
    static constexpr int LR_WIDE_BLOCKS_PER_PEER = 4;
  };

  namespace detail {
    template<int World>
    struct BaseAllGather : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 1024;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
      static constexpr int ALT_THREADS = 256;
      static constexpr size_t ALT_MIN_BYTES = 1024;
      static constexpr size_t ALT_MAX_BYTES = 64UL * 1024UL;
    };

    template<>
    struct BaseAllGather<2> : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr int ALT_THREADS = 256;
      static constexpr size_t ALT_MIN_BYTES = 32UL * 1024UL * 1024UL;
      static constexpr size_t ALT_MAX_BYTES = static_cast<size_t>(-1);
    };

    template<>
    struct BaseAllGather<4> : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 256UL*1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL*1024UL*1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 8;
    };

    template<int World>
    struct LigamentAllGather : BaseAllGather<World> {};

    template<>
    struct LigamentAllGather<8> : BaseAllGather<8> {
      // Deephalf (H100-measured): 2 consumers per peer with 16-stage chunked
      // pipelines tie the 4-per-peer shallow grid (chunked band 48 -> 32
      // blocks); the non-chunked band holds 2/peer at the shallow depth
      // (64 -> 48 blocks). The ALT band starves below 4/peer, so it keeps its
      // consumer count.
      static constexpr int MAX_CONSUMER_BLOCKS = 2;
      static constexpr int ALT_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      // per-peer cap: 2 x 7 remote peers = 14 readers >= the floor of 8
      static_assert(MAX_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAllGather<4> : BaseAllGather<4> {
      // Deephalf at world 4 (H100-measured): 4 consumers per peer with 16-stage
      // chunked pipelines tie the 8-per-peer grid (64 -> 48 blocks); the
      // non-chunked band holds 4/peer at the shallow depth.
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      // 4 x 3 remote peers = 12 readers >= the floor of 8
      static_assert(MAX_CONSUMER_BLOCKS * 3 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAllGather<2> : BaseAllGather<2> {
      // Deephalf at world 2, accepted under the relaxed 8% bar (worst +4.5% at
      // a small non-chunked size): 8/peer needs the deep pipeline in every
      // band. 8 readers sit exactly at the saturation floor.
      static constexpr int MAX_CONSUMER_BLOCKS = 8;
      static constexpr int PIPE_STAGES = 16;
      static constexpr int ALT_CONSUMER_BLOCKS = 16;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<int World>
    struct CortexAllGatherBase : BaseAllGather<World> {
      static constexpr int ALT_THREADS = 0;
      static constexpr size_t ALT_MIN_BYTES = 0;
      static constexpr size_t ALT_MAX_BYTES = 0;
    };

    template<int World>
    struct CortexAllGather : CortexAllGatherBase<World> {
    };

    template<>
    struct CortexAllGather<2> : CortexAllGatherBase<2> {
      static constexpr size_t LATENCY_THRESHOLD = 2UL * 1024UL * 1024UL;
      static constexpr int STAGE_EXTENT = 8;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
    };

    template<>
    struct CortexAllGather<4> : CortexAllGatherBase<4> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr int MAX_CONSUMER_BLOCKS = 8;
    };

    template<>
    struct CortexAllGather<8> : CortexAllGatherBase<8> {
      static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
    };

    template<int World>
    struct BaseAllReduce : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr int STAGE_EXTENT = 1;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 16;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int GATHER_BLOCKS = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
      static constexpr int PACED_MM_DEPTH = 1;
      static constexpr int MM_DEPTH = 8;
      static constexpr int MM_CONSUMER_BLOCKS = 8;
      static constexpr size_t CHUNK_SIZE = 512UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 1UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE_LARGE = 1UL * 1024UL * 1024UL;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 64UL * 1024UL * 1024UL;
    };

    template<>
    struct BaseAllReduce<2> : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
      // Broadcasting packets to a single peer gains nothing past the latency floor.
      static constexpr size_t LR_MM_MAX_BYTES = 32UL * 1024UL;
    };

    template<>
    struct BaseAllReduce<4> : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr int STAGE_EXTENT = 1;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 16;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int GATHER_BLOCKS = 16;
      static constexpr size_t CHUNK_SIZE = 512UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 1UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE_MID = 1UL * 1024UL * 1024UL;
      static constexpr size_t MID_CHUNK_MIN_BYTES = 16UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE_LARGE = 2UL * 1024UL * 1024UL;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 64UL * 1024UL * 1024UL;
      // 16 blocks x 256 threads x depth 4 = 16K outstanding 16B transactions,
      // the same fabric congestion knee measured for world 8.
      static constexpr int MM_DEPTH = 4;
      static constexpr int MM_CONSUMER_BLOCKS = 16;
      static constexpr int PACED_MM_DEPTH = 2;
      static constexpr size_t LR_PARTITION_MIN_BYTES = 128UL * 1024UL;
      static constexpr size_t LR_PARTITION_MAX_BYTES = 1UL * 1024UL * 1024UL;
      // Fan-out 3 is too small for the full-buffer packet broadcast to pay off
      // beyond the latency floor; the partitioned path keeps multimem regardless.
      static constexpr size_t LR_MM_MAX_BYTES = 32UL * 1024UL;
      static constexpr int LR_PARTITION_BLOCKS_PER_PEER = 4;
    };

    template<int World>
    struct TendonAllReduce : BaseAllReduce<World> {
      static constexpr int THREADS = 128;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static constexpr int STAGE_EXTENT = 2;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
    };

    template<int World>
    struct LigamentAllReduce : BaseAllReduce<World> {};

    template<>
    struct LigamentAllReduce<8> : BaseAllReduce<8> {
      // Deephalf (H100-measured): the paced multimem bands hold their bandwidth
      // with half the consumer blocks when the register pipeline doubles
      // (16 x depth 2 = 32 x depth 1 in-flight), trimming the NC/fine-chunked
      // grids from 64 to 48 blocks. The large band keeps its own 8-consumer
      // depth-8 shape. The unicast fallback (multimem off) pays ~7% at 64M
      // under this cap - unreachable while NVLS staging exists.
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr int PACED_MM_DEPTH = 2;
      // At 16 consumers the fine-chunked floor (2M = four 512K chunks) starves;
      // staging it whole keeps the point within noise of the 32-consumer grid.
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
    };

    template<>
    struct TendonAllReduce<2> : CodesignPolicyBase {
      static constexpr int THREADS = 256;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int STAGE_EXTENT = 2;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
    };

    template<>
    struct LigamentAllReduce<2> : BaseAllReduce<2> {
      // Halved consumers for the direct world-2 form, accepted under the
      // relaxed 8% bar: 48-block grid. The 16-stage pipeline recovers the
      // large sizes that 16 shallow consumers alone give up.
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
    };

    template<>
    struct LigamentAllReduce<4> : BaseAllReduce<4> {
      // Halved paced-mm consumers hold within noise at world 4 (paced depth
      // already 2); the large band keeps its own shape.
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
    };

    template<int World>
    struct CortexAllReduce : BaseAllReduce<World> {
    };

    template<>
    struct CortexAllReduce<2> : BaseAllReduce<2> {
      static constexpr size_t LATENCY_THRESHOLD = 2UL * 1024UL * 1024UL;
      static constexpr int STAGE_EXTENT = 4;
    };

    template<>
    struct CortexAllReduce<4> : BaseAllReduce<4> {
      static constexpr size_t LATENCY_THRESHOLD = 1UL * 1024UL * 1024UL;
      static constexpr int STAGE_EXTENT = 2;
    };

    template<>
    struct CortexAllReduce<8> : BaseAllReduce<8> {
      static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
      static constexpr int STAGE_EXTENT = 4;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
    };

    template<int World>
    struct BaseAll2All : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
      static constexpr int LOCAL_PUT_BLOCKS = 4;
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
    };

    template<>
    struct BaseAll2All<2> : CodesignPolicyBase {
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 16;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int LOCAL_PUT_BLOCKS = 8;
    };

    template<>
    struct BaseAll2All<4> : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
      static constexpr int LOCAL_PUT_BLOCKS = 4;
      static constexpr int MAX_CONSUMER_BLOCKS = 8;
    };

    template<int World>
    struct LigamentAll2All : BaseAll2All<World> {};

    template<>
    struct LigamentAll2All<8> : BaseAll2All<8> {
      // Deephalf (H100-measured): the non-chunked band improves outright at
      // 2 consumers per peer (remote-read contention), and the chunked band
      // ties the 4-per-peer grid once its pipeline deepens to 16 stages
      // (60 -> 46 blocks).
      static constexpr int MAX_CONSUMER_BLOCKS = 2;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      // The cyclic band wants 2M slots: 1M slots lose ~4% at 256M staging and
      // ~7% at thinner staging halves.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      // per-peer cap: 2 x 7 remote peers = 14 readers >= the floor of 8
      static_assert(MAX_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAll2All<4> : BaseAll2All<4> {
      // Deephalf at world 4, accepted under the relaxed 8% bar (worst +6.0% at
      // the top size): 4/peer x 16 stages is 12 readers x 64KB = 768KB in
      // flight, a little under the 1MB target - ~24 stages might close it.
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static_assert(MAX_CONSUMER_BLOCKS * 3 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAll2All<2> : BaseAll2All<2> {
      // 32 consumers oversubscribe the single peer link: 16 improve the top
      // size by 8% and regress nothing (the world-8 congestion knee, again).
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
    };

    template<int World>
    struct CortexAll2All : BaseAll2All<World> {
    };

    template<>
    struct CortexAll2All<2> : BaseAll2All<2> {
      static constexpr size_t LATENCY_THRESHOLD = 1UL * 1024UL * 1024UL;
      static constexpr int THREADS = 256;
    };

    template<>
    struct CortexAll2All<4> : BaseAll2All<4> {
      static constexpr size_t LATENCY_THRESHOLD = 1UL * 1024UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
    };

    template<>
    struct CortexAll2All<8> : BaseAll2All<8> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
    };

    template<int World>
    struct BaseReduceScatter : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      // Sized to the ~16K-outstanding multimem congestion knee at 128 threads.
      static constexpr int MM_DEPTH = 8;
      static constexpr int MM_CONSUMER_BLOCKS = 16;
    };

    template<>
    struct BaseReduceScatter<2> : CodesignPolicyBase {
      static constexpr int THREADS = 256;
      // headroom over the nominal power-of-two sizes because reduceScatterV dispatches on maxBytes.
      static constexpr size_t LATENCY_THRESHOLD = 576UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 5UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
      // Multimem reduce pulls both replicas through the switch (2S egress vs the
      // 1S of a unicast read) - a 23-31% loss at fan-out 1, so it is disabled.
      static constexpr size_t MM_MAX_BYTES = 0UL;
    };

    template<>
    struct BaseReduceScatter<4> : CodesignPolicyBase {
      // The unicast non-chunked path beats scattered LR from 512K per rank up.
      // the 384K edge keeps reduceScatterV's maxBytes (nominal + skew) below it.
      static constexpr size_t LATENCY_THRESHOLD = 384UL * 1024UL;
      // Non-chunked to 2M per rank, then 1M chunks.
      // with 1M chunks a V dispatch that lands just past the edge, still pipelines (2+ chunks) instead of
      // serializing as a single chunk.
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      // 4/3 egress penalty vs unicast loses at every size in the band (unlike
      // world 8's 8/7, which instruction efficiency pays for).
      static constexpr size_t MM_MAX_BYTES = 0UL;
    };

    template<int World>
    struct LigamentReduceScatter : BaseReduceScatter<World> {};

    template<>
    struct LigamentReduceScatter<8> : BaseReduceScatter<8> {
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      // Deephalf (H100-measured): 16 reducers x 16-stage pipelines carry the
      // same ~1MB of in-flight remote reads as 32 x 8 (the fabric BDP), holding
      // within 2.7% of the 64-block grid on 48 blocks. Halving without
      // deepening loses 8-16% (half the BDP); deepening without halving
      // oversubscribes the read queue; non-power-of-two reducer counts (24)
      // collapse outright. Cost: +8 registers, no spills, reduce SMEM 32->64KB.
      // Only the chunked bands deepen: the multimem band's puts and the
      // (multimem-off) non-chunked band stage too little per block to cover a
      // deep pipeline's fill.
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      // The cyclic band's per-slot drain round trip wants coarser slots at 16
      // consumers: 4M slots tie the 32-consumer baseline; 2M slots lose 2%.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90,
        "reducer count below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentReduceScatter<4> : BaseReduceScatter<4> {
      // Deephalf at world 4, accepted under the relaxed 8% bar (E2E worst
      // +5.0%): 16 reducers x 16-stage chunked pipelines with 2M chunks on a
      // 48-block grid. The non-chunked band is live here (no multimem at
      // world 4) and keeps the shallow fill.
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90,
        "reducer count below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentReduceScatter<2> : BaseReduceScatter<2> {
      // Deephalf at world 2, accepted under the relaxed 8% bar (worst +5.0%):
      // 16 reducers x 16-stage chunked pipelines, 4M chunks, 48-block grid.
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90,
        "reducer count below the Hopper read-saturation floor");
    };

    template<int World>
    struct CortexReduceScatter : BaseReduceScatter<World> {
    };

    template<>
    struct CortexReduceScatter<2> : BaseReduceScatter<2> {
      static constexpr size_t LATENCY_THRESHOLD = 2UL * 1024UL * 1024UL;
      static constexpr int STAGE_EXTENT = 4;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
    };

    template<>
    struct CortexReduceScatter<4> : BaseReduceScatter<4> {
      static constexpr size_t LATENCY_THRESHOLD = 1UL * 1024UL * 1024UL;
      static constexpr int THREADS = 256;
    };

    template<>
    struct CortexReduceScatter<8> : BaseReduceScatter<8> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
    };
    // The sm90 V collectives keep the pre-deephalf shapes: variable splits
    // leave borderline maxBytes slices at one deep-pipeline fill, so the
    // deep/halved grids regress them. Their V-specific band edges live here.
    template<int World>
    struct LigamentAllGatherV : BaseAllGather<World> {};

    template<>
    struct LigamentAllGatherV<4> : BaseAllGather<4> {
      // The V gather's TR entry sizes are context-sensitive where the fixed
      // path is not; latency-regime serves 256K-512K per rank faster and
      // stably. 2M chunks suit the V consumer where fixed AllGather prefers 4M.
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
    };

    template<int World>
    struct LigamentAll2AllV : BaseAll2All<World> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
    };

    template<>
    struct LigamentAll2AllV<4> : BaseAll2All<4> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t LATENCY_THRESHOLD = 288UL*1024UL;
      // One chunk size, no tier. A tier keyed on the rank-local maximum split
      // makes ranks near the boundary count different chunks and hang - a
      // demonstrated deadlock at 8M totals under 90% skew. Removing it costs
      // 15% at 128M totals; that stands until a tier can ride the rendezvous.
      static constexpr size_t CHUNK_SIZE = 512UL*1024UL;
    };

    template<>
    struct LigamentAll2AllV<8> : BaseAll2All<8> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      // One chunk size, no tier: a tier keyed on the rank-local maximum split
      // can make two ranks count different chunks under real skew (a hang).
      // 512K chunks pipeline the 1M-2M splits a 1M chunk serialized (+12% at
      // 8M totals). Measured dead ends for the 1-4M-total pocket vs NCCL:
      // extending the latency path up (its flag-per-8-byte packets double the
      // wire bytes) and a single-shot non-chunked band (loses the chunked
      // stage/consume overlap); every band scores ~19.5us at 1M total, so that
      // pocket is the V rendezvous cost itself.
      static constexpr size_t CHUNK_SIZE = 512UL * 1024UL;
    };

    template<int World>
    struct LigamentReduceScatterV : BaseReduceScatter<World> {};

    template<>
    struct LigamentReduceScatterV<8> : BaseReduceScatter<8> {
      // scatteredV never rides the multimem band, so its sub-2M sizes run the
      // non-chunked unicast path, where a deep pipeline's fill tail costs
      // 5-13%; the base shallow 32-consumer shape stays. The fixed path's
      // chunk and band edges carry over - they are where V's +18-30% came from.
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static constexpr size_t CYCLIC_CHUNK_SIZE = 2UL * 1024UL * 1024UL;
    };
  } // namespace detail

  template<int CodesignArch, int World>
  struct AllGatherCodesign : detail::BaseAllGather<World>{};

  template<int World>
  struct AllGatherCodesign<900, World> : detail::LigamentAllGather<World> {
  };

  template<int World>
  struct AllGatherCodesign<1000, World> : detail::CortexAllGather<World> {
  };

  template<int CodesignArch, int World>
  struct AllGatherVCodesign : AllGatherCodesign<CodesignArch, World> {
  };

  template<int World>
  struct AllGatherVCodesign<900, World> : detail::LigamentAllGatherV<World> {
  };

  template<int CodesignArch>
  struct AllGatherVCodesign<CodesignArch, 4> : AllGatherCodesign<CodesignArch, 4> {
    // The V gather's TR entry sizes are context-sensitive where the fixed path is
    // not; latency-regime serves 256K-512K per rank faster and stably. 2M chunks
    // suit the V consumer where fixed AllGather prefers 4M.
    static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
    static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
  };

  // Disambiguates <900, World> vs <CodesignArch, 4>.
  template<>
  struct AllGatherVCodesign<900, 4> : detail::LigamentAllGatherV<4> {
  };

  template<int CodesignArch, int World>
  struct AllReduceCodesign : detail::BaseAllReduce<World>{};

  template<int World>
  struct AllReduceCodesign<800, World> : detail::TendonAllReduce<World> {
  };

  template<int World>
  struct AllReduceCodesign<900, World> : detail::LigamentAllReduce<World> {
  };

  template<int World>
  struct AllReduceCodesign<1000, World> : detail::CortexAllReduce<World> {
  };

  template<int CodesignArch, int World>
  struct All2AllCodesign : detail::BaseAll2All<World>{};

  template<int World>
  struct All2AllCodesign<900, World> : detail::LigamentAll2All<World> {
  };

  template<int World>
  struct All2AllCodesign<1000, World> : detail::CortexAll2All<World> {
  };

  template<int CodesignArch, int World>
  struct All2AllVCodesign : All2AllCodesign<CodesignArch, World> {
    static constexpr int CHUNKED_PUT_BLOCKS = 16;
  };

  template<int World>
  struct All2AllVCodesign<900, World> : detail::LigamentAll2AllV<World> {
  };

  template<int CodesignArch>
  struct All2AllVCodesign<CodesignArch, 4> : All2AllCodesign<CodesignArch, 4> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t LATENCY_THRESHOLD = 288UL*1024UL;
      static constexpr size_t CHUNK_SIZE = 512UL*1024UL;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 3UL*1024UL*1024UL;
      static constexpr size_t CHUNK_SIZE_LARGE = 1UL*1024UL*1024UL;
    };

  // Disambiguates <900, World> vs <CodesignArch, 4>.
  template<>
  struct All2AllVCodesign<900, 4> : detail::LigamentAll2AllV<4> {
  };

  template<int CodesignArch, int World>
  struct ReduceScatterCodesign : detail::BaseReduceScatter<World> {};

  template<int World>
  struct ReduceScatterCodesign<900, World> : detail::LigamentReduceScatter<World> {
  };

  template<int World>
  struct ReduceScatterCodesign<1000, World> : detail::CortexReduceScatter<World> {
  };

  template<int CodesignArch, int World>
  struct ReduceScatterVCodesign : ReduceScatterCodesign<CodesignArch, World> {
  };

  template<int World>
  struct ReduceScatterVCodesign<900, World> : detail::LigamentReduceScatterV<World> {
  };
} // namespace purlin::host

#endif // PURLIN_HOST_CODESIGN_CUH
