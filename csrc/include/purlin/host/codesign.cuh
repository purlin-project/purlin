#ifndef PURLIN_HOST_CODESIGN_CUH
#define PURLIN_HOST_CODESIGN_CUH

namespace purlin::host {
  static constexpr int FALLBACK = 0;

  // Keep enough remote readers to use roughly 80% of the link's bandwidth in
  // one direction. Estimate the count with
  // ceil(0.8 * link bandwidth / per-SM read-issue limit), then use these minima:
  //   Ampere:     8 readers (300 GB/s link, 32 GB/s per SM)
  //   Hopper:     8 readers (450 GB/s link, about 46 GB/s per SM measured)
  //   Blackwell: 16 readers (900 GB/s link, 48 GB/s per SM measured)
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
    // Use the two-rank unroll factor for all other world sizes.
    return 2;
  }

  struct CodesignPolicyBase {
    // all2allV sends streams up to this size as packets with completion flags.
    // Larger streams use a staging window for each destination. Both paths
    // work across architectures and world sizes; the threshold tunes their
    // performance. On H100 with four or eight ranks, 128 KiB beat both 64 KiB
    // and 256 KiB. Other architectures may benefit from a different value.
    static constexpr size_t PER_STREAM_THRESHOLD = 128UL * 1024UL;
    // Slot size for an all2allV stream that outgrows its staging window.
    // Each slot must drain before reuse, so larger slots reduce how often the
    // stream waits. Use a multiple of CHUNK_SIZE; 0 uses CHUNK_SIZE itself.
    static constexpr size_t CYCLIC_STREAM_CHUNK = 0;
    // Once the largest all2allV stream reaches this size, assign blocks in
    // proportion to stream sizes. Below it, use an even assignment to avoid
    // the extra setup cost. A value of 0 always allows proportional assignment.
    static constexpr size_t WEIGHTED_MAPPING_MIN_BYTES = 0;
    // When enabled, use the small-transfer consumer count as a lower bound
    // when sizing all2allV. Otherwise, crossing one pipeline's worth of data
    // can drop the count to one consumer per peer, slowing a larger transfer.
    static constexpr bool CONSUMER_FLOOR = false;
    static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
    static constexpr int LR_THREADS = 512;
    static constexpr int THREADS = 128;
    static constexpr int PIPE_STAGES = 8;
    static constexpr int STAGE_EXTENT = 2;
    static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
    // Cyclic staging reuses slots after their readers finish. Larger chunks
    // spread that wait over more data. Set this to 0 to use CHUNK_SIZE.
    static constexpr size_t CYCLIC_CHUNK_SIZE = 0;
    // Number of copy-pipeline stages in the chunked and cyclic bands. A value
    // of 0 uses PIPE_STAGES. The "deephalf" configuration combines a deeper
    // pipeline with half as many consumers in the large bands, while smaller
    // transfers retain the faster startup of the shallow pipeline.
    static constexpr int CHUNKED_PIPE_STAGES = 0;
    // Pipeline depth used only by reduceScatter's cyclic band. A value of 0
    // uses CHUNKED_PIPE_STAGES, or PIPE_STAGES if that is also 0. The cyclic
    // band's larger slots can keep a deep pipeline full, unlike the smaller
    // transfers handled by the resident chunked band.
    static constexpr int CYCLIC_PIPE_STAGES = 0;
    // Maximum number of consumers in allGather's ALT band. AUTO uses
    // MAX_CONSUMER_BLOCKS.
    static constexpr int ALT_CONSUMER_BLOCKS = AUTO;
    // Maximum consumers per peer in all2allV's large band, which starts at
    // LARGE_CHUNK_MIN_BYTES. AUTO uses MAX_CONSUMER_BLOCKS. A deephalf policy
    // combines this lower cap with CHUNKED_PIPE_STAGES for streams large enough
    // to keep the deeper pipeline full.
    static constexpr int LARGE_CONSUMER_BLOCKS = AUTO;
    // Producer blocks all2allV uses once the largest split a rank sends reaches
    // LARGE_CHUNK_MIN_BYTES. AUTO uses CHUNKED_PUT_BLOCKS.
    static constexpr int LARGE_PUT_BLOCKS = AUTO;
    // Maximum consumers in the chunked and cyclic bands of reduceScatter and
    // allGather. AUTO uses MAX_CONSUMER_BLOCKS. This lets a variable-size
    // policy use deephalf only for chunked transfers and retain the best
    // measured configuration for smaller transfers.
    static constexpr int CHUNKED_CONSUMER_BLOCKS = AUTO;
    // allGather uses CHUNKED_PIPE_STAGES and CHUNKED_CONSUMER_BLOCKS once the
    // dispatch size reaches this threshold. Smaller chunked transfers keep
    // the shallower pipeline and more consumers. A value of 0 applies the
    // chunked settings throughout the chunked band.
    static constexpr size_t DEEP_CHUNK_MIN_BYTES = 0;
    static constexpr size_t NON_CHUNKED_MAX_BYTES = 0;
    static constexpr size_t CHUNK_SIZE_MID = 0;
    static constexpr size_t CHUNK_SIZE_LARGE = 0;
    static constexpr size_t MID_CHUNK_MIN_BYTES = static_cast<size_t>(-1);
    static constexpr size_t LARGE_CHUNK_MIN_BYTES = static_cast<size_t>(-1);
    static constexpr int MM_DEPTH = AUTO;
    static constexpr int PACED_MM_DEPTH = 1;
    // Copy-pipeline depth for puts in the multimem band. A value of 0 uses
    // PIPE_STAGES. Multimem reducers pipeline through registers, so this
    // setting affects only the put stage.
    static constexpr int MM_PIPE_STAGES = 0;
    static constexpr int MM_CONSUMER_BLOCKS = AUTO;
    // Maximum consumers in allReduce's cyclic multimem band. AUTO uses
    // MM_CONSUMER_BLOCKS. The cyclic band's larger slots can supply more
    // reducers than the resident band's smaller shards.
    static constexpr int CYCLIC_MM_CONSUMER_BLOCKS = AUTO;
    // Largest transfer handled by reduceScatter's multimem band. Set this to
    // 0 to disable the band.
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
    // The full-buffer latency path (LR) uses multicast packet
    // broadcasts only for transfers up to this size.
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

    // On A100, allGather reuses one staging window instead of assigning one to
    // each rank. Dividing that window into eight slots gives a 32 MiB slot at
    // every world size.
    template<int World>
    struct TendonAllGather : BaseAllGather<World> {
      // With eight A100 ranks, use packets through 64 KiB per rank. The old
      // 1 KiB threshold switched paths too early: at a 16 KiB total, raising
      // it cut latency from 9.60 to 6.92 us. Across 1 KiB-512 MiB, the geometric
      // mean improved by 5.0%, with a worst regression of 1.8%.
      //
      // Raising the threshold further to 256 KiB was 39% slower at a 2 MiB
      // total. An H200 experiment without staging also favored 64 KiB.
      // Other world sizes keep their base threshold.
      static constexpr size_t LATENCY_THRESHOLD = World == 8 ?
        64UL * 1024UL : BaseAllGather<World>::LATENCY_THRESHOLD;
      static constexpr size_t CYCLIC_CHUNK_SIZE = MAX_STAGING_SIZE / 8;
      // The "deephalf" configuration halves the consumers per peer and doubles
      // the pipeline depth. This keeps the same amount of data in flight while
      // reducing the grid by about one third. Across eight A100 GPUs and the
      // full 1 KiB-1 GiB range, it reached 1.27x, 1.19x, and 1.09x baseline
      // performance at world sizes 2, 4, and 8, respectively. Its largest loss
      // against the wider configuration was 7%.
      //
      // Apply this reduction only to measured world sizes. FALLBACK supports
      // arbitrary fan-out, where fewer consumers could undersupply the link.
      static constexpr int CHUNKED_PIPE_STAGES =
        (World == 2 || World == 4 || World == 8) ? 16 : 0;
      static constexpr int CHUNKED_CONSUMER_BLOCKS =
        World == 2 ? 8 : World == 4 ? 4 : World == 8 ? 2 : AUTO;
      // Each peer gets its own consumers, so the total reader count is the
      // per-peer count multiplied by the world size.
      static_assert(World < 2 || World * (World == 2 ? 8 : World == 4 ? 4 : 2)
        >= MIN_SATURATION_READERS_SM80,
        "per-peer consumer cap below the Ampere read-saturation floor");
    };

    template<int World>
    struct LigamentAllGather : BaseAllGather<World> {};

    template<>
    struct LigamentAllGather<8> : BaseAllGather<8> {
      // H100 benchmarks showed that two consumers per peer with a 16-stage
      // chunked pipeline match the performance of the shallow configuration
      // with four consumers per peer. This reduces the chunked grid from 48
      // blocks to 32. The non-chunked band also retains its performance with
      // two consumers per peer at the shallow depth, reducing its grid from
      // 64 blocks to 48. The ALT band still needs four consumers per peer to
      // avoid starvation, so it keeps the original count.
      static constexpr int MAX_CONSUMER_BLOCKS = 2;
      static constexpr int ALT_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      // Two consumers across seven remote peers provide 14 readers, safely
      // above Hopper's minimum of eight.
      static_assert(MAX_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAllGather<4> : BaseAllGather<4> {
      // At world size 4, H100 benchmarks showed that four consumers per peer
      // with a 16-stage chunked pipeline match the configuration with eight
      // consumers per peer, reducing the grid from 64 blocks to 48. The
      // non-chunked band retains its performance with four consumers per peer
      // at the shallow pipeline depth.
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      // Four consumers across three remote peers provide 12 readers, safely
      // above Hopper's minimum of eight.
      static_assert(MAX_CONSUMER_BLOCKS * 3 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAllGather<2> : BaseAllGather<2> {
      // At world size 2, eight consumers per peer with a deep pipeline in
      // every band met the relaxed 8% regression limit. The worst regression
      // was 4.5% for a small non-chunked transfer. With one remote peer, these
      // eight readers are exactly Hopper's minimum for saturating reads.
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
      // With only one peer, packet broadcasts provide no benefit once the
      // transfer is above the latency-oriented range.
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
      // This produces 16,384 outstanding 16-byte transactions
      // (16 blocks * 256 threads * depth 4), matching the fabric congestion
      // point measured at world size 8.
      static constexpr int MM_DEPTH = 4;
      static constexpr int MM_CONSUMER_BLOCKS = 16;
      static constexpr int PACED_MM_DEPTH = 2;
      static constexpr size_t LR_PARTITION_MIN_BYTES = 128UL * 1024UL;
      static constexpr size_t LR_PARTITION_MAX_BYTES = 1UL * 1024UL * 1024UL;
      // With a fan-out of three, full-buffer packet broadcasts stop paying off
      // above the latency-oriented range. The partitioned path continues to
      // use multimem at all sizes.
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
      // The reduce-then-gather path gives each shard its own staging window.
      // A100 measurements favored eight slots per window: four larger slots
      // reduced overlap and were 5% slower at 512 MiB, while smaller slots paid
      // the drain cost more often. FALLBACK can run at any supported world size,
      // so it reserves space using the largest possible divisor.
      static constexpr size_t CYCLIC_CHUNK_SIZE =
        (MAX_STAGING_SIZE / (World < 2 ? MAX_RANKS_PER_DOMAIN : World)) / 8;
    };

    template<int World>
    struct LigamentAllReduce : BaseAllReduce<World> {};

    template<>
    struct LigamentAllReduce<8> : BaseAllReduce<8> {
      // H100 benchmarks showed that the paced multimem bands retain their
      // bandwidth with half as many consumer blocks when the register pipeline
      // depth doubles: 16 consumers at depth 2 keep the same number of
      // operations in flight as 32 consumers at depth 1. This reduces the
      // non-chunked and fine-chunked grids from 64 blocks to 48. The large band
      // keeps its separate configuration of eight consumers at depth 8.
      //
      // If multimem is disabled, this lower cap makes the unicast fallback
      // about 7% slower at 64 MiB. That path is not reachable while NVLS
      // staging is available.
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr int PACED_MM_DEPTH = 2;
      // With 16 consumers, a 2 MiB transfer split into four 512 KiB chunks
      // cannot keep the fine-chunked band busy. Handling it as one staged
      // transfer performs within measurement noise of the 32-consumer grid.
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      // Eight H200s (2026-09-27): 4 MiB cyclic slots raised bandwidth from
      // 251 to 259 GB/s at 512 MiB and from 253 to 265 GB/s at 1 GiB.
      // Keep the resident chunks and the eight depth-8 multimem consumers;
      // extra readers and 8 MiB slots were slower in the same-session sweep.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 4UL * 1024UL * 1024UL;
    };

    template<>
    struct TendonAllReduce<2> : CodesignPolicyBase {
      static constexpr int THREADS = 256;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int STAGE_EXTENT = 2;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
      // The two-rank shortcut reduces directly into the packed output, so it
      // reuses one staging window instead of assigning one per rank. On eight
      // A100 GPUs, 32 MiB slots raised throughput from 195 to 228 GB/s at
      // 512 MiB and from 200 to 231 GB/s at 1 GiB. This nearly matches the
      // resident band's 233 GB/s at 256 MiB. Four 64 MiB slots were 5% slower
      // at 512 MiB.
      static constexpr size_t CYCLIC_CHUNK_SIZE = MAX_STAGING_SIZE / 8;
    };

    template<>
    struct LigamentAllReduce<2> : BaseAllReduce<2> {
      // At world size 2, reducing the direct path to 16 consumers produces a
      // 48-block grid and stays within the relaxed 8% regression limit. A
      // 16-stage pipeline recovers the large-transfer performance that would
      // be lost with 16 consumers at the shallow depth.
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
    };

    template<>
    struct LigamentAllReduce<4> : BaseAllReduce<4> {
      // At world size 4, halving the paced-multimem consumer count has no
      // measurable effect because its pipeline depth is already 2. The large
      // band retains its separate configuration.
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
      // B300 benchmarks from 2026-08-28 showed that the large-band
      // configuration (multimem depth 8, 16 consumers, and 2 MiB chunks)
      // should start at 16 MiB. At 32 MiB, it improved from 0.91x to 1.01x
      // baseline performance; at 64 and 128 MiB, it improved from 0.76x to
      // 0.87-0.90x. The paced bands keep their shallow pipeline because a
      // deeper pipeline was 14-27% slower at 2-8 MiB. Moving the threshold down
      // to 2 MiB, as done at world size 8, slowed every measured size with this
      // smaller fan-out.
      static constexpr int MM_DEPTH = 8;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 16UL * 1024UL * 1024UL;
      static constexpr size_t MID_CHUNK_MIN_BYTES = static_cast<size_t>(-1);
    };

    template<>
    struct CortexAllReduce<8> : BaseAllReduce<8> {
      static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
      // On B300, the partitioned latency path (LR) is slower than the
      // full-buffer LR path at 64 KiB (7.3 us versus 6.3 us), but faster from
      // 128 KiB onward. Start its range at 128 KiB instead of the shared
      // 64 KiB default.
      static constexpr size_t LR_PARTITION_MIN_BYTES = 128UL * 1024UL;
      // B200, 2026-09-27: extending partitioned LR through 2 MiB raises
      // 1/2 MiB throughput from 46/94 to 95/130 GB/s
      // (1.04/1.01x MSCCLPP). Wider windows lose to retuned throughput.
      static constexpr size_t LR_PARTITION_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static_assert(2 * (LR_PARTITION_MAX_BYTES / 8) <= PACKET_BUFFER_SIZE / 2,
        "world-eight partitioned LR packets must fit both buffer halves");
      // Non-partitionable payloads in this window take the direct LR fallback.
      static_assert(LR_PARTITION_MAX_BYTES <= PACKET_BUFFER_SIZE / 2,
        "world-eight direct LR fallback must fit the packet buffer");
      // Equal total threads to 28 x 1024; 56 smaller blocks also improve
      // 256/512 KiB by about 6/3% on B200.
      static constexpr int LR_WIDE_THREADS = 512;
      static constexpr int LR_WIDE_BLOCKS_PER_PEER = 8;
      static_assert((8 - 1) * LR_WIDE_BLOCKS_PER_PEER <= MAX_NUM_CTAS,
        "world-eight LR launch exceeds the CTA limit");
      static constexpr int STAGE_EXTENT = 4;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      // B200, 2026-09-27: at 4/8 MiB, paced depth 4 and 32 staging-copy
      // blocks raise 131/154 to 152/187 GB/s (1.07/1.00x MSCCLPP).
      // Restrict the extra puts to this band: 32 chunked puts hurt large sizes.
      static constexpr int PACED_MM_DEPTH = 4;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 8UL * 1024UL * 1024UL;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 16UL * 1024UL * 1024UL;
      // B200, 2026-09-27: 16 reducers at depth 8 with 1 MiB chunks raise
      // 64/128/256 MiB from 300/315/324 to 336/362/377 GB/s
      // (1.06/1.06/1.05x MSCCLPP), while also improving 16/32 MiB.
      // Multimem keeps 16 * 256 * 8 * 16 B = 512 KiB of requests in flight,
      // or 4 MiB counting eight-way fan-out. Its depth is independent of the
      // copy pipeline: 16 gather blocks * 128 KiB = 2 MiB, near the link BDP.
      // Four-MiB resident chunks hurt 16-64 MiB; retain 1 MiB chunks here.
      static constexpr int MM_CONSUMER_BLOCKS = 16;
      // Preserve the cyclic band's 4 MiB slots and 16 reducers. B200 remains
      // near 415/433 GB/s at 512 MiB/1 GiB with these settings.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CYCLIC_MM_CONSUMER_BLOCKS = 16;
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

    // On A100, all2all assigns one cyclic staging window to each destination.
    template<int World>
    struct TendonAll2All : BaseAll2All<World> {
      // A100 measurements favored eight slots per window: four larger slots
      // reduced overlap and were 5% slower at 512 MiB, while smaller slots paid
      // the drain cost more often. FALLBACK can run at any supported world size,
      // so it reserves space using the largest possible divisor.
      static constexpr size_t CYCLIC_CHUNK_SIZE =
        (MAX_STAGING_SIZE / (World < 2 ? MAX_RANKS_PER_DOMAIN : World)) / 8;
      // As in TendonAllGather, "deephalf" trades half the consumers for twice
      // the pipeline depth. all2all cannot tune consumers only for the chunked
      // band, so the change applies to every band. Across the full 1 KiB-1 GiB
      // sweep, the worst regression was 3%; at world size 2, performance
      // improved from 1.20x to 1.27x baseline performance.
      static constexpr int CHUNKED_PIPE_STAGES =
        (World == 2 || World == 4 || World == 8) ? 16 : 0;
      static constexpr int MAX_CONSUMER_BLOCKS =
        World == 2 ? 16 : World == 4 ? 4 : World == 8 ? 2
                                        : BaseAll2All<World>::MAX_CONSUMER_BLOCKS;
      // Each rank reads from every other rank, so the reader count is the
      // per-peer consumer count multiplied by world size minus one.
      static_assert(World < 2 || (World - 1) * (World == 2 ? 16 : World == 4 ? 4 : 2)
        >= MIN_SATURATION_READERS_SM80,
        "per-peer consumer cap below the Ampere read-saturation floor");
    };

    template<int World>
    struct LigamentAll2All : BaseAll2All<World> {};

    template<>
    struct LigamentAll2All<8> : BaseAll2All<8> {
      // H100 benchmarks showed that using two consumers per peer improves the
      // non-chunked band by reducing contention between remote reads. In the
      // chunked band, two consumers per peer with a 16-stage pipeline match the
      // performance of four consumers per peer and reduce the grid from 60
      // blocks to 46.
      static constexpr int MAX_CONSUMER_BLOCKS = 2;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      // The cyclic band performs best with 2 MiB slots. Using 1 MiB slots was
      // about 4% slower with 256 MiB staged and about 7% slower when each
      // staging half was smaller.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      // Two consumers across seven remote peers provide 14 readers, safely
      // above Hopper's minimum of eight.
      static_assert(MAX_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAll2All<4> : BaseAll2All<4> {
      // At world size 4, four consumers per peer with a 16-stage pipeline stay
      // within the relaxed 8% regression limit; the worst case was 6.0% at the
      // largest measured size. Twelve readers keep 768 KiB in flight at
      // 64 KiB each, slightly below the 1 MiB target. A pipeline of roughly
      // 24 stages might close the remaining gap.
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static_assert(MAX_CONSUMER_BLOCKS * 3 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAll2All<2> : BaseAll2All<2> {
      // With only one peer, 32 consumers oversubscribe the link. Reducing the
      // count to 16 makes the largest measured transfer 8% faster without
      // slowing smaller transfers. This is the same congestion point observed
      // at world size 8.
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
    };

    template<int World>
    struct CortexAll2All : BaseAll2All<World> {
    };

    // all2allV uses the same per-stream protocol on B300; only its performance
    // settings differ.
    template<int World>
    struct CortexAll2AllV : BaseAll2All<World> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      // B300 keeps 1 MiB chunks. Hopper's 512 KiB setting was 5-6% faster for
      // rows from 2-8 MiB, but 7-44% slower for rows from 16-128 MiB because it
      // doubled the number of chunks at twice the wire rate. all2allV must use
      // one chunk size because both peers rely on the chunk count as protocol
      // state.
      //
      // B300 also raises the per-stream threshold from Hopper's 128 KiB to
      // 512 KiB. Its extra bandwidth can absorb the packet path's doubled wire
      // traffic: 1 MiB rows improved by 19% (from 1.21x to 1.49x baseline) and
      // 2 MiB rows by 24% (from 0.94x to 1.23x), with no meaningful change from
      // 4-128 MiB. A 256 KiB threshold showed the same pattern one size lower.
      static constexpr size_t PER_STREAM_THRESHOLD = 512UL * 1024UL;
    };

    template<>
    struct CortexAll2AllV<8> : CortexAll2AllV<0> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      // Use four producers per peer with eight ranks, matching Hopper. The
      // inherited 16-block setting gave each peer only two. On eight B200s
      // with non-symmetric buffers and Zipf exponent 0.125, four producers
      // improved performance by 6.8-23.2% at totals of 128 KiB-128 MiB.
      //
      // On the same node, transfers below 32 KiB took about 0.08 us longer
      // (0.5-1.6%). With 90% of traffic sent to the next rank (a2avskew),
      // latency rose by 9.5%, 4.4%, and 2.2% at 8, 32, and 64 KiB per rank,
      // but fell by 12% at 512 KiB. Two and four ranks keep 16 producer blocks,
      // which also keeps their grids within MAX_NUM_CTAS.
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      // Keep enough consumers as streams grow. Without this minimum, a
      // largest stream of 64 KiB to under 2 MiB gets too few consumers, and
      // bandwidth drops from 38.7 to 33.9 GB/s as the total grows from 256 to
      // 512 KiB. Enabling the minimum raises throughput to 62.2 GB/s at
      // 512 KiB (83% higher) and 87.2 GB/s at 1 MiB (29% higher).
      static constexpr bool CONSUMER_FLOOR = true;
      static constexpr size_t CYCLIC_STREAM_CHUNK = 4UL * 1024UL * 1024UL;
    };

    template<>
    struct CortexAll2AllV<4> : BaseAll2All<4> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t PER_STREAM_THRESHOLD = 512UL * 1024UL;
      static constexpr size_t LATENCY_THRESHOLD = 288UL * 1024UL;
      // As at world size 8, B300 uses 1 MiB chunks here. Using 512 KiB chunks
      // was 5-14% slower for rows of 16 MiB or larger.
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr int THREADS = 256;
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
      // Larger cyclic slots amortize the per-slot drain, just as they do at
      // world size 8. For a 512 MiB total on B300, they reduced latency from
      // 1079 us to 783 us and improved from 0.78x to 1.07x baseline performance.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 8UL * 1024UL * 1024UL;
    };

    template<>
    struct CortexAll2All<8> : BaseAll2All<8> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      // For cyclic transfers totaling more than 512 MiB, B300 benchmarks found
      // gains of 20%, 11%, and 4% as the slot size doubled from 1 to 2, 4, and
      // finally 8 MiB. An 8 MiB slot gives each window four slots and improves
      // the band from 0.68-0.70x to 1.01-1.02x baseline performance. This
      // setting applies only to the cyclic band; the resident bands keep their
      // existing chunk size.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 8UL * 1024UL * 1024UL;
    };

    template<int World>
    struct BaseReduceScatter : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      // At 128 threads, this reaches the multimem congestion point of roughly
      // 16,384 outstanding operations.
      static constexpr int MM_DEPTH = 8;
      static constexpr int MM_CONSUMER_BLOCKS = 16;
    };

    template<>
    struct BaseReduceScatter<2> : CodesignPolicyBase {
      static constexpr int THREADS = 256;
      // Leave room above the nominal power-of-two sizes because reduceScatterV
      // chooses a policy using the largest per-rank transfer (maxBytes).
      static constexpr size_t LATENCY_THRESHOLD = 576UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 5UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
      // A multimem reduction reads both replicas through the switch, producing
      // twice the egress traffic of a unicast read. At fan-out 1 this was
      // 23-31% slower, so multimem is disabled.
      static constexpr size_t MM_MAX_BYTES = 0UL;
    };

    template<>
    struct BaseReduceScatter<4> : CodesignPolicyBase {
      // The non-chunked unicast path is faster than the scattered
      // latency path (LR) from 512 KiB per rank upward. A 384 KiB
      // threshold leaves enough room for reduceScatterV's maxBytes value,
      // which includes size skew, to stay below that crossover point.
      static constexpr size_t LATENCY_THRESHOLD = 384UL * 1024UL;
      // Use the non-chunked path through 2 MiB per rank, then switch to 1 MiB
      // chunks. A variable-size dispatch just above the threshold will still
      // have at least two chunks to pipeline instead of serializing one chunk.
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      // At world size 4, multimem produces 4/3 as much egress traffic as
      // unicast and is slower at every size in this band. At world size 8 the
      // smaller 8/7 penalty is offset by better instruction efficiency.
      static constexpr size_t MM_MAX_BYTES = 0UL;
    };

    // On A100, reduceScatter assigns one cyclic staging window to each rank.
    template<int World>
    struct TendonReduceScatter : BaseReduceScatter<World> {
      // A100 measurements favored eight slots per window: four larger slots
      // reduced overlap and were 5% slower at 512 MiB, while smaller slots paid
      // the drain cost more often. FALLBACK can run at any supported world size,
      // so it reserves space using the largest possible divisor.
      static constexpr size_t CYCLIC_CHUNK_SIZE =
        (MAX_STAGING_SIZE / (World < 2 ? MAX_RANKS_PER_DOMAIN : World)) / 8;
    };

    template<int World>
    struct LigamentReduceScatter : BaseReduceScatter<World> {};

    template<>
    struct LigamentReduceScatter<8> : BaseReduceScatter<8> {
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      // H100 benchmarks showed that 16 reducers with 16-stage pipelines keep
      // roughly 1 MiB of remote reads in flight, the same as 32 reducers with
      // eight stages and approximately equal to the fabric's bandwidth-delay
      // product. This 48-block configuration performs within 2.7% of the
      // original 64-block grid.
      //
      // Halving the reducers without deepening the pipeline loses 8-16%
      // because it covers only half of the bandwidth-delay product. Deepening
      // the pipeline without reducing the consumers oversubscribes the read
      // queue, and an intermediate count of 24 reducers performs especially
      // poorly. The deeper pipeline costs eight registers, causes no spills,
      // and increases shared-memory use from 32 KiB to 64 KiB.
      //
      // Only the chunked bands use the deeper pipeline. The multimem put band
      // and the non-chunked band used when multimem is unavailable stage too
      // little data per block to recover the extra pipeline startup cost.
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      // With 16 consumers, the cyclic band needs larger slots to amortize the
      // drain round trip after each slot. A 4 MiB slot matches the 32-consumer
      // baseline, while a 2 MiB slot is 2% slower.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90,
        "reducer count below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentReduceScatter<4> : BaseReduceScatter<4> {
      // At world size 4, a 48-block configuration with 16 reducers, 16-stage
      // chunked pipelines, and 2 MiB chunks stays within the relaxed 8%
      // regression limit. Its worst end-to-end regression was 5.0%. Multimem
      // is unavailable at this world size, so the non-chunked band remains in
      // use and retains its shallow pipeline.
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90,
        "reducer count below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentReduceScatter<2> : BaseReduceScatter<2> {
      // At world size 2, a 48-block configuration with 16 reducers, 16-stage
      // chunked pipelines, and 4 MiB chunks stays within the relaxed 8%
      // regression limit. Its worst measured regression was 5.0%.
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
      // B300 benefits from larger cyclic slots and a deep pipeline used only in
      // the cyclic band. For a 512 MiB total, this reduced latency from 1374 us
      // to 707 us and improved from 0.53x to 1.03x baseline performance.
      // reduceScatterV uses the same settings and reaches 0.99x baseline
      // performance.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 8UL * 1024UL * 1024UL;
      static constexpr int CYCLIC_PIPE_STAGES = 16;
    };

    template<>
    struct CortexReduceScatter<8> : BaseReduceScatter<8> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      // Keep 8 MiB slots for cyclic transfers above 256 MiB total. They were
      // 13% faster than 2 MiB slots on B300 and remain the best tested size on
      // B200. At a 512 MiB total, 4 MiB and 16 MiB slots were 2.1% and 2.6%
      // slower, respectively.
      //
      // Eight pipeline stages performed best on eight B200s (SM100).
      // Tests with non-symmetric buffers on 2026-09-22 measured:
      //   512 MiB total: 641.5 GB/s at depth 8, versus 406.6 at depth 16.
      //   1 GiB total:   666.2 GB/s at depth 8, versus 415.7 at depth 16.
      // These are gains of 57.8% and 60.3%, reaching 0.95x and 0.93x baseline
      // performance; depth 16 reached about 0.60x. Four stages reached only
      // 422.7 and 432.4 GB/s. Leave the override at 0 to use the default eight
      // stages.
      static constexpr size_t CYCLIC_CHUNK_SIZE = 8UL * 1024UL * 1024UL;
      static constexpr int CYCLIC_PIPE_STAGES = 0;
    };
    // The SM90 variable-size collectives generally keep their original
    // shallow, wide configurations. With uneven splits, the largest per-rank
    // transfer can contain only enough data to fill a deep pipeline once, so
    // using a deep pipeline with fewer consumers often hurts performance.
    // The exceptions and variable-size band thresholds are defined below.
    template<int World>
    struct LigamentAllGatherV : BaseAllGather<World> {};

    template<>
    struct LigamentAllGatherV<8> : BaseAllGather<8> {
      // Random-skew benchmarks from 2026-08-27 showed that the deep
      // configuration should be limited to dispatches where maxBytes is at
      // least 8 MiB. In that range, two consumers per peer and a 16-stage
      // pipeline reduce the grid from 48 blocks to 32.
      //
      // Applying this configuration to every variable-size transfer was not
      // reliable. With two consumers per peer, the non-chunked band varied
      // from 8% faster to 27% slower for skewed rows of 512 KiB to 2 MiB. Rows
      // just above the threshold at 4 MiB, where each contribution contains
      // only one chunk, were consistently 10.8% slower.
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int CHUNKED_CONSUMER_BLOCKS = 2;
      static constexpr size_t DEEP_CHUNK_MIN_BYTES = 8UL * 1024UL * 1024UL;
      static_assert(CHUNKED_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
    };

    template<>
    struct LigamentAllGatherV<4> : BaseAllGather<4> {
      // allGatherV selects its throughput path using sizes from the context.
      // At 256-512 KiB per rank, the packet path is faster and more consistent.
      // Its throughput path also works best with 2 MiB chunks, while
      // fixed-size allGather prefers 4 MiB.
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
    };

    // Streams below the packet threshold need enough consumers to drain
    // packets promptly. Halving the consumers slows this latency-bound work.
    template<int World>
    struct TendonAll2AllV : TendonAll2All<World> {
      static constexpr int CHUNKED_PIPE_STAGES = 0;
      static constexpr int MAX_CONSUMER_BLOCKS = BaseAll2All<World>::MAX_CONSUMER_BLOCKS;
      // On eight A100s with Zipf exponent 0.125, four producers per peer were
      // 4-19% faster than two at totals of 1-8 MiB. A minimum consumer count
      // also prevents larger streams from losing workers: a largest stream
      // of 32-128 KiB previously got one consumer per peer, while smaller
      // streams got four. At 256 KiB total, latency fell from 1.75x that of
      // fixed-size all2all to matching it.
      //
      // Enable both changes only for eight ranks, where they were measured.
      // With two ranks, 32 staging blocks, 8 local blocks, and 32 consumers
      // would total 72 blocks, exceeding MAX_NUM_CTAS. getBlocks can then throw
      // only on the rank with the larger incoming stream, leaving its peer
      // waiting indefinitely during one-way traffic.
      static constexpr int CHUNKED_PUT_BLOCKS = World == 8 ? 32 : 16;
      static constexpr bool CONSUMER_FLOOR = World == 8;
      // Use two producers per peer for the largest streams. Four producers
      // share each chunk and were 2.7% slower at a 1 GiB total.
      static constexpr int LARGE_PUT_BLOCKS = World == 8 ? 16 : AUTO;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = World == 8 ?
        64UL * 1024UL * 1024UL : static_cast<size_t>(-1);
    };

    template<int World>
    struct LigamentAll2AllV : BaseAll2All<World> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
    };

    template<>
    struct LigamentAll2AllV<2> : BaseAll2All<2> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      // Benchmarks from 2026-08-27 showed the same single-peer congestion point
      // as the fixed-size path at world size 2. For rows of 16 MiB or larger,
      // using 16 consumers reduces the grid from 56 blocks to 40 and improves
      // performance by 2-11%. It is 5-9% slower in the mixed 1-4 MiB band,
      // where draining packets is latency-bound and benefits from more
      // consumers. Only the consumer count changes between these bands; both
      // use the same pipeline depth.
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 8UL * 1024UL * 1024UL;
      static constexpr int LARGE_CONSUMER_BLOCKS = 16;
    };

    template<>
    struct LigamentAll2AllV<4> : BaseAll2All<4> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t LATENCY_THRESHOLD = 288UL*1024UL;
      static constexpr size_t CHUNK_SIZE = 512UL*1024UL;
      // Per-stream benchmarks at world size 4 on 2026-08-27 (seed 12345)
      // showed 4-37% gains for skewed rows, with most cases reaching
      // 0.90-1.01x baseline performance. The 1 MiB case improved by 28%.
      // Deephalf banding has not been measured at this world size, so it is not
      // enabled here.
    };

    template<>
    struct LigamentAll2AllV<8> : BaseAll2All<8> {
      // On eight H200s, four producers per peer were 4-26% faster than two for
      // rows of 32 KiB-32 MiB. Tests on 2026-09-20 used uniform and aligned
      // Zipf splits; a uniform 1 MiB row fell from 16.1 to 13.2 us.
      //
      // Ranks sending a split of at least LARGE_CHUNK_MIN_BYTES keep two
      // producers per peer. Proportional assignment puts most producers on
      // the largest stream, where they share each chunk. With Zipf exponent 1
      // and rows of at least 32 MiB, 32 producer blocks delivered only
      // 0.60-0.76x the performance of the smaller producer count.
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int LARGE_PUT_BLOCKS = 16;
      // Keep the consumer count from dropping as streams grow. A largest
      // stream of 32-128 KiB previously got one consumer per peer, while
      // smaller streams got four. Enforcing a minimum cut latency for uniform
      // 256 KiB rows from 12.2 to 6.9 us. With four ranks, this made 512 KiB
      // rows with Zipf exponent 1 slower by 24%, so it stays disabled there.
      static constexpr bool CONSUMER_FLOOR = true;
      // all2allV must use one chunk size for every band. Selecting a chunk tier
      // from each rank's largest local split can make paired ranks calculate
      // different chunk counts under real skew, causing the operation to hang.
      // A 512 KiB chunk also pipelines 1-2 MiB splits that a 1 MiB chunk would
      // serialize, improving the 8 MiB total case by 12%.
      //
      // Neither a higher packet threshold nor a single staged transfer
      // improved the 1-4 MiB range against the baseline. Packets double wire
      // traffic with a flag per eight bytes; staging the whole transfer loses
      // the overlap between writing and consuming chunks. All tested paths
      // took about 19.5 us at a 1 MiB total, suggesting that exchanging size
      // information and synchronizing ranks dominates here.
      static constexpr size_t CHUNK_SIZE = 512UL * 1024UL;
      // Per-stream benchmarks on eight H100s from 2026-08-27 (seed 12345)
      // reached at least 1.02x baseline performance for uniform rows at every
      // size; the 1 MiB case improved from 0.85x to 1.05x. For skewed rows of
      // 8 MiB and larger, performance improved from 0.60-0.71x to 0.85-1.08x.
      // Smaller 256 KiB chunks were clearly slower from 16 MiB upward.
      //
      // Use the deeper pipeline with fewer consumers only when maxOut reaches
      // 8 MiB (rows of 64 MiB or more in these tests). This reduces the grid
      // from 46 blocks to 32 and improves performance by another 1-5%.
      // Smaller transfers keep more consumers and a shallow pipeline: halving
      // the consumers was 20-93% slower for 512 KiB-1 MiB rows, where packet
      // draining is latency-bound.
      // Small staged slices also leave much of the 64 KiB pipeline unused.
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 8UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int LARGE_CONSUMER_BLOCKS = 2;
      static_assert(LARGE_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90,
        "per-peer consumer cap below the Hopper read-saturation floor");
      // Use 4 MiB slots when a stream outgrows its 32 MiB staging window, so
      // it waits for readers less often than with 512 KiB slots. This cut
      // latency for uniform 1 GiB rows from 3493 to 2916 us, close to the
      // fixed-size all2all result of 2948 us. With Zipf exponent 1, latency
      // fell from 5601 to 3516 us. Compared with 4 MiB slots, 2 MiB slots were
      // 8% slower and 8 MiB slots were within 3%.
      // Streams that fit in their window keep CHUNK_SIZE; larger chunks made
      // uniform rows of 64-256 MiB slower by 2-6%.
      static constexpr size_t CYCLIC_STREAM_CHUNK = 4UL * 1024UL * 1024UL;
      // Assign blocks evenly for smaller rows, even with uneven splits.
      // Proportional assignment costs about 2.5 us to set up. Skipping it for
      // Zipf exponent 1 cut latency from 28.2 to 18.2 us at 2 MiB and from
      // 6.5 to 4.0 us at sizes up to 64 KiB.
      // Large rows benefit from proportional assignment: without it, a 1 GiB
      // row sending 90% of traffic to the next rank took 3.4 times as long.
      static constexpr size_t WEIGHTED_MAPPING_MIN_BYTES = LARGE_CHUNK_MIN_BYTES;
    };

    template<int World>
    struct LigamentReduceScatterV : BaseReduceScatter<World> {};

    template<>
    struct LigamentReduceScatterV<8> : BaseReduceScatter<8> {
      // reduceScatterV never uses the multimem band. Transfers smaller than
      // 2 MiB therefore use non-chunked unicast, where the startup and drain
      // cost of a deep pipeline makes it 5-13% slower. This band keeps the base
      // configuration of 32 consumers and a shallow pipeline.
      //
      // The fixed-size path's chunk sizes and band thresholds also produced
      // 18-30% gains for variable-size transfers. Benchmarks from 2026-08-27
      // showed that only the chunked and cyclic bands should use the fixed
      // path's deephalf configuration of 16 reducers and a 16-stage pipeline.
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int CHUNKED_CONSUMER_BLOCKS = 16;
      static constexpr size_t CYCLIC_CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static_assert(CHUNKED_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90,
        "reducer count below the Hopper read-saturation floor");
    };
  } // namespace detail

  template<int CodesignArch, int World>
  struct AllGatherCodesign : detail::BaseAllGather<World>{};

  template<int World>
  struct AllGatherCodesign<800, World> : detail::TendonAllGather<World> {
  };

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

  // Keep the base threshold for allGatherV until it has been retuned. The
  // 64 KiB threshold was measured only for fixed-size allGather. allGatherV
  // chooses a path using the largest partition, so uneven sizes can cross a
  // shared threshold at a smaller total than in the fixed-size tests.
  template<>
  struct AllGatherVCodesign<800, 8> : AllGatherCodesign<800, 8> {
    static constexpr size_t LATENCY_THRESHOLD = detail::BaseAllGather<8>::LATENCY_THRESHOLD;
  };

  template<>
  struct AllGatherVCodesign<1000, 8> : AllGatherCodesign<1000, 8> {
    static constexpr size_t LATENCY_THRESHOLD = 288UL * 1024UL;
  };

  template<int CodesignArch>
  struct AllGatherVCodesign<CodesignArch, 4> : AllGatherCodesign<CodesignArch, 4> {
    // allGatherV selects its throughput path using sizes from the context.
    // At 256-512 KiB per rank, the packet path is faster and more consistent.
    // Its throughput path also works best with 2 MiB chunks, while
    // fixed-size allGather prefers 4 MiB.
    static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
    static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
  };

  // Resolve the overlap between the <900, World> and <CodesignArch, 4>
  // specializations.
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
  struct All2AllCodesign<800, World> : detail::TendonAll2All<World> {
  };

  template<int World>
  struct All2AllCodesign<900, World> : detail::LigamentAll2All<World> {
  };

  template<int World>
  struct All2AllCodesign<1000, World> : detail::CortexAll2All<World> {
  };

  template<int CodesignArch, int World>
  struct All2AllVCodesign : detail::BaseAll2All<World> {
    static constexpr int CHUNKED_PUT_BLOCKS = 16;
  };

  template<int World>
  struct All2AllVCodesign<800, World> : detail::TendonAll2AllV<World> {
  };

  template<int World>
  struct All2AllVCodesign<900, World> : detail::LigamentAll2AllV<World> {
  };

  template<int World>
  struct All2AllVCodesign<1000, World> : detail::CortexAll2AllV<World> {
  };

  template<int CodesignArch>
  struct All2AllVCodesign<CodesignArch, 4> : detail::BaseAll2All<4> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t LATENCY_THRESHOLD = 288UL*1024UL;
      static constexpr size_t CHUNK_SIZE = 512UL*1024UL;
      // Keep one chunk size across all2allV bands so peers agree on how many
      // chunks to exchange. The old CHUNK_SIZE_LARGE override broke this rule
      // and prevented this policy from compiling outside Hopper.
  };

  // Resolve cases that match both an architecture policy and the four-rank
  // specialization.
  template<>
  struct All2AllVCodesign<800, 4> : detail::TendonAll2AllV<4> {
    static constexpr size_t LATENCY_THRESHOLD = 288UL * 1024UL;
    static constexpr size_t CHUNK_SIZE = 512UL * 1024UL;
  };

  template<>
  struct All2AllVCodesign<900, 4> : detail::LigamentAll2AllV<4> {
  };

  template<>
  struct All2AllVCodesign<1000, 4> : detail::CortexAll2AllV<4> {
  };

  template<int CodesignArch, int World>
  struct ReduceScatterCodesign : detail::BaseReduceScatter<World> {};

  template<int World>
  struct ReduceScatterCodesign<800, World> : detail::TendonReduceScatter<World> {
  };

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


  template<>
  struct ReduceScatterVCodesign<1000, 8> : ReduceScatterCodesign<1000, 8> {
    static constexpr size_t LATENCY_THRESHOLD = 576UL * 1024UL;
  };
} // namespace purlin::host

#endif // PURLIN_HOST_CODESIGN_CUH
