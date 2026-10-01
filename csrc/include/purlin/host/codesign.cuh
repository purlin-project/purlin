#ifndef PURLIN_HOST_CODESIGN_CUH
#define PURLIN_HOST_CODESIGN_CUH

namespace purlin::host {
  static constexpr int FALLBACK = 0;

  // Keep enough remote readers to use roughly 80% of the link's bandwidth in
  // one direction. Estimated at ceil(0.8 * link bandwidth / per-SM read-issue limit):
  //   Ampere:     8 readers (300 GB/s link, 32 GB/s per SM measured)
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
    // Larger streams use a staging window for each destination/
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
    // of 0 uses PIPE_STAGES.
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
    // LARGE_CHUNK_MIN_BYTES. AUTO uses MAX_CONSUMER_BLOCKS.
    static constexpr int LARGE_CONSUMER_BLOCKS = AUTO;
    // Producer blocks all2allV uses once the largest split a rank sends reaches
    // LARGE_CHUNK_MIN_BYTES. AUTO uses CHUNKED_PUT_BLOCKS.
    static constexpr int LARGE_PUT_BLOCKS = AUTO;
    // Maximum consumers in the chunked and cyclic bands of reduceScatter and
    // allGather. AUTO uses MAX_CONSUMER_BLOCKS.
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
    struct TendonAllGather : BaseAllGather<World> {
      static constexpr size_t LATENCY_THRESHOLD = World == 8 ?
        64UL * 1024UL : BaseAllGather<World>::LATENCY_THRESHOLD;
      static constexpr size_t CYCLIC_CHUNK_SIZE = MAX_STAGING_SIZE / 8;
      static constexpr int CHUNKED_PIPE_STAGES =
        (World == 2 || World == 4 || World == 8) ? 16 : 0;
      static constexpr int CHUNKED_CONSUMER_BLOCKS =
        World == 2 ? 8 : World == 4 ? 4 : World == 8 ? 2 : AUTO;
      // Each peer gets its own consumers, so the total reader count is the
      // per-peer count multiplied by the world size.
      static_assert(World < 2 || World * (World == 2 ? 8 : World == 4 ? 4 : 2)
        >= MIN_SATURATION_READERS_SM80);
    };

    template<int World>
    struct LigamentAllGather : BaseAllGather<World> {};

    template<>
    struct LigamentAllGather<8> : BaseAllGather<8> {
      static constexpr int MAX_CONSUMER_BLOCKS = 2;
      static constexpr int ALT_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static_assert(MAX_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90);
    };

    template<>
    struct LigamentAllGather<4> : BaseAllGather<4> {
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static_assert(MAX_CONSUMER_BLOCKS * 3 >= MIN_SATURATION_READERS_SM90);
    };

    template<>
    struct LigamentAllGather<2> : BaseAllGather<2> {
      static constexpr int MAX_CONSUMER_BLOCKS = 8;
      static constexpr int PIPE_STAGES = 16;
      static constexpr int ALT_CONSUMER_BLOCKS = 16;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90);
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
      // point we measured at world size 8.
      static constexpr int MM_DEPTH = 4;
      static constexpr int MM_CONSUMER_BLOCKS = 16;
      static constexpr int PACED_MM_DEPTH = 2;
      static constexpr size_t LR_PARTITION_MIN_BYTES = 128UL * 1024UL;
      static constexpr size_t LR_PARTITION_MAX_BYTES = 1UL * 1024UL * 1024UL;
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
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr int PACED_MM_DEPTH = 2;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
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
      static constexpr size_t CYCLIC_CHUNK_SIZE = MAX_STAGING_SIZE / 8;
    };

    template<>
    struct LigamentAllReduce<2> : BaseAllReduce<2> {
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
    };

    template<>
    struct LigamentAllReduce<4> : BaseAllReduce<4> {
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
      static constexpr int MM_DEPTH = 8;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 16UL * 1024UL * 1024UL;
      static constexpr size_t MID_CHUNK_MIN_BYTES = static_cast<size_t>(-1);
    };

    template<>
    struct CortexAllReduce<8> : BaseAllReduce<8> {
      static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
      static constexpr size_t LR_PARTITION_MIN_BYTES = 128UL * 1024UL;
      static constexpr size_t LR_PARTITION_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static_assert(2 * (LR_PARTITION_MAX_BYTES / 8) <= PACKET_BUFFER_SIZE / 2,
        "world-eight partitioned LR packets must fit both buffer halves");
      static_assert(LR_PARTITION_MAX_BYTES <= PACKET_BUFFER_SIZE / 2,
        "world-eight direct LR fallback must fit the packet buffer");
      static constexpr int LR_WIDE_THREADS = 512;
      static constexpr int LR_WIDE_BLOCKS_PER_PEER = 8;
      static_assert((8 - 1) * LR_WIDE_BLOCKS_PER_PEER <= MAX_NUM_CTAS,
        "world-eight LR launch exceeds the CTA limit");
      static constexpr int STAGE_EXTENT = 4;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int PACED_MM_DEPTH = 4;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 8UL * 1024UL * 1024UL;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 16UL * 1024UL * 1024UL;
      static constexpr int MM_CONSUMER_BLOCKS = 16;
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

    template<int World>
    struct TendonAll2All : BaseAll2All<World> {
      static constexpr size_t CYCLIC_CHUNK_SIZE =
        (MAX_STAGING_SIZE / (World < 2 ? MAX_RANKS_PER_DOMAIN : World)) / 8;
      static constexpr int CHUNKED_PIPE_STAGES =
        (World == 2 || World == 4 || World == 8) ? 16 : 0;
      static constexpr int MAX_CONSUMER_BLOCKS =
        World == 2 ? 16 : World == 4 ? 4 : World == 8 ? 2
                                        : BaseAll2All<World>::MAX_CONSUMER_BLOCKS;
      static_assert(World < 2 || (World - 1) * (World == 2 ? 16 : World == 4 ? 4 : 2)
        >= MIN_SATURATION_READERS_SM80);
    };

    template<int World>
    struct LigamentAll2All : BaseAll2All<World> {};

    template<>
    struct LigamentAll2All<8> : BaseAll2All<8> {
      static constexpr int MAX_CONSUMER_BLOCKS = 2;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr size_t CYCLIC_CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static_assert(MAX_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90);
    };

    template<>
    struct LigamentAll2All<4> : BaseAll2All<4> {
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static_assert(MAX_CONSUMER_BLOCKS * 3 >= MIN_SATURATION_READERS_SM90);
    };

    template<>
    struct LigamentAll2All<2> : BaseAll2All<2> {
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
    };

    template<int World>
    struct CortexAll2All : BaseAll2All<World> {
    };

    template<int World>
    struct CortexAll2AllV : BaseAll2All<World> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t PER_STREAM_THRESHOLD = 512UL * 1024UL;
    };

    template<>
    struct CortexAll2AllV<8> : CortexAll2AllV<0> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr bool CONSUMER_FLOOR = true;
      static constexpr size_t CYCLIC_STREAM_CHUNK = 4UL * 1024UL * 1024UL;
    };

    template<>
    struct CortexAll2AllV<4> : BaseAll2All<4> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t PER_STREAM_THRESHOLD = 512UL * 1024UL;
      static constexpr size_t LATENCY_THRESHOLD = 288UL * 1024UL;
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
      static constexpr size_t CYCLIC_CHUNK_SIZE = 8UL * 1024UL * 1024UL;
    };

    template<>
    struct CortexAll2All<8> : BaseAll2All<8> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
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
      // Leave room above power-of-two sizes because reduceScatterV
      // chooses a policy using the largest per-rank transfer (maxBytes).
      static constexpr size_t LATENCY_THRESHOLD = 576UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 5UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
      static constexpr size_t MM_MAX_BYTES = 0UL;
    };

    template<>
    struct BaseReduceScatter<4> : CodesignPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 384UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr size_t MM_MAX_BYTES = 0UL;
    };

    template<int World>
    struct TendonReduceScatter : BaseReduceScatter<World> {
      static constexpr size_t CYCLIC_CHUNK_SIZE =
        (MAX_STAGING_SIZE / (World < 2 ? MAX_RANKS_PER_DOMAIN : World)) / 8;
    };

    template<int World>
    struct LigamentReduceScatter : BaseReduceScatter<World> {};

    template<>
    struct LigamentReduceScatter<8> : BaseReduceScatter<8> {
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr size_t CYCLIC_CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90);
    };

    template<>
    struct LigamentReduceScatter<4> : BaseReduceScatter<4> {
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90);
    };

    template<>
    struct LigamentReduceScatter<2> : BaseReduceScatter<2> {
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static_assert(MAX_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90);
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
      static constexpr size_t CYCLIC_CHUNK_SIZE = 8UL * 1024UL * 1024UL;
      static constexpr int CYCLIC_PIPE_STAGES = 16;
    };

    template<>
    struct CortexReduceScatter<8> : BaseReduceScatter<8> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr size_t CYCLIC_CHUNK_SIZE = 8UL * 1024UL * 1024UL;
      static constexpr int CYCLIC_PIPE_STAGES = 0;
    };
    template<int World>
    struct LigamentAllGatherV : BaseAllGather<World> {};

    template<>
    struct LigamentAllGatherV<8> : BaseAllGather<8> {
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int CHUNKED_CONSUMER_BLOCKS = 2;
      static constexpr size_t DEEP_CHUNK_MIN_BYTES = 8UL * 1024UL * 1024UL;
      static_assert(CHUNKED_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90);
    };

    template<>
    struct LigamentAllGatherV<4> : BaseAllGather<4> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
    };

    template<int World>
    struct TendonAll2AllV : TendonAll2All<World> {
      static constexpr int CHUNKED_PIPE_STAGES = 0;
      static constexpr int MAX_CONSUMER_BLOCKS = BaseAll2All<World>::MAX_CONSUMER_BLOCKS;
      static constexpr int CHUNKED_PUT_BLOCKS = World == 8 ? 32 : 16;
      static constexpr bool CONSUMER_FLOOR = World == 8;
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
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 8UL * 1024UL * 1024UL;
      static constexpr int LARGE_CONSUMER_BLOCKS = 16;
    };

    template<>
    struct LigamentAll2AllV<4> : BaseAll2All<4> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t LATENCY_THRESHOLD = 288UL*1024UL;
      static constexpr size_t CHUNK_SIZE = 512UL*1024UL;
    };

    template<>
    struct LigamentAll2AllV<8> : BaseAll2All<8> {
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int LARGE_PUT_BLOCKS = 16;
      static constexpr bool CONSUMER_FLOOR = true;
      static constexpr size_t CHUNK_SIZE = 512UL * 1024UL;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 8UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int LARGE_CONSUMER_BLOCKS = 2;
      static_assert(LARGE_CONSUMER_BLOCKS * 7 >= MIN_SATURATION_READERS_SM90);
      static constexpr size_t CYCLIC_STREAM_CHUNK = 4UL * 1024UL * 1024UL;
      static constexpr size_t WEIGHTED_MAPPING_MIN_BYTES = LARGE_CHUNK_MIN_BYTES;
    };

    template<int World>
    struct LigamentReduceScatterV : BaseReduceScatter<World> {};

    template<>
    struct LigamentReduceScatterV<8> : BaseReduceScatter<8> {
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PIPE_STAGES = 16;
      static constexpr int CHUNKED_CONSUMER_BLOCKS = 16;
      static constexpr size_t CYCLIC_CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static_assert(CHUNKED_CONSUMER_BLOCKS >= MIN_SATURATION_READERS_SM90);
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
