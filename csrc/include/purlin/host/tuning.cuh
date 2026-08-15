//
// Created by osayamen on 7/19/26.
//
#ifndef PURLIN_HOST_TUNING_CUH
#define PURLIN_HOST_TUNING_CUH

namespace purlin::host {
  static constexpr int FALLBACK = 0;

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

  struct TuningPolicyBase {
    static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
    static constexpr int LR_THREADS = 512;
    static constexpr int THREADS = 128;
    static constexpr int PIPE_STAGES = 8;
    static constexpr int STAGE_EXTENT = 2;
    static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
    static constexpr size_t NON_CHUNKED_MAX_BYTES = 0;
    static constexpr size_t CHUNK_SIZE_MID = 0;
    static constexpr size_t CHUNK_SIZE_LARGE = 0;
    static constexpr size_t MID_CHUNK_MIN_BYTES = static_cast<size_t>(-1);
    static constexpr size_t LARGE_CHUNK_MIN_BYTES = static_cast<size_t>(-1);
    static constexpr int MM_DEPTH = AUTO;
    static constexpr int PACED_MM_DEPTH = 1;
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
    struct BaseAllGather : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 1024;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
      static constexpr int ALT_THREADS = 256;
      static constexpr size_t ALT_MIN_BYTES = 1024;
      static constexpr size_t ALT_MAX_BYTES = 64UL * 1024UL;
    };

    template<>
    struct BaseAllGather<2> : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
      static constexpr int ALT_THREADS = 256;
      static constexpr size_t ALT_MIN_BYTES = 32UL * 1024UL * 1024UL;
      static constexpr size_t ALT_MAX_BYTES = static_cast<size_t>(-1);
    };

    template<>
    struct BaseAllGather<4> : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 256UL*1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL*1024UL*1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 8;
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
    struct BaseAllReduce : TuningPolicyBase {
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
    struct BaseAllReduce<2> : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
      // Broadcasting packets to a single peer gains nothing past the latency floor.
      static constexpr size_t LR_MM_MAX_BYTES = 32UL * 1024UL;
    };

    template<>
    struct BaseAllReduce<4> : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr int STAGE_EXTENT = 1;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 16;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int GATHER_BLOCKS = 16;
      // The shard is bytes/4, so chunk tiers sit an octave below the world-8 policy:
      // 512K chunks keep the 4M shard-serialization cliff away, 1M chunks serve
      // 16M-32M, and the deep multimem pipeline takes over at 64M with 2M chunks.
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

    template<>
    struct TendonAllReduce<2> : TuningPolicyBase {
      static constexpr int THREADS = 256;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int STAGE_EXTENT = 2;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
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
    struct BaseAll2All : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
      static constexpr int LOCAL_PUT_BLOCKS = 4;
      static constexpr int MAX_CONSUMER_BLOCKS = 4;
    };

    template<>
    struct BaseAll2All<2> : TuningPolicyBase {
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static constexpr int NON_CHUNKED_PUT_BLOCKS = 16;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int LOCAL_PUT_BLOCKS = 8;
    };

    template<>
    struct BaseAll2All<4> : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
      static constexpr int LOCAL_PUT_BLOCKS = 4;
      static constexpr int MAX_CONSUMER_BLOCKS = 8;
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
    struct BaseReduceScatter : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      // Sized to the ~16K-outstanding multimem congestion knee at 128 threads.
      static constexpr int MM_DEPTH = 8;
      static constexpr int MM_CONSUMER_BLOCKS = 16;
    };

    template<>
    struct BaseReduceScatter<2> : TuningPolicyBase {
      static constexpr int THREADS = 256;
      // Band edges carry headroom over the nominal power-of-two sizes because
      // reduceScatterV dispatches on maxBytes, which sits slightly above them;
      // without it V falls one band up at every boundary size.
      static constexpr size_t LATENCY_THRESHOLD = 576UL * 1024UL;
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 5UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      // The put stage stages the whole 2S input; it and the consumers must scale
      // together (32/32) - either alone regresses the chunked band.
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
      // Multimem reduce pulls both replicas through the switch (2S egress vs the
      // 1S of a unicast read) - a 23-31% loss at fan-out 1, so it is disabled.
      static constexpr size_t MM_MAX_BYTES = 0UL;
    };

    template<>
    struct BaseReduceScatter<4> : TuningPolicyBase {
      // The unicast non-chunked path beats scattered LR from 512K per rank up;
      // the 384K edge keeps reduceScatterV's maxBytes (nominal + skew) below it.
      static constexpr size_t LATENCY_THRESHOLD = 384UL * 1024UL;
      // Non-chunked to 2M per rank, then 1M chunks: with 1M chunks a V dispatch
      // that lands just past the edge still pipelines (2+ chunks) instead of
      // serializing as a single chunk.
      static constexpr size_t NON_CHUNKED_MAX_BYTES = 2UL * 1024UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 32;
      // 4/3 egress penalty vs unicast loses at every size in the band (unlike
      // world 8's 8/7, which instruction efficiency pays for).
      static constexpr size_t MM_MAX_BYTES = 0UL;
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
  } // namespace detail

  template<int TuningArch, int World>
  struct AllGatherTuning : detail::BaseAllGather<World>{};

  template<int World>
  struct AllGatherTuning<1000, World> : detail::CortexAllGather<World> {
  };

  template<int TuningArch, int World>
  struct AllGatherVTuning : AllGatherTuning<TuningArch, World> {
  };

  template<int TuningArch>
  struct AllGatherVTuning<TuningArch, 4> : AllGatherTuning<TuningArch, 4> {
    // The V gather's TR entry sizes are context-sensitive where the fixed path is
    // not; latency-regime serves 256K-512K per rank faster and stably. 2M chunks
    // suit the V consumer where fixed AllGather prefers 4M.
    static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
    static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
  };

  template<int TuningArch, int World>
  struct AllReduceTuning : detail::BaseAllReduce<World>{};

  template<int World>
  struct AllReduceTuning<800, World> : detail::TendonAllReduce<World> {
  };

  template<int World>
  struct AllReduceTuning<1000, World> : detail::CortexAllReduce<World> {
  };

  template<int TuningArch, int World>
  struct All2AllTuning : detail::BaseAll2All<World>{};

  template<int World>
  struct All2AllTuning<1000, World> : detail::CortexAll2All<World> {
  };

  template<int TuningArch, int World>
  struct All2AllVTuning : All2AllTuning<TuningArch, World> {
    static constexpr int CHUNKED_PUT_BLOCKS = 16;
  };

  template<int TuningArch>
  struct All2AllVTuning<TuningArch, 4> : All2AllTuning<TuningArch, 4> {
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr size_t LATENCY_THRESHOLD = 288UL*1024UL;
      static constexpr size_t CHUNK_SIZE = 512UL*1024UL;
      static constexpr size_t LARGE_CHUNK_MIN_BYTES = 3UL*1024UL*1024UL;
      static constexpr size_t CHUNK_SIZE_LARGE = 1UL*1024UL*1024UL;
    };

  template<int TuningArch, int World>
  struct ReduceScatterTuning : detail::BaseReduceScatter<World> {};

  template<int World>
  struct ReduceScatterTuning<1000, World> : detail::CortexReduceScatter<World> {
  };
} // namespace purlin::host

#endif // PURLIN_HOST_TUNING_CUH
