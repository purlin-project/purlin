//
// Created by osayamen on 7/19/26.
//
#ifndef PURLIN_HOST_TUNING_CUH
#define PURLIN_HOST_TUNING_CUH

namespace purlin::host {
  static constexpr int FALLBACK = 0;

  struct TuningPolicyBase {
    static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
    static constexpr int LR_THREADS = 512;
    static constexpr int THREADS = 128;
    static constexpr int PIPE_STAGES = 8;
    static constexpr int STAGE_EXTENT = 2;
    static constexpr size_t CHUNK_SIZE = 1UL * 1024UL * 1024UL;
    static constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
    static constexpr int CHUNKED_PUT_BLOCKS = 32;
    static constexpr int LOCAL_PUT_BLOCKS = UNUSED;
    static constexpr int GATHER_BLOCKS = UNUSED;
    static constexpr int MAX_CONSUMER_BLOCKS = 32;
    static constexpr int ALT_THREADS = 0;
    static constexpr size_t ALT_MIN_BYTES = 0;
    static constexpr size_t ALT_MAX_BYTES = 0;
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
      static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
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
    };

    template<>
    struct CortexAllGather<4> : CortexAllGatherBase<4> {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
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
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int GATHER_BLOCKS = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 32;
    };

    template<>
    struct BaseAllReduce<2> : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
    };

    template<>
    struct BaseAllReduce<4> : TuningPolicyBase {
      static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
      static constexpr int THREADS = 256;
      static constexpr int STAGE_EXTENT = 1;
      static constexpr int GATHER_BLOCKS = 16;
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
    };

    template<>
    struct BaseReduceScatter<2> : TuningPolicyBase {
      static constexpr int THREADS = 256;
      static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
      static constexpr int MAX_CONSUMER_BLOCKS = 16;
    };

    template<>
    struct BaseReduceScatter<4> : TuningPolicyBase {
      static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
      static constexpr int CHUNKED_PUT_BLOCKS = 16;
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

  template<int TuningArch, int World>
  struct ReduceScatterTuning : detail::BaseReduceScatter<World> {};

  template<int World>
  struct ReduceScatterTuning<1000, World> : detail::CortexReduceScatter<World> {
  };
} // namespace purlin::host

#endif // PURLIN_HOST_TUNING_CUH
