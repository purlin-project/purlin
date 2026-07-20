#ifndef PURLIN_HOST_TUNING_CUH
#define PURLIN_HOST_TUNING_CUH

#include <cstddef>

#include "../configuration.cuh"

namespace purlin::host {

template<int NArch>
inline constexpr int tuningArch = NArch == 1000 ? 1000 : 900;

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

#if defined(PURLIN_TUNE_LATENCY_THRESHOLD)
inline constexpr size_t latencyThresholdOverride = PURLIN_TUNE_LATENCY_THRESHOLD;
#else
inline constexpr size_t latencyThresholdOverride = 0;
#endif
#if defined(PURLIN_TUNE_LR_THREADS)
inline constexpr int lrThreadsOverride = PURLIN_TUNE_LR_THREADS;
#else
inline constexpr int lrThreadsOverride = 0;
#endif
#if defined(PURLIN_TUNE_THREADS)
inline constexpr int threadsOverride = PURLIN_TUNE_THREADS;
#else
inline constexpr int threadsOverride = 0;
#endif
#if defined(PURLIN_TUNE_PIPE_STAGES)
inline constexpr int pipeStagesOverride = PURLIN_TUNE_PIPE_STAGES;
#else
inline constexpr int pipeStagesOverride = 0;
#endif
#if defined(PURLIN_TUNE_STAGE_EXTENT)
inline constexpr int stageExtentOverride = PURLIN_TUNE_STAGE_EXTENT;
#else
inline constexpr int stageExtentOverride = 0;
#endif
#if defined(PURLIN_TUNE_CHUNK_SIZE)
inline constexpr size_t chunkSizeOverride = PURLIN_TUNE_CHUNK_SIZE;
#else
inline constexpr size_t chunkSizeOverride = 0;
#endif
#if defined(PURLIN_TUNE_NON_CHUNKED_PUT_BLOCKS)
inline constexpr int nonChunkedPutBlocksOverride = PURLIN_TUNE_NON_CHUNKED_PUT_BLOCKS;
#else
inline constexpr int nonChunkedPutBlocksOverride = 0;
#endif
#if defined(PURLIN_TUNE_CHUNKED_PUT_BLOCKS)
inline constexpr int chunkedPutBlocksOverride = PURLIN_TUNE_CHUNKED_PUT_BLOCKS;
#else
inline constexpr int chunkedPutBlocksOverride = 0;
#endif
#if defined(PURLIN_TUNE_LOCAL_PUT_BLOCKS)
inline constexpr int localPutBlocksOverride = PURLIN_TUNE_LOCAL_PUT_BLOCKS;
#else
inline constexpr int localPutBlocksOverride = 0;
#endif
#if defined(PURLIN_TUNE_GATHER_BLOCKS)
inline constexpr int gatherBlocksOverride = PURLIN_TUNE_GATHER_BLOCKS;
#else
inline constexpr int gatherBlocksOverride = 0;
#endif
#if defined(PURLIN_TUNE_MAX_CONSUMER_BLOCKS)
inline constexpr int maxConsumerBlocksOverride = PURLIN_TUNE_MAX_CONSUMER_BLOCKS;
#else
inline constexpr int maxConsumerBlocksOverride = 0;
#endif

template<typename Base>
struct ApplyTuningOverrides : Base {
  static constexpr size_t LATENCY_THRESHOLD = latencyThresholdOverride > 0
    ? latencyThresholdOverride : Base::LATENCY_THRESHOLD;
  static constexpr int LR_THREADS = lrThreadsOverride > 0 ? lrThreadsOverride : Base::LR_THREADS;
  static constexpr int THREADS = threadsOverride > 0 ? threadsOverride : Base::THREADS;
  static constexpr int PIPE_STAGES = pipeStagesOverride > 0 ? pipeStagesOverride : Base::PIPE_STAGES;
  static constexpr int STAGE_EXTENT = stageExtentOverride > 0 ? stageExtentOverride : Base::STAGE_EXTENT;
  static constexpr size_t CHUNK_SIZE = chunkSizeOverride > 0 ? chunkSizeOverride : Base::CHUNK_SIZE;
  static constexpr int NON_CHUNKED_PUT_BLOCKS = nonChunkedPutBlocksOverride > 0
    ? nonChunkedPutBlocksOverride : Base::NON_CHUNKED_PUT_BLOCKS;
  static constexpr int CHUNKED_PUT_BLOCKS = chunkedPutBlocksOverride > 0
    ? chunkedPutBlocksOverride : Base::CHUNKED_PUT_BLOCKS;
  static constexpr int LOCAL_PUT_BLOCKS = localPutBlocksOverride > 0
    ? localPutBlocksOverride : Base::LOCAL_PUT_BLOCKS;
  static constexpr int GATHER_BLOCKS = gatherBlocksOverride > 0
    ? gatherBlocksOverride : Base::GATHER_BLOCKS;
  static constexpr int MAX_CONSUMER_BLOCKS = maxConsumerBlocksOverride > 0
    ? maxConsumerBlocksOverride : Base::MAX_CONSUMER_BLOCKS;
};

template<int World>
struct LegacyAllGather : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 1024;
  static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
  static constexpr int CHUNKED_PUT_BLOCKS = 16;
  static constexpr int MAX_CONSUMER_BLOCKS = 4;
  static constexpr int ALT_THREADS = 256;
  static constexpr size_t ALT_MIN_BYTES = 1024;
  static constexpr size_t ALT_MAX_BYTES = 64UL * 1024UL;
};

template<>
struct LegacyAllGather<2> : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
  static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
  static constexpr int CHUNKED_PUT_BLOCKS = 16;
  static constexpr int MAX_CONSUMER_BLOCKS = 16;
  static constexpr int ALT_THREADS = 256;
  static constexpr size_t ALT_MIN_BYTES = 32UL * 1024UL * 1024UL;
  static constexpr size_t ALT_MAX_BYTES = static_cast<size_t>(-1);
};

template<>
struct LegacyAllGather<4> : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
  static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
  static constexpr int CHUNKED_PUT_BLOCKS = 32;
  static constexpr int MAX_CONSUMER_BLOCKS = 8;
};

template<int World>
struct B300AllGatherBase : LegacyAllGather<World> {
  static constexpr int ALT_THREADS = 0;
  static constexpr size_t ALT_MIN_BYTES = 0;
  static constexpr size_t ALT_MAX_BYTES = 0;
};

template<int World>
struct B300AllGather : B300AllGatherBase<World> {};

template<>
struct B300AllGather<2> : B300AllGatherBase<2> {
  static constexpr size_t LATENCY_THRESHOLD = 2UL * 1024UL * 1024UL;
  static constexpr int STAGE_EXTENT = 8;
};

template<>
struct B300AllGather<4> : B300AllGatherBase<4> {
  static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
  static constexpr int THREADS = 256;
  static constexpr int MAX_CONSUMER_BLOCKS = 16;
};

template<>
struct B300AllGather<8> : B300AllGatherBase<8> {
  static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
  static constexpr int THREADS = 256;
  static constexpr int CHUNKED_PUT_BLOCKS = 32;
  static constexpr int MAX_CONSUMER_BLOCKS = 4;
};

template<int World>
struct LegacyAllReduce : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 128UL * 1024UL;
  static constexpr int THREADS = 256;
  static constexpr int STAGE_EXTENT = 1;
  static constexpr int CHUNKED_PUT_BLOCKS = 16;
  static constexpr int GATHER_BLOCKS = 16;
  static constexpr int MAX_CONSUMER_BLOCKS = 32;
};

template<>
struct LegacyAllReduce<2> : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
  static constexpr int THREADS = 256;
  static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
  static constexpr int MAX_CONSUMER_BLOCKS = 16;
};

template<>
struct LegacyAllReduce<4> : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
  static constexpr int THREADS = 256;
  static constexpr int STAGE_EXTENT = 1;
  static constexpr int GATHER_BLOCKS = 16;
};

template<int World>
struct B300AllReduce : LegacyAllReduce<World> {};

template<>
struct B300AllReduce<2> : LegacyAllReduce<2> {
  static constexpr size_t LATENCY_THRESHOLD = 2UL * 1024UL * 1024UL;
  static constexpr int STAGE_EXTENT = 4;
};

template<>
struct B300AllReduce<4> : LegacyAllReduce<4> {
  static constexpr size_t LATENCY_THRESHOLD = 1UL * 1024UL * 1024UL;
  static constexpr int STAGE_EXTENT = 2;
};

template<>
struct B300AllReduce<8> : LegacyAllReduce<8> {
  static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
  static constexpr int STAGE_EXTENT = 2;
};

template<int World>
struct LegacyAll2All : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 4UL * 1024UL;
  static constexpr int LOCAL_PUT_BLOCKS = 4;
  static constexpr int MAX_CONSUMER_BLOCKS = 4;
};

template<>
struct LegacyAll2All<2> : TuningPolicyBase {
  static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
  static constexpr int LOCAL_PUT_BLOCKS = 8;
};

template<>
struct LegacyAll2All<4> : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 256UL * 1024UL;
  static constexpr int LOCAL_PUT_BLOCKS = 4;
  static constexpr int MAX_CONSUMER_BLOCKS = 8;
};

template<int World>
struct B300All2All : LegacyAll2All<World> {};

template<>
struct B300All2All<2> : LegacyAll2All<2> {
  static constexpr size_t LATENCY_THRESHOLD = 1UL * 1024UL * 1024UL;
  static constexpr int THREADS = 256;
};

template<>
struct B300All2All<4> : LegacyAll2All<4> {
  static constexpr size_t LATENCY_THRESHOLD = 1UL * 1024UL * 1024UL;
  static constexpr int THREADS = 256;
  static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
};

template<>
struct B300All2All<8> : LegacyAll2All<8> {
  static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
  static constexpr int THREADS = 256;
};

template<int World>
struct LegacyReduceScatter : TuningPolicyBase {
  static constexpr size_t LATENCY_THRESHOLD = 64UL * 1024UL;
  static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
};

template<>
struct LegacyReduceScatter<2> : TuningPolicyBase {
  static constexpr int THREADS = 256;
  static constexpr size_t CHUNK_SIZE = 4UL * 1024UL * 1024UL;
  static constexpr int CHUNKED_PUT_BLOCKS = 16;
  static constexpr int MAX_CONSUMER_BLOCKS = 16;
};

template<>
struct LegacyReduceScatter<4> : TuningPolicyBase {
  static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
  static constexpr int CHUNKED_PUT_BLOCKS = 16;
};

template<int World>
struct B300ReduceScatter : LegacyReduceScatter<World> {};

template<>
struct B300ReduceScatter<2> : LegacyReduceScatter<2> {
  static constexpr size_t LATENCY_THRESHOLD = 2UL * 1024UL * 1024UL;
  static constexpr int STAGE_EXTENT = 4;
  static constexpr size_t CHUNK_SIZE = 2UL * 1024UL * 1024UL;
  static constexpr int CHUNKED_PUT_BLOCKS = 32;
  static constexpr int MAX_CONSUMER_BLOCKS = 32;
};

template<>
struct B300ReduceScatter<4> : LegacyReduceScatter<4> {
  static constexpr size_t LATENCY_THRESHOLD = 1UL * 1024UL * 1024UL;
  static constexpr int THREADS = 256;
};

template<>
struct B300ReduceScatter<8> : LegacyReduceScatter<8> {
  static constexpr size_t LATENCY_THRESHOLD = 512UL * 1024UL;
  static constexpr int THREADS = 256;
};

} // namespace detail

template<int TuningArch, int World>
struct AllGatherTuning;
template<int World>
struct AllGatherTuning<900, World> : detail::LegacyAllGather<World> {};
template<int World>
struct AllGatherTuning<1000, World> : detail::ApplyTuningOverrides<detail::B300AllGather<World>> {};

template<int TuningArch, int World>
struct AllReduceTuning;
template<int World>
struct AllReduceTuning<900, World> : detail::LegacyAllReduce<World> {};
template<int World>
struct AllReduceTuning<1000, World> : detail::ApplyTuningOverrides<detail::B300AllReduce<World>> {};

template<int TuningArch, int World>
struct All2AllTuning;
template<int World>
struct All2AllTuning<900, World> : detail::LegacyAll2All<World> {};
template<int World>
struct All2AllTuning<1000, World> : detail::ApplyTuningOverrides<detail::B300All2All<World>> {};

template<int TuningArch, int World>
struct ReduceScatterTuning;
template<int World>
struct ReduceScatterTuning<900, World> : detail::LegacyReduceScatter<World> {};
template<int World>
struct ReduceScatterTuning<1000, World> : detail::ApplyTuningOverrides<detail::B300ReduceScatter<World>> {};

} // namespace purlin::host

#endif // PURLIN_HOST_TUNING_CUH
