//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_BASE_CUH
#define PURLIN_BASE_CUH
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/ptx>
#include <cuda/utility>

#include "configuration.cuh"
#include "math.cuh"
#include "packet.cuh"

namespace purlin {
  // Select how allReduce is carried out. The direct path reduces the whole
  // payload on every rank: two ranks in the throughput regime, and the
  // full-buffer packet exchange in the latency regime. The composed path is a fused
  // reduceScatter followed by allGather.
  enum class AllReducePath {
    direct,
    composed
  };
  enum TensorType {
    bf16 = 0,
    fp16 = 1,
    fp8E4M3 = 2,
    fp8E5M2 = 3,
    fp32 = 4
  };
  enum class CollectiveType {
    chunked,
    nonChunked
  };
  // Select how a payload uses the staging buffer. Resident payloads fit within
  // one half of the buffer. Larger cyclic payloads reuse a fixed set of chunk
  // slots and wait for consumers before overwriting a slot.
  enum class StagingMode {
    resident,
    cyclic
  };
  // Select the store a throughput-regime reduction issues for its result.
  // multicast writes through the NVLS multicast mapping so every staging
  // replica receives it; unicast writes to dst. SNAC requests multicast only
  // when the datapath is multimem and a gather follows, so a unicast-only
  // Atom can assert that it is never asked for a multicast store.
  enum class ReduceResult {
    multicast,
    unicast
  };
  static constexpr size_t LAT_THRESHOLD_DEFAULT = 0;
  template<
    CollectiveType ct,
    int putBlocks,
    int gatherBlocks,
    size_t chunkSize,
    int localPutBlocks = 8,
    size_t latencyThreshold = LAT_THRESHOLD_DEFAULT,
    StagingMode stagingMode = StagingMode::resident,
    size_t perStreamThreshold = 0,
    size_t cyclicStreamChunk = 0,
    size_t weightedMappingMinBytes = 0
  >
  struct CollectiveConfig {
    static constexpr int PUT_BLOCKS = putBlocks;
    static constexpr int LOCAL_PUT_BLOCKS = localPutBlocks;
    static constexpr int GATHER_BLOCKS = gatherBlocks;
    static constexpr size_t CHUNK_SIZE = chunkSize;
    static constexpr size_t LATENCY_THRESHOLD = latencyThreshold;
    static constexpr CollectiveType COLLECTIVE_TYPE = ct;
    static constexpr StagingMode STAGING_MODE = stagingMode;
    // (scattered->transposed) selects a protocol for each stream. Streams up to this size use
    // packets that carry completion flags through the latency buffer. Larger
    // streams use fixed staging windows for each destination. Set this to 0 to
    // disable per-stream selection.
    static constexpr size_t PER_STREAM_THRESHOLD = perStreamThreshold;
    // A (scattered->transposed) stream larger than its staging window cycles through the
    // window in slots of this size instead of CHUNK_SIZE. Set this to 0 to use
    // CHUNK_SIZE for every stream.
    static constexpr size_t CYCLIC_STREAM_CHUNK = cyclicStreamChunk;
    // (scattered->transposed) divides its blocks among skewed streams in proportion to their
    // sizes only once the largest stream reaches this size. Below it every
    // stream gets the same number of blocks.
    static constexpr size_t WEIGHTED_MAPPING_MIN_BYTES = weightedMappingMinBytes;
    static constexpr Regime REGIME = Regime::throughput;
    static_assert(stagingMode == StagingMode::resident || ct == CollectiveType::chunked);
    static_assert(perStreamThreshold == 0 || ct == CollectiveType::chunked);
    // Each 16-byte packet contains eight bytes of payload. A source's latency
    // buffer must therefore have twice the capacity of the largest stream that
    // can use the packet protocol.
    static_assert(2 * perStreamThreshold <= PACKET_BUFFER_SIZE);
    static_assert(perStreamThreshold % 16 == 0);
    // Both slot sizes divide the same window.
    static_assert(cyclicStreamChunk == 0 ||
      (cyclicStreamChunk > chunkSize && cyclicStreamChunk % chunkSize == 0));
  };
  // The same configuration with CYCLIC_STREAM_CHUNK as its chunk size.
  template<typename CollConfig>
  struct CyclicStreamConfig : CollConfig {
    static constexpr size_t CHUNK_SIZE = CollConfig::CYCLIC_STREAM_CHUNK;
    static constexpr size_t CYCLIC_STREAM_CHUNK = 0;
  };
  // This sentinel configuration selects the fused latency protocol.
  using CollectiveConfigLR = void;
  template<typename CollConfig>
  inline constexpr Regime regimeOf = CollConfig::REGIME;
  template<>
  inline constexpr Regime regimeOf<CollectiveConfigLR> = Regime::latency;

  // These layouts describe how a buffer is divided among ranks. A collective
  // transforms one layout into another:
  //   reduceScatter: scattered -> packed
  //   allGather:     packed -> scattered
  //   all2all:       scattered -> transposed
  //   allReduce:     scattered -> scattered
  // Layouts ending in V contain variable-size rank partitions.
  enum class DataLayout {
    packed, // One contiguous payload with no rank partitioning.
    packedV, // A contiguous payload whose size varies by rank.
    scattered, // Partitioned by rank; slice r belongs to rank r.
    scatteredV, // Rank-partitioned with variable-size slices.
    transposed, // My slice r corresponds to rank r's slice addressed to me.
    transposedV // Transposed with variable-size slices.
  };

  // Use the multimem datapath only for element and operation pairs supported by
  // PTX. Packed f16 and bf16 support addition and maximum; f32 supports only
  // addition; multiplication has no multimem mapping. fp8 is intentionally
  // excluded because the switch accumulates it in f16 at best, while unicast
  // reduction uses f32 and follows a deterministic rank order.
  template<int NArch, typename Element, ReduceOp ro>
  consteval bool multimemReducible() {
    if (NArch < 900) {
      return false;
    }
    constexpr auto packed16 = cuda::std::is_same_v<Element, __half> ||
      cuda::std::is_same_v<Element, __nv_bfloat16>;
    if (ro == ReduceOp::add) {
      return packed16 || cuda::std::is_same_v<Element, float>;
    }
    return ro == ReduceOp::max && packed16;
  }

  template<int Arch>
  consteval auto normalizeArch() {
    if constexpr (Arch >= 1000) {
      return 1000;
    }
    if constexpr (Arch >= 900) {
      return 900;
    }
    if constexpr (Arch >= 800) {
      return 800;
    }
    return 700; // Use the generic implementation for older architectures.
  }

  template<int AlignmentBytes>
  requires(cuda::is_power_of_two(AlignmentBytes))
  struct AlignedType {
    using type = uint32_t;
  };

  template<>
  struct AlignedType<1> {
    using type = cuda::std::byte;
  };

  template<>
  struct AlignedType<2> {
    using type = uint16_t;
  };

  template<typename Element>
  struct DataToRawType {
    using type = Element;
  };

  template<>
  struct DataToRawType<__half> {
    using type = __half_raw;
  };

  template<>
  struct DataToRawType<__nv_bfloat16> {
    using type = __nv_bfloat16_raw;
  };

  template<>
  struct DataToRawType<__nv_fp8_e4m3> {
    using type = __nv_fp8_storage_t;
  };

  template<>
  struct DataToRawType<__nv_fp8_e5m2> {
    using type = __nv_fp8_storage_t;
  };

  template<>
  struct DataToRawType<__half2> {
    using type = __half2_raw;
  };

  template<>
  struct DataToRawType<__nv_bfloat162> {
    using type = __nv_bfloat162_raw;
  };

  template<>
  struct DataToRawType<__nv_fp8x2_e4m3> {
    using type = fp8x2_e4m3_raw;
  };

  template<>
  struct DataToRawType<__nv_fp8x2_e5m2> {
    using type = fp8x2_e5m2_raw;
  };

  template<typename RawType>
  struct RawToDataType {
    using type = RawType;
  };
  template<>
  struct RawToDataType<__half2_raw> {
    using type = __half2;
  };
  template<>
  struct RawToDataType<__nv_bfloat162_raw> {
    using type = __nv_bfloat162;
  };
  template<>
  struct RawToDataType<fp8x2_e4m3_raw> {
    using type = __nv_fp8x2_e4m3;
  };
  template<>
  struct RawToDataType<fp8x2_e5m2_raw> {
    using type = __nv_fp8x2_e5m2;
  };

  template<typename Element>
  struct PackedElement {
    using type = Element;
  };
  template<>
  struct PackedElement<float> {
    using type = float2;
  };
  template<>
  struct PackedElement<__half> {
    using type = __half2;
  };
  template<>
  struct PackedElement<__nv_bfloat16> {
    using type = __nv_bfloat162;
  };
  template<>
  struct PackedElement<__nv_fp8_e4m3> {
    using type = __nv_fp8x2_e4m3;
  };
  template<>
  struct PackedElement<__nv_fp8_e5m2> {
    using type = __nv_fp8x2_e5m2;
  };

  template<typename Element>
  __device__ __forceinline__
  auto load(const Element* __restrict__ const& src) {
    if constexpr (alignof(Element) > 16) {
      static_assert(sizeof(Element) == alignof(Element));
      return cuda::ptx::ld(cuda::ptx::space_global, src);
    }
    else {
      return *src;
    }
  }
  template<typename Element>
  __device__ __forceinline__
  void store(Element* __restrict__ const& dst, const Element& v) {
    if constexpr (alignof(Element) > 16) {
      static_assert(sizeof(Element) == alignof(Element));
      cuda::ptx::st(cuda::ptx::space_global, dst, v);
    }
    else {
      *dst = v;
    }
  }
  struct ReduceTRArgs {
    cuda::std::byte** const sources;
    // Multicast alias for this shard. It is valid only when the Atom's
    // configuration selects MemType::multimem.
    cuda::std::byte* const mcSource = nullptr;
    cuda::std::byte* const dst;
    const size_t bytesRed;
    // Byte offset applied to every source view. SNAC passes zero. A pipelined
    // Atom sets it to hand the tail of its range to the generic reducer.
    const size_t residualOffset = 0;
    const cuda::fast_mod_div<int, true> world;
  };

  template<typename T>
  using ReduceAccumType = cuda::std::common_type_t<float, T>;

}
#endif //PURLIN_BASE_CUH
