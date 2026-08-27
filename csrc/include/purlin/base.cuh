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
  enum class World2Bypass {
    yes,
    no,
    unknown
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
  // Whether the staged payload fits a staging half (resident), or must
  // wrap through it as cyclic chunk slots with consumer backpressure (cyclic).
  enum class StagingMode {
    resident,
    cyclic
  };
  // Where a throughput-regime reduce delivers its result: multicast back into
  // every rank's staging replica (allReduce, where peers gather it), or unicast
  // straight to the destination buffer (reduceScatter, where nobody else needs it).
  enum class ReduceResult {
    multicast,
    unicast
  };
  // How the latency-regime reduce maps blocks to work: one unified sweep of the
  // full buffer, or block groups partitioned across remote peers.
  enum class LRMode {
    fullBuffer,
    partitioned
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
    size_t perStreamThreshold = 0
  >
  struct CollectiveConfig {
    static constexpr int PUT_BLOCKS = putBlocks;
    static constexpr int LOCAL_PUT_BLOCKS = localPutBlocks;
    static constexpr int GATHER_BLOCKS = gatherBlocks;
    static constexpr size_t CHUNK_SIZE = chunkSize;
    static constexpr size_t LATENCY_THRESHOLD = latencyThreshold;
    static constexpr CollectiveType COLLECTIVE_TYPE = ct;
    static constexpr StagingMode STAGING_MODE = stagingMode;
    // Per-stream protocol choice (a2aV): streams at or under the threshold move
    // as flag-carrying packets through the latency arena; larger streams stage
    // through fixed per-destination windows. Zero disables the fork entirely.
    static constexpr size_t PER_STREAM_THRESHOLD = perStreamThreshold;
    // The regime is the collective configuration's, not the Atom's: every
    // staged CollectiveConfig runs the throughput protocol.
    static constexpr Regime REGIME = Regime::throughput;
    static_assert(stagingMode == StagingMode::resident || ct == CollectiveType::chunked);
    static_assert(perStreamThreshold == 0 || ct == CollectiveType::chunked);
    // A packet carries eight payload bytes per sixteen; the per-source arena
    // region must hold the doubled footprint of a threshold-sized stream.
    static_assert(2 * perStreamThreshold <= PACKET_BUFFER_SIZE);
    static_assert(perStreamThreshold % 16 == 0);
  };
  // CollectiveConfigLR names the fused latency protocol.
  using CollectiveConfigLR = void;
  template<typename CollConfig>
  inline constexpr Regime regimeOf = CollConfig::REGIME;
  template<>
  inline constexpr Regime regimeOf<CollectiveConfigLR> = Regime::latency;

  // Buffer shapes relative to the communicator. A collective is a layout pair,
  // contribution -> destination: reduceScatter is scattered -> packed, allGather
  // packed -> scattered, all2all scattered -> transposed, and allReduce composes
  // the first two into scattered -> scattered. V variants carry variable splits.
  enum class DataLayout {
    packed, // one contiguous payload, no rank partitioning
    packedV, // packed, variable extent
    scattered, // partitioned by rank: slice r belongs to rank r
    scatteredV, // scattered, variable splits
    transposed, // partitioned by the transpose relation: my slice r <-> rank r's slice for me
    transposedV // transposed, variable splits
  };

  // The multimem datapath exists only where PTX maps the op: f16x2/bf16x2 carry
  // add and max, f32 carries add alone, and mul has no mapping. One-byte (fp8)
  // elements stay off deliberately: the switch accumulates fp8 in f16 at best,
  // while the unicast path reduces in f32 and in deterministic rank order.
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
    return 700; // base
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
  struct ST {
    template<typename Element>
    __device__ __forceinline__
    void operator()(Element* __restrict__ const& dst, const Element& v) const {
      purlin::store(dst, v);
    }
  };

  struct LRArgs {
    const cuda::std::byte* const src;
    cuda::std::byte** const staging;
    cuda::std::byte* const localStaging; // staging[rank]
    cuda::std::byte* const dst;
    // multicast alias of this rank's packet region (stagingPrefix + rank slot applied);
    // null when NVLS is unavailable
    cuda::std::byte* const mcStaging = nullptr;
    const uint64_t flag;
    const size_t bufferStride;
    const size_t stagingOffset = 0;
    const size_t bytes;
    const size_t maxBytes;
    const size_t* const inSizes = nullptr;
    const size_t* const sizes = nullptr;
    const size_t* const inOffsets = nullptr;
    const size_t* const offsets = nullptr;
    const int blocks;
    const int tIdx;
    const cuda::fast_mod_div<int, true> world;
    const int rank;
    const int bIdx;
    const int isInPlace;
  };

  struct ReduceTRArgs {
    cuda::std::byte** const sources;
    // multicast alias of the shard slice; valid iff the Atom's configuration selects
    // MemType::multimem
    cuda::std::byte* const mcSource = nullptr;
    cuda::std::byte* const dst;
    const size_t bytesRed;
    const cuda::fast_mod_div<int, true> world;
  };

  template<typename T>
  using ReduceAccumType = cuda::std::common_type_t<float, T>;

}

namespace purlin::fascia {
  template<
    typename Cfg,
    typename R2GOp,
    typename AlignedElement,
    typename Index_,
    int unrollFactor = Cfg::UNROLL_FACTOR,
    int threads = Cfg::THREADS,
    int AlignmentBytes = Cfg::GMEM_ACCESS_ALIGNMENT_BYTES
  >
  struct PeerOpConfig {
    static constexpr int THREADS = threads;
    static constexpr int UNROLL_FACTOR = unrollFactor;
    static constexpr int ALIGNMENT_BYTES = AlignmentBytes;
    static constexpr int VECTOR_WIDTH = ALIGNMENT_BYTES / sizeof(AlignedElement);
    using Operation = R2GOp;
    using Element = AlignedElement;
    using IndexType = Index_;
  };

  template<typename Config>
  __device__ __forceinline__
  void copyOp(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes,
    const uint32_t tIdx = threadIdx.x) {
    using VT = AlignedArray<typename Config::Element, Config::VECTOR_WIDTH>;
    using IndexT = Config::IndexType;
    const auto vP = static_cast<IndexT>(bytes / Config::ALIGNMENT_BYTES);
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const auto threadElems = vP / Config::THREADS;
    const auto trips = threadElems / Config::UNROLL_FACTOR;
    constexpr typename Config::Operation op{};
    for (int i = 0; i < trips; ++i) {
      VT reginald[Config::UNROLL_FACTOR];
      IndexT indices[Config::UNROLL_FACTOR];
      cuda::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        indices[j] = (i * Config::UNROLL_FACTOR + j) * Config::THREADS + tIdx;
      });
      // gmem -> rmem
      cuda::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        reginald[j] = vS[indices[j]];
      });
      // rmem -> gmem operation
      cuda::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        op(vD + indices[j], reginald[j]);
      });
    }
    const auto cutoff = trips * Config::UNROLL_FACTOR * Config::THREADS;
    if (vP > cutoff) {
      const auto residue = vP - cutoff;
      vS += cutoff;
      vD += cutoff;
      for (int i = static_cast<int>(tIdx); i < residue; i += Config::THREADS) {
        const auto v = vS[i];
        op(vD + i, v);
      }
    }
  }

  template<typename Config, DataLayout inputLayout>
  __device__ __forceinline__
  void gather(const LRArgs& gArgs) {
    using VT = LRP::RT;
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(gArgs.src);
    auto* __restrict__ vD = reinterpret_cast<VT*>(gArgs.dst);
    const auto gridSize = Config::THREADS * gArgs.blocks;
    const auto elements = gArgs.bytes / sizeof(VT);
    const auto worldTrips = gArgs.world / Config::WORLD_UNROLL;
    const auto cutoff = worldTrips * Config::WORLD_UNROLL;
    if constexpr (inputLayout == DataLayout::packed || inputLayout == DataLayout::packedV) {
      for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
        const auto value = vS[idx];
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            ptrs[p] = gArgs.staging[peer];
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            if (peer != gArgs.rank) {
              auto* __restrict__ packets = reinterpret_cast<LRP*>(ptrs[p]);
              packets[idx].write(value, gArgs.flag);
            }
          });
        }
        if (gArgs.world > cutoff) {
          for (int peer = cutoff; peer < gArgs.world; ++peer) {
            if (peer != gArgs.rank) {
              auto* __restrict__ packets = reinterpret_cast<LRP*>(gArgs.staging[peer]);
              packets[idx].write(value, gArgs.flag);
            }
          }
        }
      }
    }
    else if constexpr (inputLayout == DataLayout::scatteredV) {
      for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          LRP packets[Config::WORLD_UNROLL];
          int peers[Config::WORLD_UNROLL];
          size_t offsets[Config::WORLD_UNROLL];
          size_t sizes[Config::WORLD_UNROLL];
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            peers[p] = peer;
            ptrs[p] = gArgs.staging[peer];
            offsets[p] = gArgs.inOffsets[peer] / sizeof(VT);
            sizes[p] = gArgs.inSizes[peer] / sizeof(VT);
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = peers[p];
            const auto offset = offsets[p] + idx;
            const auto value = idx < sizes[p] ? vS[offset] : 0;
            packets[p] = LRP{value, gArgs.flag};
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = peers[p];
            if (peer != gArgs.rank) {
              auto* __restrict__ stagingPackets = reinterpret_cast<LRP*>(ptrs[p]);
              stagingPackets[idx].write(packets[p].data, packets[p].flag);
            }
          });
        }
        if (gArgs.world > cutoff) {
          for (int peer = cutoff; peer < gArgs.world; ++peer) {
            if (peer != gArgs.rank) {
              const auto offset = (gArgs.inOffsets[peer] / sizeof(VT)) + idx;
              const auto value = idx < (gArgs.inSizes[peer] / sizeof(VT)) ? vS[offset] : 0;
              auto* __restrict__ packets = reinterpret_cast<LRP*>(gArgs.staging[peer]);
              packets[idx].write(value, gArgs.flag);
            }
          }
        }
      }
    }
    else {
      for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          LRP packets[Config::WORLD_UNROLL];
          int peers[Config::WORLD_UNROLL];
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            peers[p] = peer;
            ptrs[p] = gArgs.staging[peer];
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = peers[p];
            const auto offset = static_cast<size_t>(peer) * elements + idx;
            const auto value = vS[offset];
            packets[p] = LRP{value, gArgs.flag};
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = peers[p];
            if (peer != gArgs.rank) {
              auto* __restrict__ stagingPackets = reinterpret_cast<LRP*>(ptrs[p]);
              stagingPackets[idx].write(packets[p].data, packets[p].flag);
            }
          });
        }
        if (gArgs.world > cutoff) {
          for (int peer = cutoff; peer < gArgs.world; ++peer) {
            if (peer != gArgs.rank) {
              const auto offset = static_cast<size_t>(peer) * elements + idx;
              const auto value = vS[offset];
              auto* __restrict__ packets = reinterpret_cast<LRP*>(gArgs.staging[peer]);
              packets[idx].write(value, gArgs.flag);
            }
          }
        }
      }
    }

    // gather
    if constexpr (inputLayout == DataLayout::packedV || inputLayout == DataLayout::scatteredV) {
      const auto gatherElements = gArgs.maxBytes / sizeof(VT);
      for (int idx = gArgs.tIdx; idx < gatherElements; idx += gridSize) {
        for (int i = 0; i < gArgs.world; ++i) {
          const auto peer = (gArgs.rank + i) % gArgs.world;
          const auto peerElements = gArgs.sizes[peer] / sizeof(VT);
          if (idx < peerElements) {
            const auto peerOffset = gArgs.offsets[peer] / sizeof(VT);
            const auto offset = peerOffset + idx;
            if (peer == gArgs.rank) {
              if (!gArgs.isInPlace) {
                const auto sourceOffset = inputLayout == DataLayout::packedV ? idx :
                  (gArgs.inOffsets[peer] / sizeof(VT)) + idx;
                vD[offset] = vS[sourceOffset];
              }
            }
            else {
              const auto* __restrict__ packets = reinterpret_cast<const LRP*>(
                gArgs.localStaging + gArgs.bufferStride * peer);
              vD[offset] = packets[idx].read(gArgs.flag);
            }
          }
        }
      }
    }
    else {
      for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
        for (int i = 0; i < gArgs.world; ++i) {
          const auto peer = (gArgs.rank + i) % gArgs.world;
          const auto offset = elements * peer + idx;
          if (peer == gArgs.rank) {
            if (!gArgs.isInPlace) {
              const auto sourceOffset = inputLayout == DataLayout::packed ? idx : offset;
              vD[offset] = vS[sourceOffset];
            }
          }
          else {
            const auto* __restrict__ packets = reinterpret_cast<const LRP*>(
              gArgs.localStaging + gArgs.bufferStride * peer);
            vD[offset] = packets[idx].read(gArgs.flag);
          }
        }
      }
    }
  }

  template<typename Cfg, typename RedOp, typename Element>
  __device__ __forceinline__
  void reduce(const ReduceTRArgs& redArgs,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytesRed, const size_t& residualOffset = 0) {
    constexpr RedOp op{};
    using VE = cuda::std::conditional_t<
      (Cfg::GMEM_ACCESS_ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Cfg::GMEM_ACCESS_ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<ReduceAccumType<Element>>::type,
    ReduceAccumType<Element>>;
    using VERaw = DataToRawType<VE>::type;
    constexpr int vectorWidth = Cfg::GMEM_ACCESS_ALIGNMENT_BYTES / sizeof(VE);
    using AVT = AlignedArray<AccumType, vectorWidth>;
    using LVT = AlignedArray<VERaw, vectorWidth>;
    static_assert(cuda::std::is_trivially_copyable_v<LVT>);
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VERaw, AccumType> storeConv{};
    auto* __restrict__ vD = reinterpret_cast<LVT*>(dst);
    const auto redElems = bytesRed / Cfg::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto threadElems = redElems / Cfg::THREADS;
    const auto trips = threadElems / Cfg::UNROLL_FACTOR;
    const auto worldTrips = redArgs.world / Cfg::WORLD_UNROLL;
    AVT accumulators[Cfg::UNROLL_FACTOR];
    constexpr typename RedOp::template Identity<AccumType> clear{};
    const auto cutoff = worldTrips * Cfg::WORLD_UNROLL;
    cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
      cuda::static_for<vectorWidth>([&](auto k) {
        clear(accumulators[j][k]);
      });
    });
    for (int i = 0; i < trips; ++i) {
      uint indices[Cfg::UNROLL_FACTOR];
      cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
        indices[j] = (i * Cfg::UNROLL_FACTOR + j) * Cfg::THREADS + threadIdx.x;
      });
      // reduce
      cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
        // below loop guarantees a deterministic reduction order: 0->1->...->world-1
        for (int t = 0; t < worldTrips; ++t) {
          LVT wendell[Cfg::WORLD_UNROLL];
          AVT arnold[Cfg::WORLD_UNROLL];
          cuda::static_for<Cfg::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Cfg::WORLD_UNROLL + p;
            auto* __restrict__ vData = reinterpret_cast<const LVT*>(redArgs.sources[peer] + residualOffset);
            // gmem -> rmem
            wendell[p] = vData[indices[j]];
          });
          cuda::static_for<Cfg::WORLD_UNROLL>([&](auto p) {
            AVT val{};
            const auto valRaw = wendell[p];
            cuda::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            arnold[p] = val;
          });
          cuda::static_for<Cfg::WORLD_UNROLL>([&](auto p) {
            op(accumulators[j], arnold[p]);
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = worldTrips * Cfg::WORLD_UNROLL; peer < redArgs.world; ++peer) {
            auto* __restrict__ vData = reinterpret_cast<const LVT*>(redArgs.sources[peer] + residualOffset);
            const auto valRaw = vData[indices[j]];
            AVT val{};
            cuda::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            op(accumulators[j], val);
          }
        }
      });
      // write results
      cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
        LVT resultRaw{};
        cuda::static_for<resultRaw.size()>([&](auto k) {
            resultRaw[k] = storeConv(accumulators[j][k]);
        });
        vD[indices[j]] = resultRaw;
        cuda::static_for<resultRaw.size()>([&](auto k) {
            clear(accumulators[j][k]);
        });
      });
    }
    const auto redCutoff = static_cast<size_t>(trips) * Cfg::UNROLL_FACTOR * Cfg::THREADS;
    if (redElems > redCutoff) {
      vD += redCutoff;
      const auto residue = redElems - redCutoff;
      AVT accumulator{};
      cuda::static_for<accumulator.size()>([&](auto j) {
        clear(accumulator[j]);
      });
      for (int idx = static_cast<int>(threadIdx.x); idx < residue; idx += Cfg::THREADS) {
        // do reduction
        for (int t = 0; t < worldTrips; ++t) {
          LVT wendell[Cfg::WORLD_UNROLL];
          AVT arnold[Cfg::WORLD_UNROLL];
          cuda::static_for<Cfg::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Cfg::WORLD_UNROLL + p;
            auto* __restrict__ vData = reinterpret_cast<const LVT*>(redArgs.sources[peer] + residualOffset) + redCutoff;
            // gmem -> rmem
            wendell[p] = vData[idx];
          });
          cuda::static_for<Cfg::WORLD_UNROLL>([&](auto p) {
            AVT val{};
            const auto valRaw = wendell[p];
            cuda::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            arnold[p] = val;
          });
          cuda::static_for<Cfg::WORLD_UNROLL>([&](auto p) {
            op(accumulator, arnold[p]);
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = worldTrips * Cfg::WORLD_UNROLL; peer < redArgs.world; ++peer) {
            auto* __restrict__ vData = reinterpret_cast<const LVT*>(redArgs.sources[peer] + residualOffset) + redCutoff;
            const auto valRaw = vData[idx];
            AVT val{};
            cuda::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            op(accumulator, val);
          }
        }
        // write results
        LVT resultRaw{};
        cuda::static_for<resultRaw.size()>([&](auto k) {
            resultRaw[k] = storeConv(accumulator[k]);
        });
        vD[idx] = resultRaw;
        cuda::static_for<resultRaw.size()>([&](auto k) {
          clear(accumulator[k]);
        });
      }
    }
  }

  template<typename Cfg, typename RedOp, typename Element>
  __device__ __forceinline__
  void reduce(const ReduceTRArgs& redArgs) {
    reduce<Cfg, RedOp, Element>(redArgs, redArgs.dst, redArgs.bytesRed);
  }

  template<typename Config, typename RedOp, typename Element, DataLayout iLayout>
  __device__ __forceinline__
  void reduceFullBuffer(const LRArgs& redArgs) {
    using VT = LRP::RT;
    constexpr RedOp op{};
    using VE = PackedElement<Element>::type; // promote to vector element
    using AccumType = PackedElement<ReduceAccumType<Element>>::type;
    using VERaw = DataToRawType<VE>::type;
    static_assert(alignof(VERaw) == alignof(VE) && sizeof(VERaw) == sizeof(VE));
    static_assert(sizeof(VT) % sizeof(VERaw) == 0 && alignof(VT) % alignof(VERaw) == 0);
    constexpr int vectorWidth = sizeof(VT) / sizeof(VERaw);
    using AVT = AlignedArray<AccumType, vectorWidth>;
    using LVT = AlignedArray<VERaw, vectorWidth>;
    static_assert(Config::ALIGNMENT_BYTES % alignof(VT) == 0 && Config::ALIGNMENT_BYTES % sizeof(VT) == 0);

    const auto* __restrict__ vS = reinterpret_cast<const VT*>(redArgs.src);
    auto* __restrict__ vD = reinterpret_cast<LVT*>(redArgs.dst);
    const auto gridSize = Config::THREADS * redArgs.blocks;
    const auto elements = redArgs.bytes / sizeof(VT);
    const auto worldTrips = redArgs.world / Config::WORLD_UNROLL;
    AVT accumulator{};
    static_assert(cuda::std::is_trivially_copyable_v<LVT>);
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VERaw, AccumType> storeConv{};
    constexpr typename RedOp::template Identity<AccumType> clear{};
    // Tiny messages need peer-level parallelism; larger messages retain the
    // element-striped LR schedule used by the other reduction layouts.
    const auto peerStriped = redArgs.world > 4 && redArgs.bytes <= 16UL * 1024UL;
    cuda::static_for<accumulator.size()>([&](auto i) {
      clear(accumulator[i]);
    });
    const auto cutoff = worldTrips * Config::WORLD_UNROLL;
    // put packets
    if constexpr (iLayout == DataLayout::scatteredV) {
      const auto putElems = redArgs.maxBytes / sizeof(VT);
      for (int idx = redArgs.tIdx; idx < putElems; idx += gridSize) {
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          LRP packets[Config::WORLD_UNROLL];
          int peers[Config::WORLD_UNROLL];
          size_t peerElems[Config::WORLD_UNROLL];
          size_t offsets[Config::WORLD_UNROLL];
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            peers[p] = peer;
            ptrs[p] = redArgs.staging[peer];
            peerElems[p] = redArgs.sizes[peer] / sizeof(VT);
            offsets[p] = redArgs.offsets[peer] / sizeof(VT);
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = peers[p];
            const auto offset = offsets[p] + idx;
            const auto value = idx < peerElems[p] ? vS[offset] : 0;
            packets[p] = LRP{value, redArgs.flag};
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = peers[p];
            if (peer != redArgs.rank) {
              auto* __restrict__ stagingPackets = reinterpret_cast<LRP*>(ptrs[p]);
              stagingPackets[idx].write(packets[p].data, packets[p].flag);
            }
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = cutoff; peer < redArgs.world; ++peer) {
            if (peer != redArgs.rank) {
              const auto offset = (redArgs.offsets[peer] / sizeof(VT)) + idx;
              const auto peerElem = redArgs.sizes[peer] / sizeof(VT);
              const auto value = idx < peerElem ? vS[offset] : 0;
              auto* __restrict__ packets = reinterpret_cast<LRP*>(redArgs.staging[peer]);
              packets[idx].write(value, redArgs.flag);
            }
          }
        }
      }
    }
    else if constexpr (iLayout == DataLayout::packed) {
      if constexpr (Config::MEMTYPE == MemType::multimem) {
        auto* __restrict__ mcPackets = reinterpret_cast<LRP*>(redArgs.mcStaging);
        for (size_t idx = redArgs.tIdx; idx < elements; idx += gridSize) {
          multimemStPacket(mcPackets + idx, vS[idx], redArgs.flag);
        }
      }
      else if (!peerStriped) {
        for (size_t idx = redArgs.tIdx; idx < elements; idx += gridSize) {
          const auto value = vS[idx];
          for (int t = 0; t < worldTrips; ++t) {
            cuda::std::byte* ptrs[Config::WORLD_UNROLL];
            int peers[Config::WORLD_UNROLL];
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              const auto peer = t * Config::WORLD_UNROLL + p;
              peers[p] = peer;
              ptrs[p] = redArgs.staging[peer] + redArgs.stagingOffset;
            });
            cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
              if (peers[p] != redArgs.rank) {
                auto* __restrict__ packets = reinterpret_cast<LRP*>(ptrs[p]);
                packets[idx].write(value, redArgs.flag);
              }
            });
          }
          if (redArgs.world > cutoff) {
            for (int peer = cutoff; peer < redArgs.world; ++peer) {
              if (peer != redArgs.rank) {
                auto* __restrict__ packets = reinterpret_cast<LRP*>(
                  redArgs.staging[peer] + redArgs.stagingOffset);
                packets[idx].write(value, redArgs.flag);
              }
            }
          }
        }
      }
      else {
        const auto laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
        const auto warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
        constexpr int warps = Config::THREADS / WARP_SIZE;
        for (int peerIdx = warpId; peerIdx < redArgs.world - 1; peerIdx += warps) {
          const auto peer = peerIdx < redArgs.rank ? peerIdx : peerIdx + 1;
          auto* __restrict__ packets = reinterpret_cast<LRP*>(
            redArgs.staging[peer] + redArgs.stagingOffset);
          for (size_t idx = laneId + static_cast<size_t>(redArgs.bIdx) * WARP_SIZE;
               idx < elements; idx += static_cast<size_t>(redArgs.blocks) * WARP_SIZE) {
            packets[idx].write(vS[idx], redArgs.flag);
          }
        }
      }
    }
    else {
      for (int idx = redArgs.tIdx; idx < elements; idx += gridSize) {
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          LRP packets[Config::WORLD_UNROLL];
          int peers[Config::WORLD_UNROLL];
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            peers[p] = peer;
            ptrs[p] = redArgs.staging[peer];
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = peers[p];
            const auto offset = static_cast<size_t>(peer) * elements + idx;
            const auto value = vS[offset];
            packets[p] = LRP{value, redArgs.flag};
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = peers[p];
            if (peer != redArgs.rank) {
              auto* __restrict__ stagingPackets = reinterpret_cast<LRP*>(ptrs[p]);
              stagingPackets[idx].write(packets[p].data, packets[p].flag);
            }
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = cutoff; peer < redArgs.world; ++peer) {
            if (peer != redArgs.rank) {
              const auto offset = static_cast<size_t>(peer) * elements + idx;
              const auto value = vS[offset];
              auto* __restrict__ packets = reinterpret_cast<LRP*>(redArgs.staging[peer]);
              packets[idx].write(value, redArgs.flag);
            }
          }
        }
      }
    }

    // reduce
    size_t firstElement = redArgs.tIdx;
    size_t elementStride = gridSize;
    if constexpr (iLayout == DataLayout::packed) {
      if (peerStriped) {
        constexpr int warps = Config::THREADS / WARP_SIZE;
        const auto laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
        const auto warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
        firstElement = laneId + static_cast<size_t>(redArgs.bIdx) * WARP_SIZE +
          static_cast<size_t>(warpId) * WARP_SIZE * redArgs.blocks;
        elementStride = static_cast<size_t>(warps) * WARP_SIZE * redArgs.blocks;
      }
    }
    for (size_t idx = firstElement; idx < elements; idx += elementStride) {
      const auto reducePeer = [&](const int peer) {
        LVT valRaw{};
        if (peer == redArgs.rank) {
          if constexpr (iLayout == DataLayout::packed) {
            valRaw = cuda::std::bit_cast<LVT>(vS[idx]);
          }
          else if constexpr (iLayout == DataLayout::scatteredV) {
            const auto peerElements = redArgs.sizes[peer] / sizeof(VT);
            const auto sourceOffset = redArgs.offsets[peer] / sizeof(VT);
            const auto value = idx < peerElements ? vS[sourceOffset + idx] : 0;
            valRaw = cuda::std::bit_cast<LVT>(value);
          }
          else {
            const auto sourceOffset = static_cast<size_t>(peer) * elements + idx;
            valRaw = cuda::std::bit_cast<LVT>(vS[sourceOffset]);
          }
        }
        else {
          const auto* __restrict__ packets = reinterpret_cast<const LRP*>(
            redArgs.localStaging + redArgs.bufferStride * peer);
          valRaw = cuda::std::bit_cast<LVT>(packets[idx].read(redArgs.flag));
        }
        AVT val{};
        cuda::static_for<val.size()>([&](auto i) {
          val[i] = loadConv(valRaw[i]);
        });
        op(accumulator, val);
      };
      constexpr int worldUnroll = Config::WORLD_UNROLL;
      #pragma unroll worldUnroll
      for (int peer = 0; peer < redArgs.world; ++peer) {
        reducePeer(peer);
      }
      // store accumulated result
      LVT resultRaw{};
      cuda::static_for<resultRaw.size()>([&](auto i) {
        resultRaw[i] = storeConv(accumulator[i]);
      });
      vD[idx] = resultRaw;
      cuda::static_for<resultRaw.size()>([&](auto i) {
        clear(accumulator[i]);
      });
    }
  }

  template<typename Config, typename RedOp, typename Element>
  __device__ __forceinline__
  void reducePartitioned(const LRArgs& redArgs) {
    using Payload = LRP::RT;
    using VE = cuda::std::conditional_t<
      (sizeof(Payload) > sizeof(Element)), typename PackedElement<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (sizeof(Payload) > sizeof(Element)), typename PackedElement<ReduceAccumType<Element>>::type,
      ReduceAccumType<Element>>;
    using VERaw = DataToRawType<VE>::type;
    constexpr int vectorWidth = sizeof(Payload) / sizeof(VERaw);
    using AVT = AlignedArray<AccumType, vectorWidth>;
    using LVT = AlignedArray<VERaw, vectorWidth>;
    static_assert(sizeof(LVT) == sizeof(Payload));

    constexpr size_t resultOffset = PACKET_BUFFER_SIZE / 2;
    const auto world = static_cast<int>(redArgs.world);
    const auto peers = world - 1;
    const auto packetsPerRank = redArgs.bytes / (static_cast<size_t>(world) * sizeof(Payload));
    const auto blocksPerPeer = redArgs.blocks / peers;
    const auto localBlock = redArgs.bIdx % blocksPerPeer;
    const auto peerIdx = redArgs.bIdx / blocksPerPeer;
    const auto remoteRank = peerIdx < redArgs.rank ? peerIdx : peerIdx + 1;
    const auto peerStride = static_cast<size_t>(Config::THREADS) * blocksPerPeer;
    const auto groupTid = static_cast<size_t>(threadIdx.x) +
      static_cast<size_t>(localBlock) * Config::THREADS;

    const auto* __restrict__ source = reinterpret_cast<const Payload*>(redArgs.src);
    auto* __restrict__ destination = reinterpret_cast<LVT*>(redArgs.dst);

    // Reduce-scatter: each block group sends the shard owned by its remote peer.
    auto* __restrict__ remoteInputPackets = reinterpret_cast<LRP*>(
      redArgs.staging[remoteRank] + redArgs.stagingOffset);
    const auto sourceOffset = static_cast<size_t>(remoteRank) * packetsPerRank;
    for (size_t idx = groupTid; idx < packetsPerRank; idx += peerStride) {
      remoteInputPackets[idx].write(source[sourceOffset + idx], redArgs.flag);
    }

    // Reduce the local shard in rank order, then publish it to every remote peer.
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VERaw, AccumType> storeConv{};
    constexpr RedOp op{};
    constexpr typename RedOp::template Identity<AccumType> clear{};
    const auto rankSourceOffset = static_cast<size_t>(redArgs.rank) * packetsPerRank;
    const auto gridTid = static_cast<size_t>(threadIdx.x) +
      static_cast<size_t>(redArgs.bIdx) * Config::THREADS;
    const auto gridStride = static_cast<size_t>(Config::THREADS) * redArgs.blocks;
    for (size_t idx = gridTid; idx < packetsPerRank; idx += gridStride) {
      AVT accumulator{};
      cuda::static_for<accumulator.size()>([&](auto i) {
        clear(accumulator[i]);
      });
      const auto reducePeer = [&](const int peer) {
        LVT valueRaw{};
        if (peer == redArgs.rank) {
          valueRaw = cuda::std::bit_cast<LVT>(source[rankSourceOffset + idx]);
        }
        else {
          const auto* __restrict__ inputPackets = reinterpret_cast<const LRP*>(
            redArgs.localStaging + static_cast<size_t>(peer) * redArgs.bufferStride);
          valueRaw = cuda::std::bit_cast<LVT>(inputPackets[idx].read(redArgs.flag));
        }
        AVT value{};
        cuda::static_for<value.size()>([&](auto i) {
          value[i] = loadConv(valueRaw[i]);
        });
        op(accumulator, value);
      };
      constexpr int worldUnroll = Config::WORLD_UNROLL;
      #pragma unroll worldUnroll
      for (int peer = 0; peer < world; ++peer) {
        reducePeer(peer);
      }

      LVT result{};
      cuda::static_for<result.size()>([&](auto i) {
        result[i] = storeConv(accumulator[i]);
      });
      destination[rankSourceOffset + idx] = result;

      const auto rawResult = cuda::std::bit_cast<Payload>(result);
      if constexpr (Config::MEMTYPE == MemType::multimem) {
        auto* __restrict__ mcResultPackets =
          reinterpret_cast<LRP*>(redArgs.mcStaging + resultOffset);
        multimemStPacket(mcResultPackets + idx, rawResult, redArgs.flag);
      }
      else {
        const auto publishPeer = [&](const int peer) {
          if (peer == redArgs.rank) return;
          auto* __restrict__ remoteResultPackets = reinterpret_cast<LRP*>(
            redArgs.staging[peer] + redArgs.stagingOffset + resultOffset);
          remoteResultPackets[idx].write(rawResult, redArgs.flag);
        };
        #pragma unroll worldUnroll
        for (int peer = 0; peer < world; ++peer) {
          publishPeer(peer);
        }
      }
    }

    // All-gather: the peer groups consume the same remote shard they sent above.
    const auto* __restrict__ resultPackets = reinterpret_cast<const LRP*>(
      redArgs.localStaging + static_cast<size_t>(remoteRank) * redArgs.bufferStride + resultOffset);
    const auto destinationOffset = static_cast<size_t>(remoteRank) * packetsPerRank;
    for (size_t idx = groupTid; idx < packetsPerRank; idx += peerStride) {
      destination[destinationOffset + idx] =
        cuda::std::bit_cast<LVT>(resultPackets[idx].read(redArgs.flag));
    }
  }

  template<typename Config, typename RedOp, typename Element, DataLayout inputLayout,
    LRMode mode = LRMode::fullBuffer>
  __device__ __forceinline__
  void reduce(const LRArgs& redArgs) {
    static_assert(mode == LRMode::fullBuffer || inputLayout == DataLayout::packed);
    if constexpr (mode == LRMode::partitioned) {
      reducePartitioned<Config, RedOp, Element>(redArgs);
    }
    else {
      reduceFullBuffer<Config, RedOp, Element, inputLayout>(redArgs);
    }
  }
}
#endif //PURLIN_BASE_CUH
