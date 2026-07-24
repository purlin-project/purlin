//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_BASE_CUH
#define PURLIN_BASE_CUH
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/ptx>
#include <cuda/utility>

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
  template<
    CollectiveType ct,
    int putBlocks,
    int gatherBlocks,
    size_t chunkSize,
    int localPutBlocks = 8,
    size_t latencyThreshold = 0
  >
  struct CollectiveConfig {
    static constexpr int PUT_BLOCKS = putBlocks;
    static constexpr int LOCAL_PUT_BLOCKS = localPutBlocks;
    static constexpr int GATHER_BLOCKS = gatherBlocks;
    static constexpr size_t CHUNK_SIZE = chunkSize;
    static constexpr size_t LATENCY_THRESHOLD = latencyThreshold;
    static constexpr CollectiveType COLLECTIVE_TYPE = ct;
  };
  using CollectiveConfigLR = void;

  enum class DataLayout {
    packed, // allReduce
    packedV, // allGatherV
    scattered, // reduceScatter
    scatteredV, // reduceScatter_v, all2allV
    transposed, // all2all
    transposedV // all2allV
  };

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
    cuda::std::byte** const redStaging;
    cuda::std::byte* const localStaging; // staging[rank]
    cuda::std::byte* const redDst;
    cuda::std::byte* const dst;
    const uint64_t flag;
    const size_t bufferStride;
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
    using VT = LRP16::RT;
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(gArgs.src);
    auto* __restrict__ vD = reinterpret_cast<VT*>(gArgs.dst);
    const auto gridSize = Config::THREADS * gArgs.blocks;
    const auto elements = gArgs.bytes / sizeof(VT);
    const auto worldTrips = gArgs.world / Config::WORLD_UNROLL;
    const auto cutoff = worldTrips * Config::WORLD_UNROLL;
    if constexpr (inputLayout == DataLayout::packed || inputLayout == DataLayout::packedV) {
      for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
        const auto value = vS[idx];
        LRP16 lrp{};
        lrp.pack(value, gArgs.flag);
        const auto castPacket = cuda::std::bit_cast<LRP16Raw>(lrp);
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            ptrs[p] = gArgs.staging[peer];
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(ptrs[p]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(castPacket, cuda::memory_order_relaxed);
          });
        }
        if (gArgs.world > cutoff) {
          for (int peer = cutoff; peer < gArgs.world; ++peer) {
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(gArgs.staging[peer]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(castPacket, cuda::memory_order_relaxed);
          }
        }
      }
    }
    else if constexpr (inputLayout == DataLayout::scatteredV) {
      for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          LRP16Raw larry[Config::WORLD_UNROLL];
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
            LRP16 lrp{};
            lrp.pack(value, gArgs.flag);
            larry[p] = cuda::std::bit_cast<LRP16Raw>(lrp);
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(ptrs[p]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(larry[p], cuda::memory_order_relaxed);
          });
        }
        if (gArgs.world > cutoff) {
          for (int peer = cutoff; peer < gArgs.world; ++peer) {
            const auto offset = (gArgs.inOffsets[peer] / sizeof(VT)) + idx;
            const auto value = idx < (gArgs.inSizes[peer] / sizeof(VT)) ? vS[offset] : 0;
            LRP16 lrp{};
            lrp.pack(value, gArgs.flag);
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(gArgs.staging[peer]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          }
        }
      }
    }
    else {
      for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          LRP16Raw larry[Config::WORLD_UNROLL];
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
            LRP16 lrp{};
            lrp.pack(value, gArgs.flag);
            larry[p] = cuda::std::bit_cast<LRP16Raw>(lrp);
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(ptrs[p]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(larry[p], cuda::memory_order_relaxed);
          });
        }
        if (gArgs.world > cutoff) {
          for (int peer = cutoff; peer < gArgs.world; ++peer) {
            const auto offset = static_cast<size_t>(peer) * elements + idx;
            const auto value = vS[offset];
            LRP16 lrp{};
            lrp.pack(value, gArgs.flag);
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(gArgs.staging[peer]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          }
        }
      }
    }

    // gather
    if constexpr (inputLayout == DataLayout::packedV || inputLayout == DataLayout::scatteredV) {
      const auto gatherElements = gArgs.maxBytes / sizeof(VT);
      for (int idx = gArgs.tIdx; idx < gatherElements; idx += gridSize) {
        for (int i = gArgs.isInPlace ? 1 : 0; i < gArgs.world; ++i) {
          const auto peer = (gArgs.rank + i) % gArgs.world;
          const auto peerElements = gArgs.sizes[peer] / sizeof(VT);
          if (idx < peerElements) {
            const auto peerOffset = gArgs.offsets[peer] / sizeof(VT);
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(gArgs.localStaging + gArgs.bufferStride * peer);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
            auto hPA = currentPacket.flag == gArgs.flag;
            while (!hPA) {
              currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
              hPA = currentPacket.flag == gArgs.flag;
            }
            const auto offset = peerOffset + idx;
            vD[offset] = currentPacket.data;
          }
        }
      }
    }
    else {
      for (int idx = gArgs.tIdx; idx < elements; idx += gridSize) {
        for (int i = gArgs.isInPlace ? 1 : 0; i < gArgs.world; ++i) {
          const auto peer = (gArgs.rank + i) % gArgs.world;
          auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(gArgs.localStaging + gArgs.bufferStride * peer);
          const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
          auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
          auto hPA = currentPacket.flag == gArgs.flag;
          while (!hPA) {
            currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
            hPA = currentPacket.flag == gArgs.flag;
          }
          const auto offset = elements * peer + idx;
          vD[offset] = currentPacket.data;
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
    constexpr InplaceZero<AccumType> clear{};
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
  void reduce(const LRArgs& redArgs) {
    using VT = LRP16::RT;
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
    constexpr InplaceZero<AccumType> clear{};
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
          LRP16Raw larry[Config::WORLD_UNROLL];
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
            LRP16 lrp{};
            lrp.pack(value, redArgs.flag);
            larry[p] = cuda::std::bit_cast<LRP16Raw>(lrp);
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(ptrs[p]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(larry[p], cuda::memory_order_relaxed);
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = cutoff; peer < redArgs.world; ++peer) {
            const auto offset = (redArgs.offsets[peer] / sizeof(VT)) + idx;
            const auto peerElem = redArgs.sizes[peer] / sizeof(VT);
            const auto value = idx < peerElem ? vS[offset] : 0;
            LRP16 lrp{};
            lrp.pack(value, redArgs.flag);
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(redArgs.staging[peer]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          }
        }
      }
    }
    else if constexpr (iLayout == DataLayout::packed) {
      for (int idx = redArgs.tIdx; idx < elements; idx += gridSize) {
        const auto value = vS[idx];
        LRP16 lrp{};
        lrp.pack(value, redArgs.flag);
        const auto castPacket = cuda::std::bit_cast<LRP16Raw>(lrp);
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            ptrs[p] = redArgs.staging[peer];
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(ptrs[p]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(castPacket, cuda::memory_order_relaxed);
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = cutoff; peer < redArgs.world; ++peer) {
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(redArgs.staging[peer]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(castPacket, cuda::memory_order_relaxed);
          }
        }
      }
    }
    else {
      for (int idx = redArgs.tIdx; idx < elements; idx += gridSize) {
        for (int t = 0; t < worldTrips; ++t) {
          cuda::std::byte* ptrs[Config::WORLD_UNROLL];
          LRP16Raw larry[Config::WORLD_UNROLL];
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
            LRP16 lrp{};
            lrp.pack(value, redArgs.flag);
            larry[p] = cuda::std::bit_cast<LRP16Raw>(lrp);
          });
          cuda::static_for<Config::WORLD_UNROLL>([&](auto p) {
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(ptrs[p]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(larry[p], cuda::memory_order_relaxed);
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = cutoff; peer < redArgs.world; ++peer) {
            const auto offset = static_cast<size_t>(peer) * elements + idx;
            const auto value = vS[offset];
            LRP16 lrp{};
            lrp.pack(value, redArgs.flag);
            auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(redArgs.staging[peer]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
            packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          }
        }
      }
    }
    // reduce
    for (int idx = redArgs.tIdx; idx < elements; idx += gridSize) {
      for (int peer = 0; peer < redArgs.world; ++peer) {
        LVT valRaw{};
        auto* __restrict__ vStaging = reinterpret_cast<LRP16Raw*>(redArgs.localStaging + redArgs.bufferStride * peer);
        const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vStaging + idx)};
        auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
        auto hPA = currentPacket.flag == redArgs.flag;
        while (!hPA) {
          currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
          hPA = currentPacket.flag == redArgs.flag;
        }
        currentPacket.unpack(valRaw);
        AVT val{};
        cuda::static_for<val.size()>([&](auto i) {
          val[i] = loadConv(valRaw[i]);
        });
        op(accumulator, val);
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
}
#endif //PURLIN_BASE_CUH
