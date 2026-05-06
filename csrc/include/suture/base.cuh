//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_BASE_CUH
#define SUTURE_BASE_CUH
#include <cuda/utility>
#include <cutlass/array.h>

#include "constants.cuh"
#include "math.cuh"
#include "regime.cuh"

namespace suture {
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

  enum class StateSpace {
    GMEM,
    SMEM, // TODO: RMEM, TMEM
  };

  enum StageStatus: uint32_t {
    empty = 0U,
    full = 1U
  };

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
  struct DataToRawType<__half2> {
    using type = __half2_raw;
  };

  template<>
  struct DataToRawType<__nv_bfloat162> {
    using type = __nv_bfloat162_raw;
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

  template<typename Element>
  struct Element2 {
    using type = Element;
  };
  template<>
  struct Element2<float> {
    using type = float2;
  };
  template<>
  struct Element2<__half> {
    using type = __half2;
  };
  template<>
  struct Element2<__nv_bfloat16> {
    using type = __nv_bfloat162;
  };

  struct LRArgs {
    cuda::std::byte* const dst;
    const cuda::std::byte* const srcPut;
    const cuda::std::byte* const srcPutLocal;
    cuda::std::byte* const stagingPut; // remote, symmetric
    cuda::std::byte* const stagingPutLocal; // local, symmetric
    cuda::std::byte* const stagingGet; // local, symmetric
    const uint64_t flag;
    const size_t bytesPerPeer = 0;
    const size_t bytesPut;
    const size_t bytesRed;
    const int putBlock = 0;
    const cuda::fast_mod_div<int, true> world;
    const int isInPlace = 0;
    const int rank;
  };

  struct ReduceTRArgs {
    const cuda::std::byte* const* const sources;
    cuda::std::byte* const dst;
    const size_t bytesRed;
    const cuda::fast_mod_div<int, true> world;
  };

  template<typename T>
  using ReduceAccumType = cuda::std::common_type_t<float, T>;

  enum class TransferType {
    asynchronous,
    synchronous
  };
}

namespace suture::fascia {
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
  void putOp(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes,
    const uint32_t tIdx = threadIdx.x) {
    using VT = cutlass::AlignedArray<typename Config::Element, Config::VECTOR_WIDTH>;
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

  template<typename Cfg>
  __device__ __forceinline__
  void gather(const LRArgs& gArgs) {
    using VT = LRP16::RT;
    static_assert(Cfg::ALIGNMENT_BYTES == alignof(LRP16) && sizeof(LRP16) == Cfg::ALIGNMENT_BYTES);
    // 1. Put packets
    if (gArgs.putBlock) {
      auto* __restrict__ vD = reinterpret_cast<LRP16Raw*>(gArgs.stagingPut);
      auto* __restrict__ vDLocal = reinterpret_cast<LRP16Raw*>(gArgs.stagingPutLocal);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(gArgs.srcPut);
      const auto* __restrict__ vSLocal = reinterpret_cast<const VT*>(gArgs.srcPutLocal);
      const auto vP = gArgs.bytesPut / sizeof(LRP16::RT);
      const auto threadElems = vP / Cfg::THREADS;
      const auto trips = threadElems / Cfg::UNROLL_FACTOR;
      const auto cutoff = trips * Cfg::UNROLL_FACTOR * Cfg::THREADS;
      for (int i = 0; i < trips; ++i) {
        VT reginald[Cfg::UNROLL_FACTOR];
        uint indices[Cfg::UNROLL_FACTOR];
        // precompute indices
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          indices[j] = (i * Cfg::UNROLL_FACTOR + j) * Cfg::THREADS + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          reginald[j] = vS[indices[j]];
        });
        // rmem -> gmem, packets
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          LRP16 lrp{};
          lrp.pack(reginald[j], gArgs.flag);
          const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vD + indices[j])};
          packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
        });
        if (vDLocal) {
          VT ronald[Cfg::UNROLL_FACTOR];
          cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
            ronald[j] = vSLocal[indices[j]];
          });
          cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
            LRP16 lrp{};
            lrp.pack(ronald[j], gArgs.flag);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vDLocal + indices[j])};
            packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          });
        }
      }
      if (vP > cutoff) {
        const auto residue = vP - cutoff;
        vS += cutoff;
        vD += cutoff;
        if (vDLocal) {
          vSLocal += cutoff;
          vDLocal += cutoff;
        }
        for (int i = static_cast<int>(threadIdx.x); i < residue; i += Cfg::THREADS) {
          LRP16 lrp{};
          lrp.pack(vS[i], gArgs.flag);
          const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vD + i)};
          packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          if (vDLocal) {
            lrp.pack(vSLocal[i], gArgs.flag);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> localPacket{*(vDLocal + i)};
            localPacket.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          }
        }
      }
      return;
    }

    // 2. Do Gather
    {
      const auto vPRed = gArgs.bytesRed / sizeof(LRP16::RT);
      const auto threadElemsRed = vPRed / Cfg::THREADS;
      const auto tripsRed = threadElemsRed / Cfg::UNROLL_FACTOR;
      const auto cutoff = tripsRed * Cfg::UNROLL_FACTOR * Cfg::THREADS;
      const auto residueRed = vPRed - cutoff;
      auto* __restrict__ rvS = reinterpret_cast<LRP16Raw*>(gArgs.stagingGet);
      auto* __restrict__ rvD = reinterpret_cast<VT*>(gArgs.dst);
      const auto elementsPerPeer = gArgs.bytesPerPeer / sizeof(LRP16::RT);
      static_assert(cuda::std::is_trivially_copyable_v<VT>);
      constexpr auto packetsPerPeer = PACKET_BUFFER_SIZE / sizeof(LRP16Raw);
      for (int i = 0; i < tripsRed; ++i) {
        uint indices[Cfg::UNROLL_FACTOR];
        // precompute indices
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          indices[j] = (i * Cfg::UNROLL_FACTOR + j) * Cfg::THREADS + threadIdx.x;
        });

        // await packet
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          for (int k = gArgs.isInPlace ? 1 : 0; k < gArgs.world; ++k) {
            const auto peer = (gArgs.rank + k) % gArgs.world;
            VT val{};
            auto* __restrict__ dstP = rvD + peer * elementsPerPeer;
            auto* __restrict__ packetPtr = rvS + (packetsPerPeer * peer + indices[j]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*packetPtr};
            auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
            auto hPA = currentPacket.flag == gArgs.flag;
            while (!hPA) {
              currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
              hPA = currentPacket.flag == gArgs.flag;
            }
            currentPacket.unpack(val);
            // store result
            dstP[indices[j]] = val;
          }
        });
      }
      if (residueRed) {
        rvS += cutoff;
        rvD += cutoff;
        for (int i = static_cast<int>(threadIdx.x); i < residueRed; i += Cfg::THREADS) {
          for (int k = gArgs.isInPlace ? 1 : 0; k < gArgs.world; ++k) {
            const auto peer = (gArgs.rank + k) % gArgs.world;
            VT val{};
            auto* __restrict__ dstP = rvD + peer * elementsPerPeer;
            auto* __restrict__ packetPtr = rvS + (packetsPerPeer * peer + i);
            cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*packetPtr};
            auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
            auto hPA = currentPacket.flag == gArgs.flag; // hasPacketArrived
            while (!hPA) {
              currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
              hPA = currentPacket.flag == gArgs.flag;
            }
            currentPacket.unpack(val);
            // store result
            dstP[i] = val;
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
      (Cfg::GMEM_ACCESS_ALIGNMENT_BYTES > sizeof(Element)), typename Element2<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Cfg::GMEM_ACCESS_ALIGNMENT_BYTES > sizeof(Element)), typename Element2<ReduceAccumType<Element>>::type,
    ReduceAccumType<Element>>;
    using VERaw = DataToRawType<VE>::type;
    constexpr int vectorWidth = Cfg::GMEM_ACCESS_ALIGNMENT_BYTES / sizeof(VE);
    using AVT = cutlass::AlignedArray<AccumType, vectorWidth>;
    using LVT = cutlass::AlignedArray<VERaw, vectorWidth>;
    static_assert(cuda::std::is_trivially_copyable_v<LVT>);
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VE, AccumType> storeConv{};
    auto* __restrict__ vD = reinterpret_cast<LVT*>(dst);
    const auto redElems = bytesRed / Cfg::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto threadElems = redElems / Cfg::THREADS;
    const auto trips = threadElems / Cfg::UNROLL_FACTOR;
    const auto worldTrips = redArgs.world / Cfg::WORLD_UNROLL;
    AVT accumulators[Cfg::UNROLL_FACTOR];
    constexpr InplaceZero<AccumType> clear{};
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
        const auto cutoff = worldTrips * Cfg::WORLD_UNROLL;
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
        const auto cutoff = worldTrips * Cfg::WORLD_UNROLL;
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

  template<typename Cfg, typename RedOp, typename Element>
  __device__ __forceinline__
  void reduce(const LRArgs& redArgs) {
    using VT = LRP16::RT;
    static_assert(Cfg::ALIGNMENT_BYTES == alignof(LRP16) && sizeof(LRP16) == Cfg::ALIGNMENT_BYTES);
    // 1. Put packets
    if (redArgs.putBlock) {
      auto* __restrict__ vD = reinterpret_cast<LRP16Raw*>(redArgs.stagingPut);
      auto* __restrict__ vDLocal = reinterpret_cast<LRP16Raw*>(redArgs.stagingPutLocal);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(redArgs.srcPut);
      const auto* __restrict__ vSLocal = reinterpret_cast<const VT*>(redArgs.srcPutLocal);
      const auto vP = redArgs.bytesPut / sizeof(LRP16::RT);
      const auto threadElems = vP / Cfg::THREADS;
      const auto trips = threadElems / Cfg::UNROLL_FACTOR;
      const auto cutoff = trips * Cfg::UNROLL_FACTOR * Cfg::THREADS;
      for (int i = 0; i < trips; ++i) {
        VT reginald[Cfg::UNROLL_FACTOR];
        uint indices[Cfg::UNROLL_FACTOR];
        // precompute indices
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          indices[j] = (i * Cfg::UNROLL_FACTOR + j) * Cfg::THREADS + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          reginald[j] = vS[indices[j]];
        });
        // rmem -> gmem, packets
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          LRP16 lrp{};
          lrp.pack(reginald[j], redArgs.flag);
          const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vD + indices[j])};
          packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
        });
        if (vDLocal) {
          VT localReginald[Cfg::UNROLL_FACTOR];
          cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
            localReginald[j] = vSLocal[indices[j]];
          });
          cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
            LRP16 lrp{};
            lrp.pack(localReginald[j], redArgs.flag);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vDLocal + indices[j])};
            packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          });
        }
      }
      if (vP > cutoff) {
        const auto residue = vP - cutoff;
        vS += cutoff;
        vD += cutoff;
        if (vDLocal) {
          vSLocal += cutoff;
          vDLocal += cutoff;
        }
        for (int i = static_cast<int>(threadIdx.x); i < residue; i += Cfg::THREADS) {
          LRP16 lrp{};
          lrp.pack(vS[i], redArgs.flag);
          const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vD + i)};
          packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          if (vDLocal) {
            lrp.pack(vSLocal[i], redArgs.flag);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> localPacket{*(vDLocal + i)};
            localPacket.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
          }
        }
      }
      return;
    }

    // 2. Do Reduction
    {
      const auto vPRed = redArgs.bytesRed / sizeof(LRP16::RT);
      const auto threadElemsRed = vPRed / Cfg::THREADS;
      const auto tripsRed = threadElemsRed / Cfg::UNROLL_FACTOR;
      const auto cutoff = tripsRed * Cfg::UNROLL_FACTOR * Cfg::THREADS;
      const auto residueRed = vPRed - cutoff;
      constexpr RedOp op{};
      using VE = Element2<Element>::type; // promote to vector element
      using AccumType = Element2<ReduceAccumType<Element>>::type;
      using VERaw = DataToRawType<VE>::type;
      static_assert(alignof(VERaw) == alignof(VE) && sizeof(VERaw) == sizeof(VE));
      static_assert(sizeof(VT) % sizeof(VERaw) == 0 && alignof(VT) % alignof(VERaw) == 0);
      constexpr int vectorWidth = sizeof(VT) / sizeof(VERaw);
      using AVT = cutlass::AlignedArray<AccumType, vectorWidth>;
      using LVT = cutlass::AlignedArray<VERaw, vectorWidth>;
      auto* __restrict__ rvS = reinterpret_cast<LRP16Raw*>(redArgs.stagingGet);
      auto* __restrict__ rvD = reinterpret_cast<LVT*>(redArgs.dst);
      static_assert(cuda::std::is_trivially_copyable_v<LVT>);
      constexpr Converter<AccumType, VE> loadConv{};
      constexpr Converter<VE, AccumType> storeConv{};
      AVT accumulators[Cfg::UNROLL_FACTOR];
      constexpr InplaceZero<AccumType> clear{};
      cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
        cuda::static_for<vectorWidth>([&](auto k) {
          clear(accumulators[j][k]);
        });
      });
      constexpr auto packetsPerPeer = PACKET_BUFFER_SIZE / sizeof(LRP16Raw);
      for (int i = 0; i < tripsRed; ++i) {
        uint indices[Cfg::UNROLL_FACTOR];
        // precompute indices
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          indices[j] = (i * Cfg::UNROLL_FACTOR + j) * Cfg::THREADS + threadIdx.x;
        });

        // await packet
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          for (int peer = 0; peer < redArgs.world; ++peer) {
            LVT valRaw{};
            auto* __restrict__ packetPtr = rvS + (packetsPerPeer * peer + indices[j]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*packetPtr};
            auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
            auto hPA = currentPacket.flag == redArgs.flag;
            while (!hPA) {
              currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
              hPA = currentPacket.flag == redArgs.flag;
            }
            currentPacket.unpack(valRaw);
            AVT val{};
            cuda::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            op(accumulators[j], val);
          }
          // store accumulated result
          LVT resultRaw{};
          cuda::static_for<resultRaw.size()>([&](auto k) {
              resultRaw[k] = storeConv(accumulators[j][k]); // ideally no MOV instructions would be generated here
          });
          rvD[indices[j]] = resultRaw;
          cuda::static_for<resultRaw.size()>([&](auto k) {
            clear(accumulators[j][k]);
          });
        });
      }
      if (residueRed) {
        rvS += cutoff;
        rvD += cutoff;
        AVT accumulator{};
        cuda::static_for<accumulator.size()>([&](auto j) {
          clear(accumulator[j]);
        });
        for (int i = static_cast<int>(threadIdx.x); i < residueRed; i += Cfg::THREADS) {
          for (int peer = 0; peer < redArgs.world; ++peer) {
            LVT valRaw{};
            auto* __restrict__ packetPtr = rvS + (packetsPerPeer * peer + i);
            cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*packetPtr};
            auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
            auto hPA = currentPacket.flag == redArgs.flag; // hasPacketArrived
            while (!hPA) {
              currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
              hPA = currentPacket.flag == redArgs.flag;
            }
            currentPacket.unpack(valRaw);
            AVT val{};
            cuda::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            op(accumulator, val);
          }
          // store accumulated result
          LVT resultRaw{};
          cuda::static_for<resultRaw.size()>([&](auto k) {
              resultRaw[k] = storeConv(accumulator[k]); // ideally no MOV instructions would be generated here
          });
          rvD[i] = resultRaw;
          cuda::static_for<resultRaw.size()>([&](auto k) {
            clear(accumulator[k]);
          });
        }
      }
    }
  }
}
#endif //SUTURE_BASE_CUH
