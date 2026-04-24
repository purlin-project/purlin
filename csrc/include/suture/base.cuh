//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_BASE_CUH
#define SUTURE_BASE_CUH
#include <cuda/utility>
#include <cutlass/array.h>

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

  struct ReduceLRArgs {
    cuda::std::byte* const dst;
    const cuda::std::byte* const srcPut;
    const cuda::std::byte* const srcRed;
    cuda::std::byte* const stagingPut; // remote, symmetric
    cuda::std::byte* const stagingRed; // local, symmetric
    uint8_t* const flagsPut; // non-symmetric
    uint8_t* const flagsRed;
    const size_t bytesPut;
    const size_t bytesRed;
    const int rank;
    const cuda::fast_mod_div<int> world;
  };

  struct ReduceTRArgs {
    uint32_t* const signals; // [world]
    uint32_t* const putSignals;
    cuda::std::byte* const dst;
    cuda::std::byte* const srcPut;
    cuda::std::byte* const redPut;
    cuda::std::byte* const srcRed;
    cuda::std::byte* const src;
    uint64_t* const arrivals;
    uint* const sigCounter;
    const size_t totalBytes;
    const size_t bytesPut;
    const size_t bytesRed;
    const int rank;
    const cuda::fast_mod_div<int> world;
    const cuda::fast_mod_div<int> actualWorld; // world - 1
    const int numBlocks = static_cast<int>(gridDim.x);
    const int bIdx = static_cast<int>(blockIdx.x);
    const uint senseBit;
    const uint syncRemoteOffset;
    const uint syncLocalOffset;
    const int superBlockSize;
    const int putBlock = 0;
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
      // gmem -> rmem
      cuda::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        indices[j] = (i * Config::UNROLL_FACTOR + j) * Config::THREADS + tIdx;
        reginald[j] = vS[indices[j]];
      });
      // rmem -> gmem operation
      cuda::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        op(vD + indices[j], reginald[j]);
      });
    }
    const auto residue = vP - trips * Config::UNROLL_FACTOR * Config::THREADS;
    vS += (trips * Config::UNROLL_FACTOR * Config::THREADS);
    vD += (trips * Config::UNROLL_FACTOR * Config::THREADS);
    for (int i = static_cast<int>(tIdx); i < residue; i += Config::THREADS) {
      const auto v = vS[i];
      op(vD + i, v);
    }
  }

  template<typename Cfg, typename RedOp, typename Element>
  __device__ __forceinline__
  void reduce(const ReduceLRArgs& redArgs) {
    using VT = LRP16::RT;
    static_assert(Cfg::ALIGNMENT_BYTES == alignof(LRP16) && sizeof(LRP16) == Cfg::ALIGNMENT_BYTES);
    // 1. Put packets
    {
      auto* __restrict__ vD = reinterpret_cast<LRP16Raw*>(redArgs.stagingPut);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(redArgs.srcPut);
      const auto vP = redArgs.bytesPut / Cfg::ALIGNMENT_BYTES;
      const auto threadElems = vP / Cfg::THREADS;
      const auto trips = threadElems / Cfg::UNROLL_FACTOR;
      const auto residue = vP - trips * Cfg::UNROLL_FACTOR * Cfg::THREADS;
      for (int i = 0; i < trips; ++i) {
        VT reginald[Cfg::UNROLL_FACTOR];
        uint indices[Cfg::UNROLL_FACTOR];
        uint32_t cachedFlags[Cfg::UNROLL_FACTOR];
        // precompute indices
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          indices[j] = (i * Cfg::UNROLL_FACTOR + j) * Cfg::THREADS + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          const auto cf = static_cast<uint32_t>(redArgs.flagsPut[indices[j]]);
          cachedFlags[j] = cf == 0U ? 1U : 0U;
          // flip flag for subsequent use
          redArgs.flagsPut[indices[j]] = static_cast<uint8_t>(cachedFlags[j]);
          reginald[j] = vS[indices[j]];
        });
        // rmem -> gmem, packets
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          LRP16 lrp{};
          lrp.pack(reginald[j], cachedFlags[j]);
          const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vD + indices[j])};
          packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
        });
      }
      if (residue) {
        vS += (trips * Cfg::UNROLL_FACTOR * Cfg::THREADS);
        vD += (trips * Cfg::UNROLL_FACTOR * Cfg::THREADS);
        auto* __restrict__ flagsRes = redArgs.flagsPut + (trips * Cfg::UNROLL_FACTOR * Cfg::THREADS);
        for (int i = static_cast<int>(threadIdx.x); i < residue; i += Cfg::THREADS) {
          LRP16 lrp{};
          const auto f = static_cast<uint32_t>(flagsRes[i]);
          const auto cf = f == 0U ? 1U : 0U;
          flagsRes[i] = static_cast<uint8_t>(cf);
          lrp.pack(vS[i], cf);
          const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*(vD + i)};
          packet.store(cuda::std::bit_cast<LRP16Raw>(lrp), cuda::memory_order_relaxed);
        }
      }
    }

    // 2. Do Reduction
    {
      const auto vPRed = redArgs.bytesRed / Cfg::ALIGNMENT_BYTES;
      const auto threadElemsRed = vPRed / Cfg::THREADS;
      const auto tripsRed = threadElemsRed / Cfg::UNROLL_FACTOR;
      const auto residueRed = vPRed - tripsRed * Cfg::UNROLL_FACTOR * Cfg::THREADS;
      constexpr RedOp op{};
      using AccumType = float2;
      using VE = Element2<Element>::type; // promote to vector element
      using VERaw = RawToDataType<VE>::type;
      static_assert(alignof(VERaw) == alignof(VE) && sizeof(VERaw) == sizeof(VE));
      static_assert(sizeof(VT) % sizeof(VERaw) == 0 && alignof(VT) % alignof(VERaw) == 0);
      constexpr int vectorWidth = sizeof(VT) / sizeof(VERaw);
      using AVT = cutlass::AlignedArray<AccumType, vectorWidth>;
      using LVT = cutlass::AlignedArray<VERaw, vectorWidth>;
      auto* __restrict__ rvS = reinterpret_cast<LRP16Raw*>(redArgs.stagingRed);
      auto* __restrict__ rsR = reinterpret_cast<LVT*>(redArgs.srcRed);
      auto* __restrict__ rvD = reinterpret_cast<LVT*>(redArgs.dst);
      static_assert(cuda::std::is_trivially_copyable_v<LVT>);
      constexpr Converter<AccumType, VE> loadConv{};
      constexpr Converter<VE, AccumType> storeConv{};
      AVT accum{};
      for (int i = 0; i < tripsRed; ++i) {
        uint indices[Cfg::UNROLL_FACTOR];
        LVT reginald[Cfg::UNROLL_FACTOR];
        // precompute indices
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          indices[j] = (i * Cfg::UNROLL_FACTOR + j) * Cfg::THREADS + threadIdx.x;
        });

        // gmem -> rmem
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          // read values from src
          reginald[j] = rsR[indices[j]];
        });

        // await packet
        cuda::static_for<Cfg::UNROLL_FACTOR>([&](auto j) {
          auto reggie = reginald[j];
          cuda::static_for<accum.size()>([&](auto k) {
            accum[k] = loadConv(reggie[k]); // convert to accumulator type
          });
          for (int p = 1; p < redArgs.world; ++p) {
            const auto peer = (p + redArgs.rank) % redArgs.world;
            auto* __restrict__ currentFlags = redArgs.flagsRed + (peer * FLAG_BUFFER_SIZE);
            const auto f = static_cast<uint32_t>(currentFlags[indices[j]]);
            const auto expectedFlag = f == 0U ? 1U : 0U;
            // flip flag for subsequent use
            currentFlags[indices[j]] = static_cast<uint8_t>(expectedFlag);

            auto* __restrict__ packetPtr = rvS + (PACKET_BUFFER_SIZE * peer + indices[j]);
            const cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*packetPtr};
            auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
            auto hPA = currentPacket.flag == expectedFlag;
            while (!hPA) {
              currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
              hPA = currentPacket.flag == expectedFlag;
            }
            LVT valRaw{};
            currentPacket.unpack(valRaw);
            AVT val{};
            cuda::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]); // ideally no MOV instructions would be generated here
            });
            // do reduction
            op(accum, val);
          }
          // store accumulated result
          LVT resultRaw{};
          cuda::static_for<resultRaw.size()>([&](auto k) {
              resultRaw[k] = storeConv(accum[k]); // ideally no MOV instructions would be generated here
          });
          rvD[indices[j]] = resultRaw;
        });
      }
      if (residueRed) {
        rvS += (tripsRed * Cfg::UNROLL_FACTOR * Cfg::THREADS);
        rsR += (tripsRed * Cfg::UNROLL_FACTOR * Cfg::THREADS);
        rvD += (tripsRed * Cfg::UNROLL_FACTOR * Cfg::THREADS);
        auto* __restrict__ fRR = redArgs.flagsRed + (tripsRed * Cfg::UNROLL_FACTOR * Cfg::THREADS);
        for (int i = static_cast<int>(threadIdx.x); i < residueRed; i += Cfg::THREADS) {
          auto reggie = rsR[i];
          cuda::static_for<accum.size()>([&](auto k) {
            accum[k] = loadConv(reggie[k]); // convert to accumulator type
          });
          for (int p = 1; p < redArgs.world; ++p) {
            const auto peer = (p + redArgs.rank) % redArgs.world;
            auto* __restrict__ currentFlags = fRR + (peer * FLAG_BUFFER_SIZE);
            const auto f = static_cast<uint32_t>(currentFlags[i]);
            const auto expectedFlag = f == 0U ? 1U : 0U;
            currentFlags[i] = static_cast<uint8_t>(expectedFlag);

            auto* __restrict__ packetPtr = rvS + (PACKET_BUFFER_SIZE * peer + i);
            cuda::atomic_ref<LRP16Raw, cuda::thread_scope_system> packet{*packetPtr};
            auto currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
            auto hPA = currentPacket.flag == expectedFlag; // hasPacketArrived
            while (!hPA) {
              currentPacket = cuda::std::bit_cast<LRP16>(packet.load(cuda::memory_order_relaxed));
              hPA = currentPacket.flag == expectedFlag;
            }
            LVT valRaw{};
            currentPacket.unpack(valRaw);
            AVT val{};
            cuda::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]); // ideally no MOV instructions would be generated here
            });
            // do reduction
            op(accum, val);
          }
          // store accumulated result
          LVT resultRaw{};
          cuda::static_for<resultRaw.size()>([&](auto k) {
              resultRaw[k] = storeConv(accum[k]); // ideally no MOV instructions would be generated here
          });
          rvD[i] = resultRaw;
        }
      }
    }
  }
}
#endif //SUTURE_BASE_CUH