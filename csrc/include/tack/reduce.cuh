//
// Created by Osayamen on 3/27/26.
//

#ifndef TACK_REDUCE_CUH
#define TACK_REDUCE_CUH
#include <cuda/atomic>
#include <cuda/utility>
#include <cutlass/array.h>

#include "copy.cuh"
#include "rvt.cuh"

namespace tack {
  enum class RedDataType {
    fp16,
    bf16,
    fp32,
    fp64
  };

  template<int RedArch, RedDataType r>
  consteval auto redVectorWidth() {
    if (RedArch < 900) {
      if (r == RedDataType::bf16 || r == RedDataType::fp16) {
        return 2;
      }
      return 1;
    }
    // Hopper and above
    if (r == RedDataType::bf16 || r == RedDataType::fp16) {
      return 8;
    }
    if (r == RedDataType::fp32) {
      return 4;
    }
    return 1;
  }
  using RedElement = __half;
  constexpr auto RE = RedDataType::fp16;
  constexpr auto TR_RED_ALIGNMENT = redVectorWidth<ARCH, RE>() * sizeof(RedElement);
  constexpr auto LR_RED_ALIGNMENT = cuda::std::is_same_v<double, RedElement> ? 1 :
  ARCH >= 900 ? sizeof(uint64_t) : sizeof(uint32_t);
  constexpr auto LR_PACKET_ALIGNMENT= LR_RED_ALIGNMENT * 2;
  // *2 to include flags
  constexpr auto PACKET_BUFFER_SIZE = 2 * AR_LATENCY_BOUND_THRESHOLD;

  enum class Regime {
    latency,
    throughput
  };

  template<Regime ruth, int PutArch, int RedArch>
  struct Reduce {
    static_assert(ruth == Regime::throughput && PutArch >= 700 && PutArch < 800);
    __device__ __forceinline__
    void operator()(const cuda::std::byte* __restrict__ const& src,
      cuda::std::byte* __restrict__ const& dst, const size_t& bytes) const {
      using RVD = RedAddType<RedElement, TR_RED_ALIGNMENT>;
      using RAT = RVD::Type;
      constexpr int redVW = TR_RED_ALIGNMENT / sizeof(RAT);
      using RedAddOp = RedAdd<RedArch, RAT, redVW>;
      constexpr RedAddOp op{};
      constexpr auto alignment = redVectorWidth<RedArch, RE>();
      using VT = cutlass::AlignedArray<RVD::RawType, redVW>;
      const int vP = static_cast<int>(bytes / TR_RED_ALIGNMENT);
      auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
      // use unrolled direct loads as pipelining is not necessary
      const auto threadElems = vP / threads;
      const auto trips = threadElems / unrollFactor;
      for (int i = 0; i < trips; ++i) {
        VT reginald[unrollFactor];
        uint indices[unrollFactor];
        // precompute indices
        cuda::static_for<unrollFactor>([&i, &indices](auto j) {
          indices[j] = (i * unrollFactor + j) * threads + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<unrollFactor>([&vS, &indices, &reginald](auto j) {
          reginald[j] = vS[indices[j]];
        });
        // rmem -> gmem reduction
        cuda::static_for<unrollFactor>([&vD, &indices, &reginald](auto j) {
          auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + indices[j]);
          op(dstP, reginald[j]);
        });
      }
      const auto residue = vP - trips * unrollFactor * threads;
      vS += (trips * unrollFactor * threads);
      vD += (trips * unrollFactor * threads);
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
        const auto v = vS[i];
        auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + i);
        op(dstP, v);
      }
    }
  };

  template<int RedArch>
  struct Reduce<Regime::throughput, 800, RedArch> {
    __device__ __forceinline__
    void operator()(const cuda::std::byte* __restrict__ const& src, cuda::std::byte* __restrict__ const& dst,
      cuda::std::byte* __restrict__ const& workspace, const size_t& bytes) const {
      using RVD = RedAddType<RedElement, TR_RED_ALIGNMENT>;
      using RAT = RVD::Type;
      constexpr int redVW = TR_RED_ALIGNMENT / sizeof(RAT);
      using RedAddOp = RedAdd<RedArch, RAT, redVW>;
      constexpr RedAddOp op{};
      constexpr auto copyAlignment = TR_RED_ALIGNMENT;
      if (bytes <= threads * copyAlignment * pipeStages * stageExtent) {
        using VT = cutlass::AlignedArray<RVD::RawType, redVW>;
        const int vP = static_cast<int>(bytes / TR_RED_ALIGNMENT);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        // use unrolled direct loads as pipelining is not necessary
        const auto threadElems = vP / threads;
        const auto trips = threadElems / unrollFactor;
        for (int i = 0; i < trips; ++i) {
          VT reginald[unrollFactor];
          uint indices[unrollFactor];
          // precompute indices
          cuda::static_for<unrollFactor>([&i, &indices](auto j) {
            indices[j] = (i * unrollFactor + j) * threads + threadIdx.x;
          });
          // gmem -> rmem
          cuda::static_for<unrollFactor>([&vS, &indices, &reginald](auto j) {
            reginald[j] = vS[indices[j]];
          });
          // rmem -> gmem reduction
          cuda::static_for<unrollFactor>([&vD, &indices, &reginald](auto j) {
            auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + indices[j]);
            op(dstP, reginald[j]);
          });
        }
        const auto residue = vP - trips * unrollFactor * threads;
        vS += (trips * unrollFactor * threads);
        vD += (trips * unrollFactor * threads);
        for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
          const auto v = vS[i];
          auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + i);
          op(dstP, v);
        }
      }
      else {
        constexpr int VectorWidth = copyAlignment / sizeof(RAT);
        constexpr int nAddOps = VectorWidth / RedAddOp::VectorWidth::value;
        static_assert(nAddOps == 1);
        using VT = cutlass::AlignedArray<RVD::RawType, VectorWidth, copyAlignment>;
        using AT = cutlass::AlignedArray<RVD::RawType, RedAddOp::VectorWidth::value>;
        static_assert(pipeStages >= 1);
        auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        const int stages = static_cast<int>(bytes / (threads * copyAlignment * stageExtent));
        cuda::static_for<pipeStages>([&vW, &vS](auto i) {
          cuda::static_for<stageExtent>([&i, &vW, &vS](auto j) {
            const int slot = ((i * stageExtent + j) * threads) + threadIdx.x;
            // async gmem -> smem
            cp_async_global_to_shared<copyAlignment>(vW + slot, vS + slot);
          });
          cute::cp_async_fence();
        });
        VT reginald[stageExtent];
        for (int i = pipeStages; i < stages; ++i) {
          cute::cp_async_wait<pipeStages - 1>();
          const int stage_out = i - pipeStages;
          const int cs = stage_out % pipeStages;
          cuda::static_for<stageExtent>([&i, &cs, &vW, &reginald, &vS](auto j) {
            const int csW = (cs * stageExtent + j) * threads + threadIdx.x;
            const long int slot = (i * stageExtent + j) * threads + threadIdx.x;
            // smem -> rmem
            reginald[j] = vW[csW];
            // async gmem -> smem prefetch
            cp_async_global_to_shared<copyAlignment>(vW + csW, vS + slot);
          });
          cuda::static_for<stageExtent>([&stage_out, &reginald, &vD](auto j) {
            const long int slot = (stage_out * stageExtent + j) * threads + threadIdx.x;
            // rmem -> gmem, reduction
            const auto v = reginald[j];
            auto* __restrict__ vDp = reinterpret_cast<RAT*>(vD + slot);
            cuda::static_for<nAddOps>([&vDp, &v](auto k) {
              AT aot{};
              aot[0] = v[k]; // <- Ideally, no MOV instructions would be emitted here
              op(vDp + k, aot);
            });
          });
          // commit async transfers from this stage
          cute::cp_async_fence();
        }
        // tail
        cuda::static_for<pipeStages>([&vW, &reginald, &vS, &vD, &stages](auto i) {
          const int stage = (stages - pipeStages) + i;
          const int cs = stage % pipeStages;
          cute::cp_async_wait<pipeStages - 1 - i>();
          cuda::static_for<stageExtent>([&i, &cs, &vW, &reginald, &vS, &stages](auto j) {
            const int csW = (cs * stageExtent + j) * threads + threadIdx.x;
            // smem -> rmem
            reginald[j] = vW[csW];
          });
          cuda::static_for<stageExtent>([&stage, &reginald, &vD](auto j) {
            const long int slot = (stage * stageExtent + j) * threads + threadIdx.x;
            // rmem -> gmem
            const auto v = reginald[j];
            auto* __restrict__ vDp = reinterpret_cast<RAT*>(vD + slot);
            cuda::static_for<nAddOps>([&vDp, &v](auto k) {
              AT aot{};
              aot[0] = v[k]; // <- Ideally, no MOV instructions would be emitted here
              op(vDp + k, aot);
            });
          });
        });
        // residue
        const auto cutoff = stages * static_cast<size_t>(threads * copyAlignment * stageExtent);
        const auto cutoffElems = cutoff / copyAlignment;
        const auto residue = (bytes - cutoff) / copyAlignment; // elements not bytes
        vS += cutoffElems;
        vD += cutoffElems;
        for (size_t i = threadIdx.x; i < residue; i += threads) {
          const auto v = vS[i];
          auto* __restrict__ vDp = reinterpret_cast<RAT*>(vD + i);
          cuda::static_for<nAddOps>([&vDp, &v](auto k) {
            AT aot{};
            aot[0] = v[k]; // <- Ideally, no MOV instructions would be emitted here
            op(vDp + k, aot);
          });
        }
      }
    }
  };

  // 8-byte Latency Regime Packet
  struct __align__(8) LRP8{
    using RT = uint32_t; // Raw Type
    RT data; // holds 2 half or 1 single precision value(s)
    RT flag;
    template<typename V>
    __device__ __forceinline__
    void pack(const V& v, const RT& flag_) {
      static_assert(sizeof(V) == sizeof(RT) && alignof(V) == alignof(RT));
      data = cuda::std::bit_cast<RT>(v);
      flag = flag_;
    }
    template<typename V>
    __device__ __forceinline__
    void unpack(V& v) {
      static_assert(sizeof(V) == sizeof(RT) && alignof(V) == alignof(RT));
      v = cuda::std::bit_cast<V>(data);
    }
  };
  using LRP8Raw = ulong;

  // 16-byte Latency Regime Packet
  struct __align__(16) LRP16{
    using RT = uint64_t;
    RT data; // holds 4 half, 2 single or 1 double precision value(s)
    RT flag;
    template<typename V>
    __device__ __forceinline__
    void pack(const V& v, const uint32_t& flag_) {
      static_assert(sizeof(V) == sizeof(RT) && alignof(V) == alignof(RT));
      data = cuda::std::bit_cast<RT>(v);
      flag = static_cast<RT>(flag_);
    }
    template<typename V>
    __device__ __forceinline__
    void unpack(V& v) {
      static_assert(sizeof(V) == sizeof(RT) && alignof(V) == alignof(RT));
      v = cuda::std::bit_cast<V>(data);
    }
  };
  using LRP16Raw = ulong2;

  // low-latency allReduce
  template<int PutArch, int RedArch>
  struct Reduce<Regime::latency, PutArch, RedArch> {
    __device__ __forceinline__
    void operator()(const cuda::std::byte* __restrict__ const& src, // non-symmetric
      cuda::std::byte* __restrict__ const& rStaging, // remote, symmetric
      cuda::std::byte* __restrict__ const& lStaging, // local, symmetric
      cuda::std::byte* __restrict__ const& dst,  // non-symmetric
      uint8_t* __restrict__ const& flags, // non-symmetric
      const uint& bytes) const {
      constexpr auto isLRP16 = LR_RED_ALIGNMENT == sizeof(uint64_t);
      constexpr auto alignment = LR_RED_ALIGNMENT;
      using LRP = cuda::std::conditional_t<isLRP16, LRP16, LRP8>;
      using LRPRaw = cuda::std::conditional_t<isLRP16, LRP16Raw, LRP8Raw>;
      using RVD = RedAddType<RedElement, alignment>;
      constexpr int vectorWidth = alignment / sizeof(RVD::RawType);
      using RAT = RVD::Type;
      using VT = cutlass::AlignedArray<typename RVD::RawType, vectorWidth>;
      static_assert(cuda::std::is_trivially_copyable_v<VT>);
      const int vP = static_cast<int>(bytes / LR_RED_ALIGNMENT);
      auto* __restrict__ vD = reinterpret_cast<LRPRaw*>(rStaging);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
      const auto threadElems = vP / threads;
      const auto trips = threadElems / unrollFactor;
      using RedAddOp = RedAdd<RedArch, RAT, alignment / sizeof(RAT)>;
      constexpr RedAddOp op{};
      using RVT = cutlass::AlignedArray<typename RVD::RawType, alignment / sizeof(RAT)>;
      // 1. Put packets
      for (int i = 0; i < trips; ++i) {
        VT reginald[unrollFactor];
        uint indices[unrollFactor];
        uint32_t cachedFlags[unrollFactor];
        // precompute indices
        cuda::static_for<unrollFactor>([&i, &indices](auto j) {
          indices[j] = (i * unrollFactor + j) * threads + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<unrollFactor>([&vS, &indices, &reginald, &cachedFlags, &flags](auto j) {
          const auto cf = static_cast<uint32_t>(flags[indices[j]]);
          cachedFlags[j] = cf == 0U ? 1U : 0U;
          reginald[j] = vS[indices[j]];
        });
        // rmem -> gmem, packets
        cuda::static_for<unrollFactor>([&vD, &indices, &reginald, &cachedFlags](auto j) {
          LRP lrp{};
          lrp.pack(reginald[j], cachedFlags[j]);
          cuda::atomic_ref<LRPRaw, cuda::thread_scope_system> packet{*(vD + indices[j])};
          packet.store(cuda::std::bit_cast<LRPRaw>(lrp), cuda::memory_order_relaxed);
        });
      }
      const auto residue = vP - trips * unrollFactor * threads;
      vS += (trips * unrollFactor * threads);
      vD += (trips * unrollFactor * threads);
      auto* __restrict__ flagsRes = flags + (trips * unrollFactor * threads);
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
        LRP lrp{};
        const auto f = static_cast<uint32_t>(flagsRes[i]);
        const auto cf = f == 0U ? 1U : 0U;
        lrp.pack(vS[i], cf);
        cuda::atomic_ref<LRPRaw, cuda::thread_scope_system> packet{*(vD + i)};
        packet.store(cuda::std::bit_cast<LRPRaw>(lrp), cuda::memory_order_relaxed);
      }

      // 2. Do reduction
      auto* __restrict__ rvD = reinterpret_cast<VT*>(dst);
      auto* __restrict__ rvS = reinterpret_cast<LRPRaw*>(lStaging);
      for (int i = 0; i < trips; ++i) {
        uint indices[unrollFactor];
        uint32_t cachedFlags[unrollFactor];
        // precompute indices
        cuda::static_for<unrollFactor>([&i, &indices](auto j) {
          indices[j] = (i * unrollFactor + j) * threads + threadIdx.x;
        });
        cuda::static_for<unrollFactor>([&indices, &cachedFlags, &flags](auto j) {
          const auto f = static_cast<uint32_t>(flags[indices[j]]);
          cachedFlags[j] = f == 0U ? 1U : 0U;
          // flip flag for subsequent use
          flags[indices[j]] = static_cast<uint8_t>(cachedFlags[j]);
        });
        // await packet
        cuda::static_for<unrollFactor>([&rvS, &rvD, &indices, &cachedFlags](auto j) {
          const auto expectedFlag = cachedFlags[j];
          cuda::atomic_ref<LRPRaw, cuda::thread_scope_system> packet{*(rvS + indices[j])};
          auto currentPacket = cuda::std::bit_cast<LRP>(packet.load(cuda::memory_order_relaxed));
          auto hPA = currentPacket.flag == expectedFlag;
          while (!hPA) {
            currentPacket = cuda::std::bit_cast<LRP>(packet.load(cuda::memory_order_relaxed));
            hPA = currentPacket.flag == expectedFlag;
          }
          RVT val{};
          currentPacket.unpack(val);
          // do reduction
          op(reinterpret_cast<RAT*>(rvD + indices[j]), val);
        });
      }
      rvS += (trips * unrollFactor * threads);
      rvD += (trips * unrollFactor * threads);
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += threads) {
        const auto f = static_cast<uint32_t>(flagsRes[i]);
        const auto expectedFlag = f == 0U ? 1U : 0U;
        flagsRes[i] = static_cast<uint8_t>(expectedFlag);
        cuda::atomic_ref<LRPRaw, cuda::thread_scope_system> packet{*(rvS + i)};
        auto currentPacket = cuda::std::bit_cast<LRP>(packet.load(cuda::memory_order_relaxed));
        auto hPA = currentPacket.flag == expectedFlag; // hasPacketArrived
        while (!hPA) {
          currentPacket = cuda::std::bit_cast<LRP>(packet.load(cuda::memory_order_relaxed));
          hPA = currentPacket.flag == expectedFlag;
        }
        RVT val{};
        currentPacket.unpack(val);
        // do reduction
        op(reinterpret_cast<RAT*>(rvD + i), val);
      }
    }
  };
}
#endif //TACK_REDUCE_CUH