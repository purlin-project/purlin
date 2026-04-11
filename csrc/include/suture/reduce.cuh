//
// Created by Osayamen on 3/27/26.
//

#ifndef SUTURE_REDUCE_CUH
#define SUTURE_REDUCE_CUH
#include <cuda/atomic>
#include <cuda/utility>
#include <cutlass/array.h>

#include "copy.cuh"
#include "regime.cuh"
#include "rvt.cuh"

namespace suture {
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

  template<Regime ruth, int PutArch, int RedArch>
  struct AtomicReduce {
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
      const auto threadElems = vP / kThreads;
      const auto trips = threadElems / kUnrollFactor;
      for (int i = 0; i < trips; ++i) {
        VT reginald[kUnrollFactor];
        uint indices[kUnrollFactor];
        // precompute indices
        cuda::static_for<kUnrollFactor>([&i, &indices](auto j) {
          indices[j] = (i * kUnrollFactor + j) * kThreads + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<kUnrollFactor>([&vS, &indices, &reginald](auto j) {
          reginald[j] = vS[indices[j]];
        });
        // rmem -> gmem reduction
        cuda::static_for<kUnrollFactor>([&vD, &indices, &reginald](auto j) {
          auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + indices[j]);
          op(dstP, reginald[j]);
        });
      }
      const auto residue = vP - trips * kUnrollFactor * kThreads;
      vS += (trips * kUnrollFactor * kThreads);
      vD += (trips * kUnrollFactor * kThreads);
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += kThreads) {
        const auto v = vS[i];
        auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + i);
        op(dstP, v);
      }
    }
  };

  template<int RedArch>
  struct AtomicReduce<Regime::throughput, 800, RedArch> {
    __device__ __forceinline__
    void operator()(const cuda::std::byte* __restrict__ const& src, cuda::std::byte* __restrict__ const& dst,
      cuda::std::byte* __restrict__ const& workspace, const size_t& bytes) const {
      using RVD = RedAddType<RedElement, TR_RED_ALIGNMENT>;
      using RAT = RVD::Type;
      constexpr int redVW = TR_RED_ALIGNMENT / sizeof(RAT);
      using RedAddOp = RedAdd<RedArch, RAT, redVW>;
      constexpr RedAddOp op{};
      constexpr auto copyAlignment = TR_RED_ALIGNMENT;
      if (bytes <= kThreads * copyAlignment * kPipeStages * kStageExtent) {
        using VT = cutlass::AlignedArray<RVD::RawType, redVW>;
        const int vP = static_cast<int>(bytes / TR_RED_ALIGNMENT);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        // use unrolled direct loads as pipelining is not necessary
        const auto threadElems = vP / kThreads;
        const auto trips = threadElems / kUnrollFactor;
        for (int i = 0; i < trips; ++i) {
          VT reginald[kUnrollFactor];
          uint indices[kUnrollFactor];
          // precompute indices
          cuda::static_for<kUnrollFactor>([&i, &indices](auto j) {
            indices[j] = (i * kUnrollFactor + j) * kThreads + threadIdx.x;
          });
          // gmem -> rmem
          cuda::static_for<kUnrollFactor>([&vS, &indices, &reginald](auto j) {
            reginald[j] = vS[indices[j]];
          });
          // rmem -> gmem reduction
          cuda::static_for<kUnrollFactor>([&vD, &indices, &reginald](auto j) {
            auto* __restrict__ dstP = reinterpret_cast<RAT*>(vD + indices[j]);
            op(dstP, reginald[j]);
          });
        }
        const auto residue = vP - trips * kUnrollFactor * kThreads;
        vS += (trips * kUnrollFactor * kThreads);
        vD += (trips * kUnrollFactor * kThreads);
        for (int i = static_cast<int>(threadIdx.x); i < residue; i += kThreads) {
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
        static_assert(kPipeStages >= 1);
        auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
        auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
        const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
        const int stages = static_cast<int>(bytes / (kThreads * copyAlignment * kStageExtent));
        cuda::static_for<kPipeStages>([&vW, &vS](auto i) {
          cuda::static_for<kStageExtent>([&i, &vW, &vS](auto j) {
            const int slot = ((i * kStageExtent + j) * kThreads) + threadIdx.x;
            // async gmem -> smem
            cpAsync<copyAlignment>(vW + slot, vS + slot);
          });
          cpAsyncCommit();
        });
        VT reginald[kStageExtent];
        for (int i = kPipeStages; i < stages; ++i) {
          cpAsyncWait<kPipeStages - 1>();
          const int stage_out = i - kPipeStages;
          const int cs = stage_out % kPipeStages;
          cuda::static_for<kStageExtent>([&i, &cs, &vW, &reginald, &vS](auto j) {
            const int csW = (cs * kStageExtent + j) * kThreads + threadIdx.x;
            const long int slot = (i * kStageExtent + j) * kThreads + threadIdx.x;
            // smem -> rmem
            reginald[j] = vW[csW];
            // async gmem -> smem prefetch
            cpAsync<copyAlignment>(vW + csW, vS + slot);
          });
          cuda::static_for<kStageExtent>([&stage_out, &reginald, &vD](auto j) {
            const long int slot = (stage_out * kStageExtent + j) * kThreads + threadIdx.x;
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
          cpAsyncCommit();
        }
        // tail
        cuda::static_for<kPipeStages>([&vW, &reginald, &vS, &vD, &stages](auto i) {
          const int stage = (stages - kPipeStages) + i;
          const int cs = stage % kPipeStages;
          cpAsyncWait<kPipeStages - 1 - i>();
          cuda::static_for<kStageExtent>([&i, &cs, &vW, &reginald, &vS, &stages](auto j) {
            const int csW = (cs * kStageExtent + j) * kThreads + threadIdx.x;
            // smem -> rmem
            reginald[j] = vW[csW];
          });
          cuda::static_for<kStageExtent>([&stage, &reginald, &vD](auto j) {
            const long int slot = (stage * kStageExtent + j) * kThreads + threadIdx.x;
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
        const auto cutoff = stages * static_cast<size_t>(kThreads * copyAlignment * kStageExtent);
        const auto cutoffElems = cutoff / copyAlignment;
        const auto residue = (bytes - cutoff) / copyAlignment; // elements not bytes
        vS += cutoffElems;
        vD += cutoffElems;
        for (size_t i = threadIdx.x; i < residue; i += kThreads) {
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

  // low-latency allReduce
  template<int PutArch, int RedArch>
  struct AtomicReduce<Regime::latency, PutArch, RedArch> {
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
      const auto threadElems = vP / kThreads;
      const auto trips = threadElems / kUnrollFactor;
      using RedAddOp = RedAdd<RedArch, RAT, alignment / sizeof(RAT)>;
      constexpr RedAddOp op{};
      using RVT = cutlass::AlignedArray<typename RVD::RawType, alignment / sizeof(RAT)>;
      // 1. Put packets
      for (int i = 0; i < trips; ++i) {
        VT reginald[kUnrollFactor];
        uint indices[kUnrollFactor];
        uint32_t cachedFlags[kUnrollFactor];
        // precompute indices
        cuda::static_for<kUnrollFactor>([&i, &indices](auto j) {
          indices[j] = (i * kUnrollFactor + j) * kThreads + threadIdx.x;
        });
        // gmem -> rmem
        cuda::static_for<kUnrollFactor>([&vS, &indices, &reginald, &cachedFlags, &flags](auto j) {
          const auto cf = static_cast<uint32_t>(flags[indices[j]]);
          cachedFlags[j] = cf == 0U ? 1U : 0U;
          reginald[j] = vS[indices[j]];
        });
        // rmem -> gmem, packets
        cuda::static_for<kUnrollFactor>([&vD, &indices, &reginald, &cachedFlags](auto j) {
          LRP lrp{};
          lrp.pack(reginald[j], cachedFlags[j]);
          cuda::atomic_ref<LRPRaw, cuda::thread_scope_system> packet{*(vD + indices[j])};
          packet.store(cuda::std::bit_cast<LRPRaw>(lrp), cuda::memory_order_relaxed);
        });
      }
      const auto residue = vP - trips * kUnrollFactor * kThreads;
      vS += (trips * kUnrollFactor * kThreads);
      vD += (trips * kUnrollFactor * kThreads);
      auto* __restrict__ flagsRes = flags + (trips * kUnrollFactor * kThreads);
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += kThreads) {
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
        uint indices[kUnrollFactor];
        uint32_t cachedFlags[kUnrollFactor];
        // precompute indices
        cuda::static_for<kUnrollFactor>([&i, &indices](auto j) {
          indices[j] = (i * kUnrollFactor + j) * kThreads + threadIdx.x;
        });
        cuda::static_for<kUnrollFactor>([&indices, &cachedFlags, &flags](auto j) {
          const auto f = static_cast<uint32_t>(flags[indices[j]]);
          cachedFlags[j] = f == 0U ? 1U : 0U;
          // flip flag for subsequent use
          flags[indices[j]] = static_cast<uint8_t>(cachedFlags[j]);
        });
        // await packet
        cuda::static_for<kUnrollFactor>([&rvS, &rvD, &indices, &cachedFlags](auto j) {
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
      rvS += (trips * kUnrollFactor * kThreads);
      rvD += (trips * kUnrollFactor * kThreads);
      for (int i = static_cast<int>(threadIdx.x); i < residue; i += kThreads) {
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
#endif //SUTURE_REDUCE_CUH