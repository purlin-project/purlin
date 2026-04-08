//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_FASCIA_CUH
#define SUTURE_FASCIA_CUH
#include <cuda/utility>

#include "base.cuh"
#include "copy.cuh"
#include "regime.cuh"
#include "rvt.cuh"
namespace suture::fascia {
  // nArch is implicitly 700 in fascia
  constexpr int nArch = 700;
  template<typename Element>
  __device__ __forceinline__
  consteval auto redWidth() {
    static_assert(cuda::std::is_same_v<Element, __half> ||
      cuda::std::is_same_v<Element, float> ||
      cuda::std::is_same_v<Element, double>);
    if constexpr (cuda::std::is_same_v<Element, __half>) {
      return 2;
    }
    return 1;
  }

  struct Red {
    template<typename T>
    __device__ __forceinline__
    void operator()(T* __restrict__ const& dst, const T& v) {
      // convert from raw type to datatype
      using RawElement = T::value_type;
      using BaseElement = RawToDataType<RawElement>::type;
      using RedAddOp = RedAdd<nArch, BaseElement, v.size()>;
      constexpr RedAddOp op{};
      auto* __restrict__ dstP = reinterpret_cast<BaseElement*>(dst);
      op(dstP, v);
    }
  };
}

template<suture::StateSpace sourceSpace>
struct suture::Atom<700, sourceSpace> {
  static_assert(fascia::nArch == 700);
  using MaxAlignmentBytes = cuda::std::integral_constant<int, 16>;
  template<int threads, int unrollFactor = 2, int AlignmentBytes = MaxAlignmentBytes::value>
  __device__ __forceinline__
  static void put(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes) {
    fascia::peerOp<threads, unrollFactor, AlignmentBytes, ST, AlignedType<AlignmentBytes>>(src, dst, bytes);
  }

  template<int threads, int unrollFactor = 2, int AlignmentBytes = MaxAlignmentBytes::value>
  __device__ __forceinline__
  static void get(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes /*in bytes*/) {
    put<threads, unrollFactor, AlignmentBytes>(src, dst, bytes);
  }

  // throughput regime
  template<int threads, int unrollFactor, typename Element>
  __device__ __forceinline__
  static void atomicReduce(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes /*in bytes*/) {
    constexpr auto AlignmentBytes = fascia::redWidth<Element>() * sizeof(Element);
    using RVD = RedAddType<Element, AlignmentBytes>;
    fascia::peerOp<threads, unrollFactor, AlignmentBytes, fascia::Red, typename RVD::RawType>(src, dst, bytes);
  }

  // latency regime
  template<int threads, int unrollFactor, typename Element>
  __device__ __forceinline__
  static void atomicReduceLL(const cuda::std::byte* __restrict__ const& src, // non-symmetric
    cuda::std::byte* __restrict__ const& rStaging, // remote, symmetric
    cuda::std::byte* __restrict__ const& lStaging, // local, symmetric
    cuda::std::byte* __restrict__ const& dst,  // non-symmetric
    uint8_t* __restrict__ const& flags, // non-symmetric
    const size_t& bytes) {
    constexpr auto dataAlignment = sizeof(uint32_t);
    using LRP = LRP8;
    using LRPRaw = LRP8Raw;
    using RVD = RedAddType<Element, dataAlignment>;
    constexpr int vectorWidth = dataAlignment / sizeof(RVD::RawType);
    using RAT = RVD::Type;
    using VT = cutlass::AlignedArray<typename RVD::RawType, vectorWidth>;
    static_assert(cuda::std::is_trivially_copyable_v<VT>);
    const auto vP = bytes / dataAlignment;
    auto* __restrict__ vD = reinterpret_cast<LRPRaw*>(rStaging);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const auto threadElems = vP / threads;
    const auto trips = threadElems / unrollFactor;
    using RedAddOp = RedAdd<fascia::nArch, RAT, dataAlignment / sizeof(RAT)>;
    constexpr RedAddOp op{};
    using RVT = cutlass::AlignedArray<typename RVD::RawType, dataAlignment / sizeof(RAT)>;
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
#endif //SUTURE_FASCIA_CUH