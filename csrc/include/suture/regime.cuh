//
// Created by Osy on 4/8/26.
//

#ifndef SUTURE_REGIME_CUH
#define SUTURE_REGIME_CUH
namespace suture {
  enum class Regime {
    latency,
    throughput
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
}
#endif //SUTURE_REGIME_CUH