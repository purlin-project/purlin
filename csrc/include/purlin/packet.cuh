//
// Created by Osayamen on 4/8/26.
//

#ifndef PURLIN_REGIME_CUH
#define PURLIN_REGIME_CUH
namespace purlin {
  // 16-byte Latency Regime Packet
  struct __align__(16) LRP16{
    using RT = uint64_t;
    RT data; // holds 4 half, 2 single or 1 double precision value(s)
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
  using LRP16Raw = ulong2;
}
#endif //PURLIN_REGIME_CUH