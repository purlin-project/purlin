//
// Created by Osayamen on 4/15/26.
//

#ifndef SUTURE_MATH_CUH
#define SUTURE_MATH_CUH
#include <cuda/utility>
namespace suture {
  template<typename T, typename S>
  struct Converter {
    __device__ auto operator()(const S &x) const {
      return static_cast<T>(x);
    }
  };

  template<>
  struct Converter<float, __half> {
    __device__ auto operator()(const __half &x) const {
      return __half2float(x);
    }
  };

  template<>
  struct Converter<__half, float> {
    __device__ auto operator()(const float &x) const {
      return __float2half(x);
    }
  };

  template<>
  struct Converter<float, __nv_bfloat16> {
    __device__ auto operator()(const __nv_bfloat16 &x) const {
      return __bfloat162float(x);
    }
  };

  template<>
  struct Converter<__nv_bfloat16, float> {
    __device__ auto operator()(const float &x) const {
      return __float2bfloat16(x);
    }
  };
  template<>
  struct Converter<float2, __half2> {
    __device__ auto operator()(const __half2 &x) const {
      return __half22float2(x);
    }
  };

  template<>
  struct Converter<__half2, float2> {
    __device__ auto operator()(const float2 &x) const {
      return __float22half2_rn(x);
    }
  };

  template<>
  struct Converter<float2, __nv_bfloat162> {
    __device__ auto operator()(const __nv_bfloat162 &x) const {
      return __bfloat1622float2(x);
    }
  };

  template<>
  struct Converter<__nv_bfloat162, float2> {
    __device__ auto operator()(const float2 &x) const {
      return __float22bfloat162_rn(x);
    }
  };
  template<typename T, int nArch>
  struct InplaceSum {
    __device__ __forceinline__
    void operator()(T& lhs, const T& rhs) const {
      lhs = lhs + rhs;
    }
  };
  template<int nArch>
  struct InplaceSum<float2, nArch> {
    __device__ __forceinline__
    void operator()(float2& lhs, const float2& rhs) const {
      lhs = float2{lhs.x + rhs.x, lhs.y + rhs.y};
    }
  };

  template<int nArch>
  struct ArrayInplaceSum {
    template<typename T>
    __device__ __forceinline__
    void operator()(T& accum, const T& x) {
      InplaceSum<typename T::value_type, nArch> sum{};
      cuda::static_for<accum.size()>([&](auto idx) {
        sum(accum[idx], x[idx]);
      });
    }
  };
}
#endif //SUTURE_MATH_CUH