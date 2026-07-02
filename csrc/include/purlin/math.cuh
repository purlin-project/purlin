//
// Created by Osayamen on 4/15/26.
//

#ifndef PURLIN_MATH_CUH
#define PURLIN_MATH_CUH
#include <cuda/utility>
namespace purlin {
  template<typename T, int N, int Alignment = sizeof(T) * N>
  struct alignas(Alignment) AlignedArray {
    T data[N];
    static constexpr int kElements = N;
    using value_type = T;

    __host__ __device__ __forceinline__
    constexpr T& operator[](int i) {
      return data[i];
    }

    __host__ __device__ __forceinline__
    constexpr const T& operator[](int i) const {
      return data[i];
    }

    __host__ __device__ __forceinline__
    static constexpr int size() {
      return N;
    }
  };

  struct fp8x2_e4m3_raw {
    __nv_fp8x2_storage_t storage;
  };
  struct fp8x2_e5m2_raw {
    __nv_fp8x2_storage_t storage;
  };

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
  struct Converter<float, __nv_fp8_e4m3> {
    __device__ auto operator()(const __nv_fp8_e4m3 &x) const {
      return x.operator float();
    }
  };

  template<>
  struct Converter<__nv_fp8_e4m3, float> {
    __device__ auto operator()(const float &x) const {
      return __nv_fp8_e4m3{x};
    }
  };

  template<>
  struct Converter<float, __nv_fp8_e5m2> {
    __device__ auto operator()(const __nv_fp8_e5m2 &x) const {
      return x.operator float();
    }
  };

  template<>
  struct Converter<__nv_fp8_e5m2, float> {
    __device__ auto operator()(const float &x) const {
      return __nv_fp8_e5m2{x};
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
  struct Converter<__half2_raw, float2> {
    __device__ auto operator()(const float2 &x) const {
      return static_cast<__half2_raw>(__float22half2_rn(x));
    }
  };

  template<>
  struct Converter<float2, __nv_fp8x2_e4m3> {
    template<typename T>
    __device__ auto operator()(const T &x) const {
      static_assert(cuda::std::is_same_v<T, __nv_fp8x2_e4m3> || cuda::std::is_same_v<T, fp8x2_e4m3_raw>);
      if constexpr (cuda::std::is_same_v<T, __nv_fp8x2_e4m3>) {
        return x.operator float2();
      }
      else {
        __nv_fp8x2_e4m3 val{};
        val.__x = x.storage;
        return val.operator float2();
      }
    }
  };

  template<>
  struct Converter<__nv_fp8x2_e4m3, float2> {
    __device__ auto operator()(const float2 &x) const {
      return __nv_fp8x2_e4m3{x};
    }
  };

  template<>
  struct Converter<float2, __nv_fp8x2_e5m2> {
    template<typename T>
    __device__ auto operator()(const T &x) const {
      static_assert(cuda::std::is_same_v<T, __nv_fp8x2_e5m2> || cuda::std::is_same_v<T, fp8x2_e5m2_raw>);
      if constexpr (cuda::std::is_same_v<T, __nv_fp8x2_e5m2>) {
        return x.operator float2();
      }
      else {
        __nv_fp8x2_e5m2 val{};
        val.__x = x.storage;
        return val.operator float2();
      }
    }
  };

  template<>
  struct Converter<__nv_fp8x2_e5m2, float2> {
    __device__ auto operator()(const float2 &x) const {
      return __nv_fp8x2_e5m2{x};
    }
  };

  template<>
  struct Converter<fp8x2_e4m3_raw, float2> {
    __device__ auto operator()(const float2 &x) const {
      return fp8x2_e4m3_raw{__nv_fp8x2_e4m3{x}.__x};
    }
  };

  template<>
  struct Converter<fp8x2_e5m2_raw, float2> {
    __device__ auto operator()(const float2 &x) const {
      return fp8x2_e5m2_raw{__nv_fp8x2_e5m2{x}.__x};
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
  template<>
  struct Converter<__nv_bfloat162_raw, float2> {
    __device__ auto operator()(const float2 &x) const {
      return static_cast<__nv_bfloat162_raw>(__float22bfloat162_rn(x));
    }
  };
  template<typename T>
  struct InplaceZero {
    __device__ __forceinline__
    void operator()(T& v) const {
      v = static_cast<T>(0);
    }
  };
  template<>
  struct InplaceZero<float2> {
    __device__ __forceinline__
    void operator()(float2& v) const {
      v = float2{0.f, 0.f};
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
    void operator()(T& accum, const T& x) const {
      InplaceSum<typename T::value_type, nArch> sum{};
      cuda::static_for<T::kElements>([&](auto idx) {
        sum(accum[idx], x[idx]);
      });
    }
  };
}
#endif //PURLIN_MATH_CUH