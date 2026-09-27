//
// Created by Osayamen on 4/15/26.
//

#ifndef PURLIN_MATH_CUH
#define PURLIN_MATH_CUH
#include "static_for.cuh"
#include <cuda/std/limits>
#include <cuda/utility>
namespace purlin {
  // Reduction operator vocabulary: collectives carry one of these as a template
  // parameter and each Atom lowers it to its arch's ArrayInplace* functor.
  enum class ReduceOp {
    add,
    mul,
    max
  };
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
  template<typename T>
  struct InplaceOne {
    __device__ __forceinline__
    void operator()(T& v) const {
      v = static_cast<T>(1);
    }
  };
  template<>
  struct InplaceOne<float2> {
    __device__ __forceinline__
    void operator()(float2& v) const {
      v = float2{1.f, 1.f};
    }
  };
  // Max identity is -inf rather than lowest-finite, so any data value wins.
  template<typename T>
  struct InplaceLowest {
    __device__ __forceinline__
    void operator()(T& v) const {
      v = -cuda::std::numeric_limits<T>::infinity();
    }
  };
  template<>
  struct InplaceLowest<float2> {
    __device__ __forceinline__
    void operator()(float2& v) const {
      constexpr auto lowest = -cuda::std::numeric_limits<float>::infinity();
      v = float2{lowest, lowest};
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
  template<typename T, int nArch>
  struct InplaceMul {
    __device__ __forceinline__
    void operator()(T& lhs, const T& rhs) const {
      lhs = lhs * rhs;
    }
  };
  // cortex.cuh specializes <float2, 1000> with Blackwell's packed f32x2 multiply.
  template<int nArch>
  struct InplaceMul<float2, nArch> {
    __device__ __forceinline__
    void operator()(float2& lhs, const float2& rhs) const {
      lhs = float2{lhs.x * rhs.x, lhs.y * rhs.y};
    }
  };
  template<typename T, int nArch>
  struct InplaceMax {
    __device__ __forceinline__
    void operator()(T& lhs, const T& rhs) const {
      lhs = lhs < rhs ? rhs : lhs;
    }
  };
  template<int nArch>
  struct InplaceMax<float2, nArch> {
    __device__ __forceinline__
    void operator()(float2& lhs, const float2& rhs) const {
      lhs = float2{fmaxf(lhs.x, rhs.x), fmaxf(lhs.y, rhs.y)};
    }
  };

  template<int nArch>
  struct ArrayInplaceSum {
    template<typename T>
    using Identity = InplaceZero<T>;
    template<typename T>
    __device__ __forceinline__
    void operator()(T& accum, const T& x) const {
      InplaceSum<typename T::value_type, nArch> sum{};
      purlin::static_for<T::kElements>([&](auto idx) {
        sum(accum[idx], x[idx]);
      });
    }
  };
  template<int nArch>
  struct ArrayInplaceMul {
    template<typename T>
    using Identity = InplaceOne<T>;
    template<typename T>
    __device__ __forceinline__
    void operator()(T& accum, const T& x) const {
      InplaceMul<typename T::value_type, nArch> mul{};
      purlin::static_for<T::kElements>([&](auto idx) {
        mul(accum[idx], x[idx]);
      });
    }
  };
  template<int nArch>
  struct ArrayInplaceMax {
    template<typename T>
    using Identity = InplaceLowest<T>;
    template<typename T>
    __device__ __forceinline__
    void operator()(T& accum, const T& x) const {
      InplaceMax<typename T::value_type, nArch> max{};
      purlin::static_for<T::kElements>([&](auto idx) {
        max(accum[idx], x[idx]);
      });
    }
  };

  // Lowers the ReduceOp vocabulary to the arch's ArrayInplace* functor.
  template<ReduceOp ro, int nArch>
  struct LoweredReduceOp;
  template<int nArch>
  struct LoweredReduceOp<ReduceOp::add, nArch> {
    using type = ArrayInplaceSum<nArch>;
  };
  template<int nArch>
  struct LoweredReduceOp<ReduceOp::mul, nArch> {
    using type = ArrayInplaceMul<nArch>;
  };
  template<int nArch>
  struct LoweredReduceOp<ReduceOp::max, nArch> {
    using type = ArrayInplaceMax<nArch>;
  };
}
#endif //PURLIN_MATH_CUH