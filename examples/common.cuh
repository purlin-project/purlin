//
// Created by osayamen on 2/11/26.
//

#ifndef TACK_COMMON_CUH
#define TACK_COMMON_CUH
#include <curanddx.hpp>
#include <cute/int_tuple.hpp>
#include <cutlass/array.h>
template<typename T, typename S>
    struct Converter {
    __device__ auto operator()(const S& x) const {
        return static_cast<T>(x);
    }
};
template<>
struct Converter<__half, float> {
    __device__ auto operator()(const float& x) const {
        return  __float2half(x);
    }
};
template<>
struct Converter<__nv_bfloat16, float> {
    __device__ auto operator()(const float& x) const {
        return  __float2bfloat16(x);
    }
};

template<typename T, int Alignment = 16>
    struct VectorTypeDescriptor {
    using VectorWidth = cute::C<Alignment / sizeof(T)>;
    using VectorType = cutlass::AlignedArray<T, VectorWidth::value, Alignment>;
};

template <int Arch, bool predicate, typename Element>
__global__ void generateRandUniform(
    Element* __restrict__ out,
    const __grid_constant__ size_t n,
    const __grid_constant__ size_t seed,
    const __grid_constant__ float  minv,
    const __grid_constant__ float  maxv,
    const __grid_constant__ unsigned long long global_offset = 0ULL
) {
    using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
                         curanddx::SM<Arch>() +
                         curanddx::Thread());
    const auto tid = static_cast<unsigned long long int>(blockIdx.x)
    * blockDim.x + threadIdx.x;

    constexpr int vF = 4;
    const size_t out_base = static_cast<size_t>(tid) * vF;

    RNG rng(seed, tid, global_offset);

    curanddx::uniform<float> dist(minv, maxv);

    auto v = dist.generate4(rng);

    constexpr Converter<Element, float> storeOp{};

    if constexpr (predicate) {
        if (out_base + (vF - 1) >= n) return;
        // n % 4 == 0
        using VTD = VectorTypeDescriptor<Element, vF * sizeof(Element)>;
        using VT = VTD::VectorType;
        static_assert(VTD::VectorWidth::value == vF);
        auto* __restrict__ vo = reinterpret_cast<VT*>(out);
        VT vt{};
        vt[0] = storeOp(v.x);
        vt[1] = storeOp(v.y);
        vt[2] = storeOp(v.z);
        vt[3] = storeOp(v.w);
        vo[tid] = vt;
    }
    else {
        if (out_base >= n) return;
        out[out_base + 0] = storeOp(v.x);
        if (out_base + 1 < n) out[out_base + 1] = storeOp(v.y);
        if (out_base + 2 < n) out[out_base + 2] = storeOp(v.z);
        if (out_base + 3 < n) out[out_base + 3] = storeOp(v.w);
    }
}

template<int Arch, typename Element>
__host__ __forceinline__
void randUniform(Element* __restrict__ const& out, const size_t& n, const size_t& seed, const float& minv,
    const float& maxv, cudaStream_t stream) {
    constexpr uint threads = 128;
    const auto blocks = static_cast<uint>(cute::ceil_div(n, threads * 4));
    if (n % 4 == 0) {
        generateRandUniform<Arch, true><<<blocks, threads, 0, stream>>>(out, n, seed, minv, maxv);
    }
    else {
        generateRandUniform<Arch, false><<<blocks, threads, 0, stream>>>(out, n, seed, minv, maxv);
    }
}

template <typename Element>
consteval const char* element_string() {
    static_assert(
      cuda::std::is_same_v<Element, __half> ||
      cuda::std::is_same_v<Element, __nv_bfloat16> ||
      cuda::std::is_same_v<Element, float> ||
      cuda::std::is_same_v<Element, double>,
      "Unsupported Element type"
    );
    if constexpr (cuda::std::is_same_v<Element, double>) return "fp64";
    else if constexpr (cuda::std::is_same_v<Element, float>) return "fp32";
    else if constexpr (cuda::std::is_same_v<Element, __half>) return "fp16";
    else return "bf16";
}
#endif //TACK_COMMON_CUH