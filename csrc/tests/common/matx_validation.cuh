#ifndef PURLIN_TESTS_COMMON_MATX_VALIDATION_CUH
#define PURLIN_TESTS_COMMON_MATX_VALIDATION_CUH

#include <algorithm>
#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>
#include <matx.h>

#include "checks.cuh"

namespace bench {

template<typename Element>
using MatXElement = cuda::std::conditional_t<cuda::std::is_same_v<Element, __half>,
  matx::matxFp16, cuda::std::conditional_t<
    cuda::std::is_same_v<Element, __nv_bfloat16>, matx::matxBf16, Element>>;

namespace detail {
template<typename Element>
inline unsigned long long matxMismatchesSlice(const Element* actual,
  const Element* reference, const size_t count, cudaStream_t stream) {
  using TensorElement = MatXElement<Element>;
  matx::cudaExecutor executor{stream};
  auto matches = matx::make_tensor<long int>({});
  auto actualTensor = matx::make_tensor<TensorElement>(
    reinterpret_cast<TensorElement*>(const_cast<Element*>(actual)),
    {1, static_cast<matx::index_t>(count)});
  auto referenceTensor = matx::make_tensor<TensorElement>(
    reinterpret_cast<TensorElement*>(const_cast<Element*>(reference)),
    {1, static_cast<matx::index_t>(count)});
  // isclose yields int; widen before summing so large slices do not wrap
  (matches = matx::sum(matx::as_type<long int>(
    matx::isclose(actualTensor, referenceTensor, 0, 0)))).run(executor);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return static_cast<unsigned long long>(count) -
    static_cast<unsigned long long>(matches());
}
} // namespace detail

template<typename Element>
inline unsigned long long matxMismatches(const Element* actual,
  const Element* reference, const size_t count, cudaStream_t stream) {
  // The elementwise comparison misindexes at or above 2^32 elements, so huge
  // buffers are compared in bounded slices.
  constexpr size_t sliceElements = 1ull << 30;
  unsigned long long mismatches = 0;
  for (size_t offset = 0; offset < count; offset += sliceElements) {
    const auto sliceCount = std::min(sliceElements, count - offset);
    mismatches += detail::matxMismatchesSlice(actual + offset, reference + offset,
      sliceCount, stream);
  }
  return mismatches;
}

inline unsigned long long matxByteMismatches(const void* actual,
  const void* reference, const size_t count, cudaStream_t stream) {
  return matxMismatches(static_cast<const unsigned char*>(actual),
    static_cast<const unsigned char*>(reference), count, stream);
}

} // namespace bench

#endif
