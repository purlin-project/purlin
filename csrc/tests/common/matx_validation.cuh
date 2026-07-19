#ifndef PURLIN_TESTS_COMMON_MATX_VALIDATION_CUH
#define PURLIN_TESTS_COMMON_MATX_VALIDATION_CUH

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

template<typename Element>
inline unsigned long long matxMismatches(const Element* actual,
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
  (matches = matx::sum(matx::isclose(actualTensor, referenceTensor, 0, 0))).run(executor);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return static_cast<unsigned long long>(count) -
    static_cast<unsigned long long>(matches());
}

inline unsigned long long matxByteMismatches(const void* actual,
  const void* reference, const size_t count, cudaStream_t stream) {
  return matxMismatches(static_cast<const unsigned char*>(actual),
    static_cast<const unsigned char*>(reference), count, stream);
}

} // namespace bench

#endif
