//
// Created by osayamen on 5/26/26.
//

#ifndef SUTURE_HOST_CUH
#define SUTURE_HOST_CUH
namespace suture {
  __host__ __forceinline__
  void all2all(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes,
    const Context& ctx, cudaStream_t stream) {

  }
  __host__ __forceinline__
  void allGather(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes,
    const Context& ctx, cudaStream_t stream) {

  }

  template<typename Element>
  __host__ __forceinline__
  void allReduce(const Element* __restrict__ const& src,
    Element* __restrict__ const& dst,
    const size_t& bytes,
    const Context& ctx, cudaStream_t stream) {

  }

  template<typename Element>
  __host__ __forceinline__
  void reduceScatter(const Element* __restrict__ const& src,
    Element* __restrict__ const& dst,
    const size_t& bytes,
    const Context& ctx, cudaStream_t stream) {

  }

  // P2P transfer
  __host__ __forceinline__
  void copyAsync(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes, cudaStream_t stream) {

  }
}
#endif //SUTURE_HOST_CUH
