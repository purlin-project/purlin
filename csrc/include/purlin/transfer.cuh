//
// Created by osayamen on 5/25/26.
//

#ifndef PURLIN_TRANSFER_CUH
#define PURLIN_TRANSFER_CUH
namespace purlin {
  // super block copy
  template<typename PurlinAtom, typename BT = int>
  __device__ __forceinline__
  static void superCopy(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_integral_v<BT> || cuda::std::is_same_v<cuda::fast_mod_div<long int>, BT>);
    // assert(bytes % PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES)
    constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto [bytesP, startOffset] = partition<alignmentBytes>(bytes, blocks, bIdx);
    const auto* __restrict__ srcP = src + startOffset;
    auto* __restrict__ dstP = dst + startOffset;
    PurlinAtom::copy(dstP, srcP, bytesP, workspace);
  }
  template<typename PurlinAtom, int blocks>
  __device__ __forceinline__
  static void superCopy(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    // assert(bytes % PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES)
    constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto [bytesP, startOffset] = partition<blocks, alignmentBytes>(bytes, bIdx);
    const auto* __restrict__ srcP = src + startOffset;
    auto* __restrict__ dstP = dst + startOffset;
    PurlinAtom::copy(dstP, srcP, bytesP, workspace);
  }

  template<typename PurlinAtom, size_t bytes, typename BT = int>
  __device__ __forceinline__
  static void superCopy(cuda::std::byte* __restrict__ const& dst, // local
    const cuda::std::byte* __restrict__ const& src, // remote
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    superCopy<PurlinAtom>(dst, src, bytes, workspace, blocks, bIdx);
  }
}
#endif //PURLIN_TRANSFER_CUH
