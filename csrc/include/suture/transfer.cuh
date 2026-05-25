//
// Created by osayamen on 5/25/26.
//

#ifndef SUTURE_TRANSFER_CUH
#define SUTURE_TRANSFER_CUH
namespace suture {
  // super block put
  template<typename SutureAtom, TransferType pt = TransferType::asynchronous, typename BT = int>
  __device__ __forceinline__
  static void superPut(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_integral_v<BT> || cuda::std::is_same_v<cuda::fast_mod_div<long int>, BT>);
    // assert(bytes % SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES)
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto [bytesP, startOffset] = partition<alignmentBytes>(bytes, blocks, bIdx);
    const auto* __restrict__ srcP = src + startOffset;
    auto* __restrict__ dstP = dst + startOffset;
    if constexpr (pt == TransferType::asynchronous) {
      SutureAtom::putAsync(dstP, srcP, bytesP, workspace);
    }
    else {
      SutureAtom::put(dstP, srcP, bytesP, workspace);
    }
  }
  // super block put
  template<typename SutureAtom, TransferType pt = TransferType::asynchronous, typename BT = int>
  __device__ __forceinline__
  static void superGet(cuda::std::byte* __restrict__ const& dst, // local
    const cuda::std::byte* __restrict__ const& src, // remote
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    superPut<SutureAtom, pt>(dst, src, bytes, workspace, blocks, bIdx);
  }

  template<typename SutureAtom, size_t bytes, TransferType pt = TransferType::asynchronous, typename BT = int>
  __device__ __forceinline__
  static void superGet(cuda::std::byte* __restrict__ const& dst, // local
    const cuda::std::byte* __restrict__ const& src, // remote
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    superPut<SutureAtom, pt>(dst, src, bytes, workspace, blocks, bIdx);
  }
}
#endif //SUTURE_TRANSFER_CUH
