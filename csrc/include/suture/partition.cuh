//
// Created by osayamen on 5/25/26.
//

#ifndef SUTURE_PARTITION_CUH
#define SUTURE_PARTITION_CUH
namespace suture {
  struct PartitionResult {
    const size_t bytes;
    const size_t startOffset;
  };
  struct PeerBlock {
    const int peer;
    const int intraIdx;
    const int blockSetSize;
  };
  __device__ __forceinline__
  static auto mapPeerBlock(const int bIdx, const int blockSetSize) {
    return PeerBlock{
      .peer = bIdx / blockSetSize,
      .intraIdx = bIdx % blockSetSize,
      .blockSetSize = blockSetSize,
    };
  }
  template<int AlignmentBytes, typename BT = int>
  __device__ __forceinline__
  constexpr auto partition(const size_t& bytes, const int& blocks, const int& bIdx) {
    static_assert(cuda::std::is_integral_v<BT> || cuda::std::is_same_v<cuda::fast_mod_div<long int>, BT>);
    const long int scaledChunkSize = bytes / AlignmentBytes;
    const auto ctaBaseChunk = static_cast<size_t>(scaledChunkSize / blocks);
    const auto ctaResidue = static_cast<int>(scaledChunkSize % blocks);
    const auto ctaChunk = ctaBaseChunk + (bIdx < ctaResidue);
    const auto offsetElems = ctaBaseChunk * bIdx + cute::min(bIdx, ctaResidue);
    const auto startOffset = offsetElems * AlignmentBytes;
    const size_t bytesSliced = static_cast<size_t>(ctaChunk) * AlignmentBytes;
    return PartitionResult{
      .bytes = bytesSliced,
      .startOffset = startOffset
    };
  }
  template<size_t bytes, int blocks, int AlignmentBytes>
  __device__ __forceinline__
  constexpr auto partition(const int& bIdx) {
    return partition<AlignmentBytes>(bytes, blocks, bIdx);
  }
  template<size_t bytes, int AlignmentBytes, typename BT = int>
  __device__ __forceinline__
  constexpr auto partition(const BT& blocks, const int& bIdx) {
    return partition<AlignmentBytes>(bytes, blocks, bIdx);
  }
  template<int blocks, int AlignmentBytes>
  __device__ __forceinline__
  constexpr auto partition(const size_t& bytes, const int& bIdx) {
    return partition<AlignmentBytes>(bytes, blocks, bIdx);
  }
}
#endif //SUTURE_PARTITION_CUH
