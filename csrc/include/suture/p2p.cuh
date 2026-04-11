//
// Created by azureuser on 3/9/26.
//

#ifndef SUTURE_P2P_CUH
#define SUTURE_P2P_CUH
#include "tendon.cuh"
struct P2PArgs {
  cuda::std::byte* const srcBuf = nullptr;
  cuda::std::byte* const dstBuf = nullptr;
  const size_t ctaBaseChunk = 0;
  const uint chunkResidue = 0;
  const int rank = 0;
  const int peer = 0;
};

template<typename SutureAtom>
__global__ __launch_bounds__(SutureAtom::Config::THREADS)
void p2pK(const __grid_constant__ P2PArgs args) {
  constexpr auto alignmentBytes = SutureAtom::Config::ALIGNMENT_BYTES;
  extern __shared__ __align__(alignmentBytes) cuda::std::byte workspace[];
  const auto bIdx = blockIdx.x;

  const size_t ctaChunk = args.ctaBaseChunk + (bIdx < args.chunkResidue);
  const auto startOffset = (args.ctaBaseChunk * bIdx + min(bIdx, args.chunkResidue)) * alignmentBytes;
  const auto* __restrict__ srcP = args.srcBuf + startOffset;
  auto* __restrict__ dstP = args.dstBuf + startOffset;
  const size_t bytes = ctaChunk * alignmentBytes;
  SutureAtom::putAsync(dstP, srcP, bytes, workspace);
}
#endif //SUTURE_P2P_CUH