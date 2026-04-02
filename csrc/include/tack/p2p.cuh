//
// Created by azureuser on 3/9/26.
//

#ifndef TACK_P2P_CUH
#define TACK_P2P_CUH
#include <nvshmem.h>
#include "put.cuh"

struct P2PArgs {
  cuda::std::byte* const srcBuf = nullptr;
  cuda::std::byte* const dstBuf = nullptr;
  const size_t ctaBaseChunk = 0;
  const uint chunkResidue = 0;
  const int rank = 0;
  const int peer = 0;
};

__global__ __launch_bounds__(tack::threads)
void p2pK(const __grid_constant__ P2PArgs args) {
  extern __shared__ __align__(tack::Alignment) cuda::std::byte workspace[];
  const auto bIdx = blockIdx.x;

  const size_t ctaChunk = args.ctaBaseChunk + (bIdx < args.chunkResidue);
  const auto startOffset = (args.ctaBaseChunk * bIdx + min(bIdx, args.chunkResidue)) * tack::MAX_ACCESS_ALIGNMENT;
  const auto* __restrict__ srcP = args.srcBuf + startOffset;
  auto* __restrict__ dstP = args.dstBuf + startOffset;
  const size_t bytes = ctaChunk * tack::MAX_ACCESS_ALIGNMENT;
  constexpr tack::Put<ARCH> put{};
  put(dstP, srcP, workspace, bytes);
  //nvshmemx_putmem_nbi_block(dstP, srcP, bytes, args.peer);
}
#endif //TACK_P2P_CUH