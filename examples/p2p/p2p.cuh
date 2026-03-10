//
// Created by azureuser on 3/9/26.
//

#ifndef TACK_P2P_CUH
#define TACK_P2P_CUH
#include "../put.cuh"

#include <nvshmem.h>

struct P2PArgs {
  cuda::std::byte* const srcBuf = nullptr;
  cuda::std::byte* const dstBuf = nullptr;
  const size_t size = 0;
  const int rank = 0;
  const int peer = 0;
};

__global__ void p2pK(const __grid_constant__ P2PArgs args) {
  extern __shared__ __align__(Alignment) cuda::std::byte workspace[];
  const auto blocks = gridDim.x;
  const auto bIdx = blockIdx.x;

  const size_t scaledChunkSize = args.size / MAX_ACCESS_ALIGNMENT;
  const size_t ctaBaseChunk = scaledChunkSize / blocks;
  const int residue = static_cast<int>(scaledChunkSize % blocks);
  const size_t ctaChunk = ctaBaseChunk + (bIdx < residue);
  const auto startOffset = (ctaBaseChunk * bIdx + min(bIdx, residue)) * MAX_ACCESS_ALIGNMENT;
  const auto* __restrict__ srcP = args.srcBuf + startOffset;
  auto* __restrict__ dstP = args.dstBuf + startOffset;
  const size_t bytes = ctaChunk * MAX_ACCESS_ALIGNMENT;

  constexpr tack::Put<800> put{};
  put(dstP, srcP, workspace, bytes);
}
#endif //TACK_P2P_CUH