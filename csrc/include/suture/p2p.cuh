//
// Created by azureuser on 3/9/26.
//

#ifndef SUTURE_P2P_CUH
#define SUTURE_P2P_CUH
#include <nvshmem.h>

#include "constants.cuh"
#include "suture.cuh"

struct P2PArgs {
  cuda::std::byte* const srcBuf = nullptr;
  cuda::std::byte* const dstBuf = nullptr;
  const size_t ctaBaseChunk = 0;
  const uint chunkResidue = 0;
  const int rank = 0;
  const int peer = 0;
};

template<
  int threads,
  int pipeStages,
  int stageExtent,
  int unrollFactor
>
__global__ __launch_bounds__(threads)
void p2pK(const __grid_constant__ P2PArgs args) {
  using SutureAtom = suture::Atom<ARCH>;
  constexpr auto AlignmentBytes = SutureAtom::MaxAlignmentBytes::value;
  extern __shared__ __align__(AlignmentBytes) cuda::std::byte workspace[];
  const auto bIdx = blockIdx.x;

  const size_t ctaChunk = args.ctaBaseChunk + (bIdx < args.chunkResidue);
  const auto startOffset = (args.ctaBaseChunk * bIdx + min(bIdx, args.chunkResidue)) * AlignmentBytes;
  const auto* __restrict__ srcP = args.srcBuf + startOffset;
  auto* __restrict__ dstP = args.dstBuf + startOffset;
  const size_t bytes = ctaChunk * AlignmentBytes;
  SutureAtom::put<threads, pipeStages, stageExtent, unrollFactor>(dstP, srcP, workspace, bytes);
}
#endif //SUTURE_P2P_CUH