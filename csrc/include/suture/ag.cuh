//
// Created by osayamen on 2/23/26.
//

#ifndef SUTURE_AG_CUH
#define SUTURE_AG_CUH
#include <nvshmem.h>

#include "suture.cuh"
#include "constants.cuh"
#include "sync.cuh"

struct __align__(16) AGArgs {
  cuda::std::byte* src = nullptr; // [size], symmetric
  uint64_t* const completions = nullptr; // [world, maxSuperBlockSize], symmetric
  uint64_t* const arrivals = nullptr; // [world, maxSuperBlockSize], symmetric
  uint8_t* const senseBits = nullptr; // [world, maxSuperBlockSize], local
  const size_t ctaBaseChunk = 0;
  const cuda::fast_mod_div<int> superBlockSize_v;
  const cuda::fast_mod_div<int> world_v;
  const int chunkResidue = 0;
  const int maxSuperBlockSize = 1;
  const int rank = 0;
  const int world = 1;
};

template<typename SutureAtom>
__launch_bounds__(SutureAtom::THREADS, 1)
__global__ void allGather(const __grid_constant__ AGArgs args) {
  static_assert(SutureAtom::THREADS > suture::WARP_SIZE && SutureAtom::THREADS % suture::WARP_SIZE == 0);
  constexpr auto alignmentBytes = SutureAtom::Config::ALIGNMENT_BYTES;
  extern __shared__ __align__(128) cuda::std::byte workspace[];
  const int superBlockIdx = static_cast<int>(blockIdx.x) / args.superBlockSize_v;
  const int intraIdx = static_cast<int>(blockIdx.x) % args.superBlockSize_v;
  const auto peer = (superBlockIdx + args.rank + 1) % args.world_v;
  const auto myOffset = peer * args.maxSuperBlockSize + intraIdx;
  auto* __restrict__ senseBits = args.senseBits + myOffset;
  const auto senseBit = *senseBits;

  // compute buffer offset
  const auto startOffset = (args.ctaBaseChunk * intraIdx + min(intraIdx, args.chunkResidue)) * alignmentBytes;
  const auto* __restrict__ srcP = args.src + startOffset;
  auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(args.src + startOffset, peer));
  // total number of aligned elements
  const size_t ctaChunk = args.ctaBaseChunk + (intraIdx < args.chunkResidue);
  const size_t bytes = ctaChunk * alignmentBytes;

  const auto peerOffset = args.rank * args.maxSuperBlockSize + intraIdx;
  auto* __restrict__ peerMailbox = static_cast<uint64_t*>(nvshmem_ptr(args.arrivals + peerOffset, peer));
  auto* __restrict__ myMailbox = args.arrivals + myOffset;
  const auto payload = senseBit == 0 ? 1 : 0;
  auto* __restrict__ peerMailbox1 = static_cast<uint64_t*>(nvshmem_ptr(args.completions + peerOffset, peer));
  auto* __restrict__ myMailbox1 = args.completions + myOffset;

  suture::syncRelaxed(peerMailbox, myMailbox, payload);
  SutureAtom::putAsync(dstP, srcP, bytes, workspace);
  suture::syncStrong(peerMailbox1, myMailbox1, payload, senseBits);
}
#endif //SUTURE_AG_CUH
