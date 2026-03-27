//
// Created by azureuser on 3/25/26.
//

#ifndef TACK_AR_CUH
#define TACK_AR_CUH
#include <cuda/cmath>
#include <cuda/utility>
#include <cutlass/array.h>
#include <cute/arch/copy_sm80.hpp>
#include <cuda/std/cstddef>

#include <nvshmem.h>
#include "constants.cuh"
#include "reduce.cuh"
#include "rvt.cuh"
#include "sync.cuh"

struct __align__(16) ARArgs {
  cuda::std::byte* const src = nullptr;
  cuda::std::byte* const dst = nullptr;
  uint64_t* const completions = nullptr; // [world, maxSuperBlockSize], symmetric
  uint64_t* const arrivals = nullptr; // [world, maxSuperBlockSize], symmetric
  uint64_t* const senseBits = nullptr; // [world, maxSuperBlockSize], local
  const size_t ctaBaseChunk = 0;
  const cuda::fast_mod_div<int> superBlockSize_v;
  const cuda::fast_mod_div<int> world_v;
  const int chunkResidue = 0;
  const int maxSuperBlockSize = 1;
  const int rank = 0;
  const int world = 1;
};

__launch_bounds__(tack::threads, 1)
__global__ void allReduce(const __grid_constant__ ARArgs args) {
  static_assert(tack::threads > tack::WARP_SIZE && tack::threads % tack::WARP_SIZE == 0);
  extern __shared__ __align__(tack::RED_MAX_ALIGNMENT) cuda::std::byte workspace[];
  const int superBlockIdx = static_cast<int>(blockIdx.x) / args.superBlockSize_v;
  const int intraIdx = static_cast<int>(blockIdx.x) % args.superBlockSize_v;
  const auto peer = (superBlockIdx + args.rank + 1) % args.world_v;
  const auto myOffset = peer * args.maxSuperBlockSize + intraIdx;
  auto* __restrict__ senseBits = args.senseBits + myOffset;
  const auto senseBit = *senseBits;

  // compute buffer offset
  const auto startOffset = (args.ctaBaseChunk * intraIdx + min(intraIdx, args.chunkResidue)) * tack::RED_ALIGNMENT;
  const auto* __restrict__ srcP = args.src + startOffset;
  auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(args.dst + startOffset, peer));
  // total number of aligned elements
  const size_t ctaChunk = args.ctaBaseChunk + (intraIdx < args.chunkResidue);
  const size_t bytes = ctaChunk * tack::RED_ALIGNMENT;

  const auto peerOffset = args.rank * args.maxSuperBlockSize + intraIdx;
  auto* __restrict__ peerMailbox = static_cast<uint64_t*>(nvshmem_ptr(args.arrivals + peerOffset, peer));
  auto* __restrict__ myMailbox = args.arrivals + myOffset;
  const auto payload = senseBit == 0 ? 1 : 0;
  auto* __restrict__ peerMailbox1 = static_cast<uint64_t*>(nvshmem_ptr(args.completions + peerOffset, peer));
  auto* __restrict__ myMailbox1 = args.completions + myOffset;

  constexpr tack::Reduce<700, ARCH> reduce{};
  tack::arrive(peerMailbox, myMailbox, payload);
  reduce(srcP, dstP, bytes);
  tack::wait(peerMailbox1, myMailbox1, payload, senseBits);
}
#endif //TACK_AR_CUH