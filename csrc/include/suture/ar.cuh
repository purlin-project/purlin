//
// Created by azureuser on 3/25/26.
//

#ifndef SUTURE_AR_CUH
#define SUTURE_AR_CUH
#include <cuda/cmath>
#include <cuda/std/cstddef>

#include <nvshmem.h>
#include "constants.cuh"
#include "reduce.cuh"
#include "rvt.cuh"
#include "sync.cuh"

struct __align__(16) ARArgs {
  const cuda::std::byte* const src = nullptr; // [size], non-symmetric
  cuda::std::byte* const dst = nullptr; // [size], symmetric
  uint64_t* const completions = nullptr; // [world, maxSuperBlockSize], symmetric
  uint64_t* const arrivals = nullptr; // [world, maxSuperBlockSize], symmetric
  uint8_t* const senseBitsTR = nullptr; // [world, maxSuperBlockSize], non-symmetric
  uint8_t* const senseBitsLR = nullptr; // [world, maxSuperBlockSize], non-symmetric
  cuda::std::byte* const staging = nullptr; // [2, world, LAT_THRESHOLD], symmetric,
  uint8_t* const flagSense = nullptr; // [2, world, LAT_THRESHOLD / (sizeof(LRP.data)]
  const size_t ctaBaseChunk = 0;
  const cuda::fast_mod_div<int> superBlockSize_v;
  const cuda::fast_mod_div<int> world_v;
  const int chunkResidue = 0;
  const int maxSuperBlockSize = 1;
  const int rank = 0;
  const int world = 1;
};

// throughput-bound regime
__launch_bounds__(suture::threads, 1)
__global__ void allReduceTR(const __grid_constant__ ARArgs args) {
  static_assert(suture::kThreads > suture::WARP_SIZE && suture::kThreads % suture::WARP_SIZE == 0);
  extern __shared__ __align__(suture::RED_MAX_ALIGNMENT) cuda::std::byte workspace[];
  const int superBlockIdx = static_cast<int>(blockIdx.x) / args.superBlockSize_v;
  const int intraIdx = static_cast<int>(blockIdx.x) % args.superBlockSize_v;
  const auto peer = (superBlockIdx + args.rank + 1) % args.world_v;
  const auto myOffset = peer * args.maxSuperBlockSize + intraIdx;
  auto* __restrict__ senseBits = args.senseBitsTR + myOffset;
  const auto senseBit = static_cast<uint64_t>(*senseBits);

  // compute buffer offset
  const auto startOffset = (args.ctaBaseChunk * intraIdx + min(intraIdx, args.chunkResidue)) * suture::TR_RED_ALIGNMENT;
  const auto* __restrict__ srcP = args.src + startOffset;
  auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(args.dst + startOffset, peer));
  // total number of aligned elements
  const size_t ctaChunk = args.ctaBaseChunk + (intraIdx < args.chunkResidue);
  const size_t bytes = ctaChunk * suture::TR_RED_ALIGNMENT;

  const auto peerOffset = args.rank * args.maxSuperBlockSize + intraIdx;
  auto* __restrict__ peerMailbox = static_cast<uint64_t*>(nvshmem_ptr(args.arrivals + peerOffset, peer));
  auto* __restrict__ myMailbox = args.arrivals + myOffset;
  const auto payload = senseBit == 0 ? 1 : 0;
  auto* __restrict__ peerMailbox1 = static_cast<uint64_t*>(nvshmem_ptr(args.completions + peerOffset, peer));
  auto* __restrict__ myMailbox1 = args.completions + myOffset;

  constexpr suture::AtomicReduce<suture::Regime::throughput, ARCH, ARCH> reduce{};
  suture::arrive(peerMailbox, myMailbox, payload);
  reduce(srcP, dstP, workspace, bytes);
  suture::wait(peerMailbox1, myMailbox1, payload, senseBits);
}

// latency-bound regime
__launch_bounds__(suture::threads, 1)
__global__ void allReduceLR(const __grid_constant__ ARArgs args) {
  const int superBlockIdx = static_cast<int>(blockIdx.x) / args.superBlockSize_v;
  const int intraIdx = static_cast<int>(blockIdx.x) % args.superBlockSize_v;
  const auto peer = (superBlockIdx + args.rank + 1) % args.world_v;
  const auto myOffset = peer * args.maxSuperBlockSize + intraIdx;
  auto* __restrict__ senseBits = args.senseBitsLR + myOffset;
  const auto senseBit = *senseBits;
#if defined(__CUDA_ARCH__)
  __builtin_assume(senseBit == 0 || senseBit == 1);
#endif
  const auto currentSense = senseBit == 0 ? 1 : 0;

  const auto offSetElems = args.ctaBaseChunk * intraIdx + min(intraIdx, args.chunkResidue);
  const auto startOffset = offSetElems * suture::LR_RED_ALIGNMENT;
  const auto* __restrict__ srcP = args.src + startOffset;
  auto* __restrict__ dstP = args.dst + startOffset;
  // Use double-buffering to obviate prologue synchronization
  const auto stagingOffset = (senseBit * args.world * suture::PACKET_BUFFER_SIZE) + (offSetElems * suture::LR_PACKET_ALIGNMENT);
  auto* __restrict__ staging = args.staging + stagingOffset;
  auto* __restrict__ rStaging = static_cast<cuda::std::byte*>(nvshmem_ptr(staging + (args.rank * suture::PACKET_BUFFER_SIZE), peer));
  auto* __restrict__ lStaging = staging + peer * suture::PACKET_BUFFER_SIZE;
  static_assert(suture::AR_LATENCY_BOUND_THRESHOLD % suture::LR_RED_ALIGNMENT == 0);
  const auto flagOffset = (((senseBit * args.world + peer) * suture::AR_LATENCY_BOUND_THRESHOLD) / suture::LR_RED_ALIGNMENT) + offSetElems;
  auto* __restrict__ flags = args.flagSense + flagOffset;
  const size_t ctaChunk = args.ctaBaseChunk + (intraIdx < args.chunkResidue);
  const size_t bytes = ctaChunk * suture::LR_RED_ALIGNMENT;

  constexpr suture::AtomicReduce<suture::Regime::latency, ARCH, ARCH> reduce{};
  reduce(srcP, rStaging, lStaging, dstP, flags, bytes);
  __syncthreads();
  if (!threadIdx.x) {
    *senseBits = currentSense;
  }
}
#endif //SUTURE_AR_CUH