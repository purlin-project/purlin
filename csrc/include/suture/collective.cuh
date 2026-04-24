//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_COLLECTIVE_CUH
#define SUTURE_COLLECTIVE_CUH
#include <nvshmem.h> // nvshmem_ptr

#include "base.cuh"
#include "context.cuh"
#include "regime.cuh"

namespace suture {
  // super block put
  template<typename SutureAtom>
  __device__ __forceinline__
  static void superPut(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const int& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    // assert(bytes % SutureAtom::Config::ALIGNMENT_BYTES)
    const size_t scaledChunkSize = bytes / SutureAtom::Config::ALIGNMENT_BYTES;
    const auto ctaBaseChunk = scaledChunkSize / blocks;
    const auto chunkResidue = static_cast<int>(scaledChunkSize % blocks);
    const size_t ctaChunk = ctaBaseChunk + (bIdx < chunkResidue);
    const auto startOffset = (ctaBaseChunk * bIdx + min(bIdx, chunkResidue)) * SutureAtom::Config::ALIGNMENT_BYTES;
    const auto* __restrict__ srcP = src + startOffset;
    auto* __restrict__ dstP = dst + startOffset;
    const size_t bytesP = ctaChunk * SutureAtom::Config::ALIGNMENT_BYTES;
    SutureAtom::putAsync(dstP, srcP, bytesP, workspace);
  }

  template<typename SutureAtom, typename Element>
  __device__ __forceinline__
  static void allReduce(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    Element* __restrict__ const& typedWorkspace, // shared
    const size_t& bytes,
    const SutureContext& ctx,
    const int& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    const int superBlockIdx = bIdx / ctx.superBlockSize;
    const int intraIdx = bIdx % ctx.superBlockSize;
    const auto peer = (superBlockIdx + ctx.rank + 1) % ctx.world;
    const auto myOffset = static_cast<uint>(peer * ctx.maxSuperBlockSize + intraIdx);
    auto* __restrict__ senseBits = ctx.senseBitsLR + myOffset;
    const auto senseBit = *senseBits;
#if defined(__CUDA_ARCH__)
    __builtin_assume(senseBit == static_cast<uint8_t>(0) || senseBit == static_cast<uint8_t>(1));
#endif

    const auto currentSense = senseBit == 0 ? 1 : 0;
    if (bytes < suture::AR_LATENCY_BOUND_THRESHOLD) {
      cuda::std::byte* __restrict__ dstP = nullptr;
      const cuda::std::byte* __restrict__ srcPut = nullptr;
      size_t bytesPut = 0, bytesRed = 0;
      const cuda::std::byte* __restrict__ srcRed = nullptr;
      cuda::std::byte* __restrict__ stagingPut = nullptr;
      cuda::std::byte* __restrict__ stagingRed = nullptr;
      uint8_t* __restrict__ flagsPut = nullptr;
      uint8_t* __restrict__ flagsRed = nullptr;
      constexpr auto dAB = sizeof(LRP16::RT); // data alignment bytes
      const size_t scaledChunkSize = bytes / dAB;
      constexpr auto pAB = sizeof(LRP16::RT) * 2; // packet alignment bytes
      const auto stagingPrefix = (senseBit * ctx.world * suture::PACKET_BUFFER_SIZE);
      // latency regime
      {
        // put offsets
        const auto ctaBaseChunk = scaledChunkSize / ctx.superBlockSize;
        const auto chunkResidue = static_cast<int>(scaledChunkSize % ctx.superBlockSize);
        const size_t ctaChunk = ctaBaseChunk + (intraIdx < chunkResidue);
        const auto offSetElems = ctaBaseChunk * intraIdx + min(intraIdx, chunkResidue);
        const auto startOffset = offSetElems * dAB;
        const auto stagingOffset = stagingPrefix + (offSetElems * pAB);
        const auto* __restrict__ staging = ctx.staging + stagingOffset;
        const auto flagOffset = (senseBit * ctx.world + peer) * suture::FLAG_BUFFER_SIZE + offSetElems;

        bytesPut = ctaChunk * dAB;
        srcPut = src + startOffset;
        const auto rankOffset = (ctx.rank * suture::PACKET_BUFFER_SIZE);
        stagingPut = static_cast<cuda::std::byte*>(nvshmem_ptr(staging + rankOffset, peer));
        flagsPut = ctx.flagPutSense + flagOffset;
      }
      {
        // reduction offsets
        const auto ctaBaseRedChunk = scaledChunkSize / blocks;
        const auto ctaRedResidue = static_cast<int>(scaledChunkSize % blocks);
        const auto ctaRedChunk = ctaBaseRedChunk + (bIdx < ctaRedResidue);
        const auto redOffsetElems = ctaBaseRedChunk * bIdx + min(bIdx, blocks);
        const auto redStartOffset = redOffsetElems * dAB;

        bytesRed = ctaRedChunk * dAB;
        srcRed = src + redStartOffset;
        dstP = dst + redStartOffset;
        stagingRed = ctx.staging + (stagingPrefix + (redOffsetElems * pAB));
        flagsRed = ctx.flagRedSense + redOffsetElems;
      }

      const ReduceLRArgs redArgs{
        .dst = dstP,
        .srcPut = srcPut,
        .srcRed = srcRed,
        .stagingPut = stagingPut,
        .stagingRed = stagingRed,
        .flagsPut = flagsPut,
        .flagsRed = flagsRed,
        .bytesPut = bytesPut,
        .bytesRed = bytesRed,
        .rank = ctx.rank,
        .world = ctx.world, // <- TODO: check SASS that no constructor instructions are emitted for this subobject
      };
      SutureAtom::reduce(redArgs, typedWorkspace);
    }
    else {
      const auto syncRemoteOffset = static_cast<uint>(ctx.rank * ctx.maxSuperBlockSize + intraIdx);
      size_t bytesPut = 0;
      size_t bytesRed = 0;
      cuda::std::byte* __restrict__ dstP = nullptr;
      cuda::std::byte* __restrict__ srcP = nullptr;
      cuda::std::byte* __restrict__ srcPut = nullptr;
      cuda::std::byte* __restrict__ redPut = nullptr;
      cuda::std::byte* __restrict__ srcRed = nullptr;
      uint64_t* __restrict__ arrivals = nullptr;
      uint32_t* __restrict__ signals = nullptr;

      const size_t scaledChunkSize = bytes / SutureAtom::Config::ALIGNMENT_BYTES;
      {
        // transfer offsets
        const auto ctaBaseChunk = scaledChunkSize / ctx.superBlockSize;
        const auto chunkResidue = static_cast<int>(scaledChunkSize % ctx.superBlockSize);
        const size_t ctaChunk = ctaBaseChunk + (intraIdx < chunkResidue);
        const auto offSetElems = ctaBaseChunk * intraIdx + min(intraIdx, chunkResidue);
        const auto startOffset = offSetElems * SutureAtom::Config::ALIGNMENT_BYTES;
        srcPut = src + startOffset;
        redPut = static_cast<cuda::std::byte*>(nvshmem_ptr(ctx.reduceBuffer + (ctx.rank * bytes + startOffset)));
        bytesPut = ctaChunk * SutureAtom::Config::ALIGNMENT_BYTES;
      }
      {
        // reduction offsets
        const auto ctaBaseRedChunk = scaledChunkSize / blocks;
        const auto ctaRedResidue = static_cast<int>(scaledChunkSize % blocks);
        const auto ctaRedChunk = ctaBaseRedChunk + (bIdx < ctaRedResidue);
        const auto redOffsetElems = ctaBaseRedChunk * bIdx + min(bIdx, blocks);
        const auto redStartOffset = redOffsetElems * SutureAtom::Config::ALIGNMENT_BYTES;
        bytesRed = ctaRedChunk * SutureAtom::Config::ALIGNMENT_BYTES;
        srcRed = ctx.reduceBuffer + redStartOffset;
        srcP = src + redStartOffset;
        dstP = dst + redStartOffset;
      }

      // throughput regime
      arrivals = ctx.sync0;
      signals = static_cast<uint32_t*>(nvshmem_ptr(ctx.signals, peer));

      const ReduceTRArgs redArgs{
        .signals = ctx.signals,
        .putSignals = signals,
        .dst = dstP,
        .srcPut = srcPut,
        .redPut = redPut,
        .srcRed = srcRed,
        .src = srcP,
        .arrivals = arrivals,
        .sigCounter = ctx.sigCounter,
        .totalBytes = bytes,
        .bytesPut = bytesPut,
        .bytesRed = bytesRed,
        .rank = ctx.rank,
        .world = ctx.world, // <- TODO: check SASS that no constructor instructions are emitted for this subobject
        .actualWorld = ctx.actualWorld,
        .numBlocks = blocks,
        .bIdx = bIdx,
        .senseBit = static_cast<uint>(currentSense),
        .syncRemoteOffset = syncRemoteOffset,
        .syncLocalOffset = myOffset,
        .superBlockSize = ctx.superBlockSize,
        .putBlock = (bIdx < (ctx.actualWorld * ctx.superBlockSize))
      };
      SutureAtom::reduce(redArgs, typedWorkspace);
    }
    __syncthreads();
    if (!threadIdx.x) {
      *senseBits = static_cast<uint8_t>(currentSense);
    }
  }
  template<typename SutureAtom>
  __device__ __forceinline__
  static void allGather(cuda::std::byte* __restrict__ const& dst, const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes, const SutureContext& ctx) {

  }
  template<typename SutureAtom, typename Element>
  __device__ __forceinline__
  static void reduceScatter(Element* __restrict__ const& dst, const Element* __restrict__ const& src,
    const size_t& bytes, const SutureContext& ctx) {

  }
}
#endif //SUTURE_COLLECTIVE_CUH