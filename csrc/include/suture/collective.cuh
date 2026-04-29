//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_COLLECTIVE_CUH
#define SUTURE_COLLECTIVE_CUH
#include <nvshmem.h> // nvshmem_ptr

#include "base.cuh"
#include "context.cuh"
#include "regime.cuh"
#include "sync.cuh"

namespace suture {
  // super block put
  template<typename SutureAtom>
  __device__ __forceinline__
  static void superPut(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const int& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    // assert(bytes % SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES)
    const size_t scaledChunkSize = bytes / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto ctaBaseChunk = scaledChunkSize / blocks;
    const auto chunkResidue = static_cast<int>(scaledChunkSize % blocks);
    const size_t ctaChunk = ctaBaseChunk + (bIdx < chunkResidue);
    const auto startOffset = (ctaBaseChunk * bIdx + min(bIdx, chunkResidue)) *
      SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto* __restrict__ srcP = src + startOffset;
    auto* __restrict__ dstP = dst + startOffset;
    const size_t bytesP = ctaChunk * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    SutureAtom::putAsync(dstP, srcP, bytesP, workspace);
  }

  template<typename SutureAtom, typename Element>
  __device__ __forceinline__
  static void reduceColl(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const SutureContext& ctx,
    const int& blocks,
    const int& bIdx, const bool& isSrcSpread = false) {
    // Assumptions
    // assert(blocks <= suture::MAX_NUM_CTAS);
    const auto isPutBlock = bIdx < ctx.maxPutBlocks;
    const int superBlockIdx = bIdx / ctx.superBlockSize;
    const int intraIdx = bIdx % ctx.superBlockSize;
    const auto peer = (superBlockIdx + ctx.rank + 1) % ctx.world;
    const auto myOffset = static_cast<uint>(peer * ctx.maxSuperBlockSize + intraIdx);
    const auto epoch = ctx.epochs[bIdx];
    const auto nextEpoch = epoch + static_cast<uint64_t>(1);
    const auto senseBit = static_cast<uint>(epoch % 2);

    if (bytes < suture::AR_LATENCY_BOUND_THRESHOLD) {
      cuda::std::byte* __restrict__ dstP = nullptr;
      const cuda::std::byte* __restrict__ srcPut = nullptr;
      size_t bytesPut = 0, bytesRed = 0;
      const cuda::std::byte* __restrict__ srcRed = nullptr;
      cuda::std::byte* __restrict__ stagingPut = nullptr;
      cuda::std::byte* __restrict__ stagingRed = nullptr;
      constexpr auto dAB = sizeof(LRP16::RT); // data alignment bytes
      const size_t scaledChunkSize = bytes / dAB;
      constexpr auto pAB = sizeof(LRP16::RT) * 2; // packet alignment bytes
      const auto stagingPrefix = (senseBit * ctx.world * suture::PACKET_BUFFER_SIZE);
      // latency regime
      if (isPutBlock) {
        // put offsets
        const auto ctaBaseChunk = scaledChunkSize / ctx.superBlockSize;
        const auto chunkResidue = static_cast<int>(scaledChunkSize % ctx.superBlockSize);
        const size_t ctaChunk = ctaBaseChunk + (intraIdx < chunkResidue);
        const auto offSetElems = ctaBaseChunk * intraIdx + min(intraIdx, chunkResidue);
        const auto startOffset = offSetElems * dAB;
        const auto stagingOffset = stagingPrefix + (offSetElems * pAB);
        const auto* __restrict__ staging = ctx.staging + stagingOffset;

        bytesPut = ctaChunk * dAB;
        srcPut = src + startOffset + (isSrcSpread ? bytes * peer : 0);
        const auto rankOffset = (ctx.rank * suture::PACKET_BUFFER_SIZE);
        stagingPut = static_cast<cuda::std::byte*>(nvshmem_ptr(staging + rankOffset, peer));
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
      }

      const ReduceLRArgs redArgs{
        .dst = dstP,
        .srcPut = srcPut,
        .srcRed = srcRed,
        .stagingPut = stagingPut,
        .stagingRed = stagingRed,
        .flag = nextEpoch,
        .bytesPut = bytesPut,
        .bytesRed = bytesRed,
        .rank = ctx.rank,
        .putBlock = isPutBlock,
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
      uint64_t* __restrict__ signals = nullptr;

      const size_t scaledChunkSize = bytes / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      if (isPutBlock) {
        // transfer offsets
        const auto ctaBaseChunk = scaledChunkSize / ctx.superBlockSize;
        const auto chunkResidue = static_cast<int>(scaledChunkSize % ctx.superBlockSize);
        const size_t ctaChunk = ctaBaseChunk + (intraIdx < chunkResidue);
        const auto offSetElems = ctaBaseChunk * intraIdx + min(intraIdx, chunkResidue);
        const auto startOffset = offSetElems * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        srcPut = src + startOffset + (isSrcSpread ? bytes * peer : 0);
        redPut = static_cast<cuda::std::byte*>(nvshmem_ptr(ctx.reduceBuffer + (ctx.rank * bytes + startOffset)));
        bytesPut = ctaChunk * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      }
      {
        // reduction offsets
        const auto ctaBaseRedChunk = scaledChunkSize / blocks;
        const auto ctaRedResidue = static_cast<int>(scaledChunkSize % blocks);
        const auto ctaRedChunk = ctaBaseRedChunk + (bIdx < ctaRedResidue);
        const auto redOffsetElems = ctaBaseRedChunk * bIdx + min(bIdx, blocks);
        const auto redStartOffset = redOffsetElems * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        bytesRed = ctaRedChunk * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        srcRed = ctx.reduceBuffer + redStartOffset;
        srcP = src + redStartOffset;
        dstP = dst + redStartOffset;
      }

      // throughput regime
      signals = static_cast<uint64_t*>(nvshmem_ptr(ctx.signals + ctx.rank, peer));
      auto* __restrict__ remoteSync = static_cast<uint64_t*>(nvshmem_ptr(ctx.sync + syncRemoteOffset, peer));

      const ReduceTRArgs redArgs{
        .signals = ctx.signals,
        .putSignals = signals,
        .dst = dstP,
        .srcPut = srcPut,
        .redPut = redPut,
        .srcRed = srcRed,
        .src = srcP,
        .remoteSync = remoteSync,
        .localSync = ctx.sync + myOffset,
        .sigCounter = ctx.sigCounter + peer,
        .flag = nextEpoch,
        .totalBytes = bytes,
        .bytesPut = bytesPut,
        .bytesRed = bytesRed,
        .rank = ctx.rank,
        .world = ctx.world, // <- TODO: check SASS that no constructor instructions are emitted for this subobject
        .numBlocks = blocks,
        .bIdx = bIdx,
        .superBlockSize = ctx.superBlockSize,
        .putBlock = isPutBlock
      };
      SutureAtom::reduce(redArgs, typedWorkspace);
    }
    __syncthreads();
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = nextEpoch;
    }
    {
      const auto leftover = suture::MAX_NUM_CTAS - blocks;
      auto* __restrict__ epochs = ctx.epochs + blocks;
      const auto tid = bIdx * SutureAtom::Config::THREADS + threadIdx.x;
      for (int i = tid; i < leftover; i += SutureAtom::Config::THREADS) {
        epochs[i] = nextEpoch;
      }
    }
  }

  template<typename SutureAtom>
  __device__ __forceinline__
  static void allGather(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const SutureContext& ctx,
    const int& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    if (bIdx >= ctx.maxPutBlocks) {
      return;
    }
    const auto epoch = ctx.epochs[bIdx];
    const int superBlockIdx = bIdx / ctx.superBlockSize;
    const int intraIdx = bIdx % ctx.superBlockSize;
    const auto peer = (superBlockIdx + ctx.rank + 1) % ctx.world;
    auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(dst + ctx.rank * bytes, peer));
    const auto localOffset = static_cast<uint>(peer * ctx.maxSuperBlockSize + intraIdx);
    const auto remoteOffset = static_cast<uint>(ctx.rank * ctx.maxSuperBlockSize + intraIdx);
    auto* __restrict__ remoteSync = static_cast<uint64_t*>(nvshmem_ptr(ctx.sync + remoteOffset, peer));
    // syncRelaxed
    syncRelaxed(remoteSync, ctx.sync + localOffset, epoch + 1);
    superPut<SutureAtom>(dstP, src, bytes, workspace, ctx.superBlockSize, intraIdx);
    // syncStrong
    syncStrong(remoteSync, ctx.sync + localOffset, epoch + 2);
    // update epoch
    {
      const auto leftover = suture::MAX_NUM_CTAS - blocks;
      auto* __restrict__ epochs = ctx.epochs + blocks;
      const auto nextEpoch = epoch + 2;
      const auto tid = bIdx * SutureAtom::Config::THREADS + threadIdx.x;
      for (int i = tid; i < leftover; i += (SutureAtom::Config::THREADS * blocks)) {
        epochs[i] = nextEpoch;
      }
    }
  }

  template<typename SutureAtom, typename Element>
  __device__ __forceinline__
  static void allReduce(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const SutureContext& ctx,
    const int& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    reduceColl<SutureAtom>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx);
  }

  template<typename SutureAtom, typename Element>
  __device__ __forceinline__
  static void reduceScatter(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const SutureContext& ctx,
    const int& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    reduceColl<SutureAtom>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, true);
  }
}
#endif //SUTURE_COLLECTIVE_CUH