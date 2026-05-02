//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_COLLECTIVE_CUH
#define SUTURE_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"
#include "regime.cuh"
#include "sync.cuh"

namespace suture {
  // super block put
  template<typename SutureAtom, typename BT = int>
  __device__ __forceinline__
  static void superPut(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_integral_v<BT> || cuda::std::is_same_v<cuda::fast_mod_div<long int>, BT>);
    // assert(bytes % SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES)
    const long int scaledChunkSize = bytes / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
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

  template<typename SutureAtom, typename Element, typename BT>
  __device__ __forceinline__
  static void reduceColl(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const SutureContext& ctx,
    const BT& blocks,
    const int& bIdx, const bool& isSrcSpread = false) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>>);
    // Assumptions
    // assert(blocks <= suture::MAX_NUM_CTAS);
    // assert(ctx.world > 1)
    const auto epoch = ctx.epochs[bIdx];
    const auto nextEpoch = epoch + static_cast<uint64_t>(1);
    const auto senseBit = static_cast<uint>(epoch % 2);

    if (bytes < suture::AR_LATENCY_BOUND_THRESHOLD) {
      const int superBlockIdx = bIdx / ctx.superBlockSize;
      const int intraIdx = bIdx % ctx.superBlockSize;
      const auto peer = (superBlockIdx + ctx.rank + 1) % ctx.world;
      cuda::std::byte* __restrict__ dstP = nullptr;
      const cuda::std::byte* __restrict__ srcPut = nullptr;
      size_t bytesPut = 0, bytesRed = 0;
      const cuda::std::byte* __restrict__ srcRed = nullptr;
      cuda::std::byte* __restrict__ stagingPut = nullptr;
      cuda::std::byte* __restrict__ stagingRed = nullptr;
      constexpr auto dAB = sizeof(LRP16::RT); // data alignment bytes
      const long int scaledChunkSize = bytes / dAB;
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
        auto* __restrict__ staging = ctx.staging[peer] + stagingOffset;

        bytesPut = ctaChunk * dAB;
        srcPut = src + startOffset + (isSrcSpread ? bytes * peer : 0);
        const auto rankOffset = (ctx.rank * suture::PACKET_BUFFER_SIZE);
        stagingPut = staging + rankOffset;
      }
      else {
        const auto bIdxR = bIdx - ctx.maxPutBlocks;
        // reduction offsets
        const auto ctaBaseRedChunk = scaledChunkSize / ctx.superBlockSize;
        const auto ctaRedResidue = static_cast<int>(scaledChunkSize % ctx.superBlockSize);
        const auto ctaRedChunk = ctaBaseRedChunk + (bIdxR < ctaRedResidue);
        const auto redOffsetElems = ctaBaseRedChunk * bIdxR + min(bIdxR, ctaRedResidue);
        const auto redStartOffset = redOffsetElems * dAB;

        bytesRed = ctaRedChunk * dAB;
        srcRed = src + redStartOffset;
        dstP = dst + redStartOffset;
        stagingRed = ctx.staging[ctx.rank] + (stagingPrefix + (redOffsetElems * pAB));
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
      auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(typedWorkspace) + SutureAtom::PIPELINE_BYTES;
      for (int i = 0; i < ctx.world; i += Config::THREADS) {
        staging[i] = ctx.stagingTR[i];
      }
      size_t bytesRed = 0;
      cuda::std::byte* __restrict__ dstP = nullptr;
      const cuda::std::byte* __restrict__ srcP = nullptr;

      const long int scaledChunkSize = bytes / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      // reduction offsets
      const auto ctaBaseRedChunk = scaledChunkSize / blocks;
      const auto ctaRedResidue = static_cast<int>(scaledChunkSize % blocks);
      const auto ctaRedChunk = ctaBaseRedChunk + (bIdx < ctaRedResidue);
      const auto redOffsetElems = ctaBaseRedChunk * bIdx + min(bIdx, ctaRedResidue);
      const auto redStartOffset = redOffsetElems * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      bytesRed = ctaRedChunk * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      srcP = src + redStartOffset;
      dstP = dst + redStartOffset;

      // throughput regime
      const ReduceTRArgs redArgs{
        .staging = staging,
        .signals = ctx.signals[ctx.rank],
        .putSignals = ctx.signals,
        .dst = dstP,
        .src = srcP,
        .flag = nextEpoch,
        .totalBytes = bytes,
        .bytesRed = bytesRed,
        .rank = ctx.rank,
        .world = ctx.world, // <- TODO: check SASS that no constructor instructions are emitted for this subobject
        .transferBlock = isPutBlock
      };
      SutureAtom::reduce2(redArgs, typedWorkspace);
    }
    __syncthreads();
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = nextEpoch;
    }
    {
      const auto leftover = suture::MAX_NUM_CTAS - blocks;
      auto* __restrict__ epochs = ctx.epochs + blocks;
      const auto tid = bIdx * SutureAtom::Config::THREADS + threadIdx.x;
      for (int i = tid; i < leftover; i += (SutureAtom::Config::THREADS * blocks)) {
        epochs[i] = nextEpoch;
      }
    }
  }

  template<typename SutureAtom>
  __device__ __forceinline__
  static void allGather(cuda::std::byte** __restrict__ const& dst,
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
    auto* __restrict__ dstP = dst[peer] + ctx.rank * bytes;
    const auto localOffset = static_cast<uint>(peer * ctx.maxSuperBlockSize + intraIdx);
    const auto remoteOffset = static_cast<uint>(ctx.rank * ctx.maxSuperBlockSize + intraIdx);
    auto* __restrict__ localSync = ctx.sync[ctx.rank];
    auto* __restrict__ remoteSync = ctx.sync[peer] + remoteOffset;
    // syncRelaxed
    syncRelaxed(remoteSync, localSync + localOffset, epoch + 1);
    superPut<SutureAtom>(dstP, src, bytes, workspace, ctx.superBlockSize, intraIdx);
    // syncStrong
    syncStrong(remoteSync, localSync + localOffset, epoch + 2);
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

  template<typename SutureAtom, typename Element, typename BT = int>
  __device__ __forceinline__
  static void allReduce(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const SutureContext& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    reduceColl<SutureAtom>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx);
  }

  template<typename SutureAtom, typename Element, typename BT = int>
  __device__ __forceinline__
  static void reduceScatter(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const SutureContext& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    reduceColl<SutureAtom>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, true);
  }
}
#endif //SUTURE_COLLECTIVE_CUH