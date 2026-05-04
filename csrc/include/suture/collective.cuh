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
  static void reduce(
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
    const auto senseBit = static_cast<uint>(epoch % 2);

    if (bytes <= suture::RED_LATENCY_BOUND_THRESHOLD) {
      const auto nextEpoch = epoch + static_cast<uint64_t>(1);
      const auto isPutBlock = bIdx < ctx.maxPutBlocks;
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
      __syncthreads();
      if (!threadIdx.x) {
        ctx.epochs[bIdx] = nextEpoch;
      }
      const auto leftover = suture::MAX_NUM_CTAS - blocks;
      auto* __restrict__ epochs = ctx.epochs + blocks;
      const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
      for (int i = tid; i < leftover; i += (SutureAtom::THREADS * blocks)) {
        epochs[i] = nextEpoch;
      }
      return;
    }
    // throughput regime
    const auto stagingPrefix = MAX_ALL_REDUCE_SIZE_ * senseBit;
    const auto chunks = static_cast<int>(bytes / RED_CHUNK_SIZE);
    const auto cutoff = RED_CHUNK_SIZE * chunks;
    constexpr long int scaledChunkSize = RED_CHUNK_SIZE / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    if (bIdx < RED_PUT_BLOCKS) {
      auto flag = epoch;
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> sense{*ctx.groupSense};
      uint32_t localSense = sense.load(cuda::memory_order_relaxed);
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::PIPELINE_BYTES);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
        signals[i] = ctx.signals[i] + ctx.rank;
      }
      __syncthreads();
      constexpr auto ctaBasePutChunk = scaledChunkSize / RED_PUT_BLOCKS;
      const auto ctaPutResidue = static_cast<int>(scaledChunkSize % RED_PUT_BLOCKS);
      const auto ctaPutChunk = ctaBasePutChunk + (bIdx < ctaPutResidue);
      const auto putOffsetElems = ctaBasePutChunk * bIdx + min(bIdx, ctaPutResidue);
      const auto putStartOffset = putOffsetElems * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      const size_t bytesPut = static_cast<size_t>(ctaPutChunk) * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      const auto* __restrict__ srcP = src + putStartOffset;
      auto* __restrict__ dstBase = ctx.stagingTR[ctx.rank] + stagingPrefix;
      auto* __restrict__ dstP = dstBase + putStartOffset;
      for (int chunk = 0; chunk < chunks; ++chunk) {
        SutureAtom::putAsync(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        flag++;
        uint32_t nextSense = localSense ^ 1;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = RED_PUT_BLOCKS == 1 ? 1 : 0;
          if (!threadIdx.x) {
            // wait until groupSense matches localSense
            bool canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            while (!canProceed) {
              canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            }
            cuda::std::ignore = sense.load(cuda::memory_order_acquire);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == RED_PUT_BLOCKS;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
              sense.store(nextSense, cuda::memory_order_release);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            for (int i = static_cast<int>(threadIdx.x % WARP_SIZE); i < ctx.world; i += WARP_SIZE) {
              cuda::atomic_ref<uint64_t, cuda::thread_scope_system> signal{*signals[i]};
              signal.store(flag, cuda::std::memory_order_release);
            }
            __syncwarp();
          }
        }
        __syncthreads();
        localSense = nextSense;
        dstP += RED_CHUNK_SIZE;
        srcP += RED_CHUNK_SIZE;
      }
      if (bytes > cutoff) {
        const auto residue = bytes - cutoff;
        const long int scaledChunkSizeLeft = residue / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        const auto ctaBasePutChunkLeft = scaledChunkSizeLeft / RED_PUT_BLOCKS;
        const auto ctaPutResidueLeft = static_cast<int>(scaledChunkSizeLeft % RED_PUT_BLOCKS);
        const auto ctaPutChunkLeft = ctaBasePutChunkLeft + (bIdx < ctaPutResidueLeft);
        const auto putOffsetElemsLeft = ctaBasePutChunkLeft * bIdx + min(bIdx, ctaPutResidueLeft);
        const auto putStartOffsetLeft = putOffsetElemsLeft * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        const size_t bytesPutLeft = static_cast<size_t>(ctaPutChunkLeft) * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;

        srcP = src + (RED_CHUNK_SIZE * chunks + putStartOffsetLeft);
        dstP = dstBase + (RED_CHUNK_SIZE * chunks + putStartOffsetLeft);
        SutureAtom::putAsync(dstP, srcP, bytesPutLeft, workspace);
        __syncthreads();
        flag++;
        uint32_t nextSense = localSense ^ 1;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = RED_PUT_BLOCKS == 1 ? 1 : 0;
          if (!threadIdx.x) {
            // wait until groupSense matches localSense
            bool canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            while (!canProceed) {
              canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            }
            cuda::std::ignore = sense.load(cuda::memory_order_acquire);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == RED_PUT_BLOCKS;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
              sense.store(nextSense, cuda::memory_order_release);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            for (int i = static_cast<int>(threadIdx.x % WARP_SIZE); i < ctx.world; i += WARP_SIZE) {
              cuda::atomic_ref<uint64_t, cuda::thread_scope_system> signal{*signals[i]};
              signal.store(flag, cuda::std::memory_order_release);
            }
            __syncwarp();
          }
        }
        __syncthreads();
      }
      if (!threadIdx.x) {
        ctx.epochs[bIdx] = flag;
      }
      const auto leftover = suture::MAX_NUM_CTAS - blocks;
      auto* __restrict__ epochs = ctx.epochs + blocks;
      const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
      for (int i = tid; i < leftover; i += (SutureAtom::THREADS * RED_PUT_BLOCKS)) {
        epochs[i] = flag;
      }
      return;
    }
    const auto reduceBIdx = bIdx - RED_PUT_BLOCKS;
    const auto reduceBlocks = blocks - RED_PUT_BLOCKS;
    const auto ctaBaseRedChunk = scaledChunkSize / reduceBlocks;
    const auto ctaRedResidue = static_cast<int>(scaledChunkSize % reduceBlocks);
    const auto ctaRedChunk = ctaBaseRedChunk + (reduceBIdx < ctaRedResidue);
    const auto redOffsetElems = ctaBaseRedChunk * reduceBIdx + min(reduceBIdx, ctaRedResidue);
    const auto redStartOffset = redOffsetElems * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    const size_t bytesRed = static_cast<size_t>(ctaRedChunk) * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::PIPELINE_BYTES);
    static_assert(sizeof(cuda::std::byte**) == sizeof(uint64_t**) && alignof(cuda::std::byte**) == alignof(uint64_t**));
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(signals + MAX_RANKS_PER_DOMAIN);
    const auto prefixOffset = stagingPrefix + (isSrcSpread ? ctx.rank * bytes : 0);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      // signals
      signals[peer] = ctx.signals[ctx.rank] + peer;
      // staging
      const auto offset = prefixOffset + redStartOffset;
      staging[peer] = ctx.stagingTR[peer] + offset;
    }
    auto flag = epoch;
    cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      flag++;
      const ReduceTRArgs redArgs{
        .sources = staging,
        .dst = dstP,
        .bytesRed = bytesRed,
        .world = ctx.world, // <- TODO: check SASS that no constructor instructions are emitted for this subobject
      };
      for (int peer = static_cast<int>(threadIdx.x); peer < redArgs.world; peer += SutureAtom::THREADS) {
        auto* __restrict__ signal = signals[peer];
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
        auto isHere = sig.load(cuda::memory_order_relaxed) >= flag;
        while (!isHere) {
          isHere = sig.load(cuda::memory_order_relaxed) >= flag;
        }
        cuda::std::ignore = sig.load(cuda::memory_order_acquire);
      }
      __syncthreads();
      SutureAtom::reduce2(redArgs, typedWorkspace);
      dstP += RED_CHUNK_SIZE;
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
        staging[i] += RED_CHUNK_SIZE;
      }
    }
    if (bytes > cutoff) {
      flag++;
      const auto residue = bytes - cutoff;
      const long int scaledChunkSizeLeft = residue / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      const auto ctaBaseRedChunkLeft = scaledChunkSizeLeft / reduceBlocks;
      const auto ctaRedResidueLeft = static_cast<int>(scaledChunkSizeLeft % reduceBlocks);
      const auto ctaRedChunkLeft = ctaBaseRedChunkLeft + (reduceBIdx < ctaRedResidueLeft);
      const auto redOffsetElemsLeft = ctaBaseRedChunkLeft * reduceBIdx + min(reduceBIdx, ctaRedResidueLeft);
      const auto redStartOffsetLeft = redOffsetElemsLeft * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      const size_t bytesRedLeft = static_cast<size_t>(ctaRedChunkLeft) * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;

      dstP = dst + (RED_CHUNK_SIZE * chunks + redStartOffsetLeft);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
        auto* __restrict__ stagingBase = staging[i] - (chunks * RED_CHUNK_SIZE + redStartOffset);
        staging[i] = stagingBase + (chunks * RED_CHUNK_SIZE + redStartOffsetLeft);
      }
      const ReduceTRArgs redArgs{
        .sources = staging,
        .dst = dstP,
        .bytesRed = bytesRedLeft,
        .world = ctx.world,
      };
      for (int peer = static_cast<int>(threadIdx.x); peer < redArgs.world; peer += SutureAtom::THREADS) {
        auto* __restrict__ signal = signals[peer];
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
        auto isHere = sig.load(cuda::memory_order_relaxed) >= flag;
        while (!isHere) {
          isHere = sig.load(cuda::memory_order_relaxed) >= flag;
        }
        cuda::std::ignore = sig.load(cuda::memory_order_acquire);
      }
      __syncthreads();
      SutureAtom::reduce2(redArgs, typedWorkspace);
    }

    if (!threadIdx.x) {
      ctx.epochs[bIdx] = flag;
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
    reduce<SutureAtom>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx);
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
    reduce<SutureAtom>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, true);
  }
}
#endif //SUTURE_COLLECTIVE_CUH
