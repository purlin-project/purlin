//
// Created by osayamen on 5/25/26.
//

#ifndef PURLIN_GATHER_CUH
#define PURLIN_GATHER_CUH
#include "base.cuh"
#include "epoch.cuh"
#include "context.cuh"
#include "partition.cuh"
#include "transfer.cuh"

namespace purlin {
  template<typename PurlinAtom, typename CollConfig, DataLayout outputLayout>
  __device__ __forceinline__
  static void gatherConsumer(cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const Context& ctx,
    const EpochState& epochState,
    const int& bIdx,
    const PeerBlock& peerBlock,
    uint64_t* __restrict__ const& signalBase,
    const size_t& stagingPrefix, const size_t& globalMaxBytes = 0) {
    // Under the multimem allReduce the reduced shards were broadcast into every
    // replica, so the gather is a local read of this rank's own staging.
    constexpr auto localGather =
      PurlinAtom::BaseConfig::DATAPATH == Datapath::multimem && outputLayout == DataLayout::scattered;
    size_t sourceOffset = 0;
    if constexpr (outputLayout == DataLayout::scattered) {
      sourceOffset = bytes * peerBlock.peer;
    }
    else if constexpr (outputLayout == DataLayout::transposed) {
      sourceOffset = bytes * ctx.rank;
    }
    const auto sigPrefix = (epochState.epoch % 2) * ctx.world;
    auto* __restrict__ vSignal = ctx.varOffsetSignals[ctx.rank] + (sigPrefix + peerBlock.peer);
    if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
      auto* __restrict__ srcOffset = reinterpret_cast<size_t*>(workspace);
      if (!threadIdx.x) {
        if constexpr (outputLayout == DataLayout::transposedV) {
          const auto cv = vSignal->waitUntilAtLeast(epochState.nextEpoch);
          *srcOffset = cv.data; // obtain offset
          cuda::std::ignore = vSignal->loadAcquire();
        }
        else {
          auto* __restrict__ signal = signalBase + peerBlock.peer;
          waitUntilAtLeast(signal, epochState.nextEpoch);
        }
      }
      __syncthreads();
      if constexpr (outputLayout == DataLayout::transposedV) {
        sourceOffset = *srcOffset;
        __syncthreads(); // <- ensures everyone has read the above
      }
      const auto* __restrict__ srcBase =
        ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + sourceOffset);
      const auto* __restrict__ srcP = srcBase;
      auto* __restrict__ dstP = dst;
      superCopy<PurlinAtom>(dstP, srcP, bytes, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
      markEpoch(ctx, bIdx, epochState.nextEpoch);
    }
    else {
      auto* __restrict__ srcBase =
        ctx.staging[localGather ? ctx.rank : peerBlock.peer] + (stagingPrefix + sourceOffset);
      auto* __restrict__ srcP = srcBase;
      auto* __restrict__ dstP = dst;
      constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
      const auto chunks = static_cast<int>(bytes / CollConfig::CHUNK_SIZE);
      const auto chunkCutoff = CollConfig::CHUNK_SIZE * chunks;
      auto flag = epochState.epoch;
      auto* __restrict__ srcOffset = reinterpret_cast<size_t*>(workspace);
      auto* __restrict__ signal = signalBase + peerBlock.peer;
      if constexpr (outputLayout == DataLayout::transposedV) {
        if (chunks > 0) {
          flag++;
          if (!threadIdx.x) {
            const auto cv = vSignal->waitUntilAtLeast(flag);
            *srcOffset = cv.data;
            cuda::std::ignore = vSignal->loadAcquire();
          }
          __syncthreads();
          sourceOffset = *srcOffset;
          srcP += sourceOffset;
          srcBase += sourceOffset;
          __syncthreads();
          superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
            peerBlock.blockSetSize, peerBlock.intraIdx);
          srcP += CollConfig::CHUNK_SIZE;
          dstP += CollConfig::CHUNK_SIZE;
        }
        for (int i = 1; i < chunks; ++i) {
          flag++;
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
            peerBlock.blockSetSize, peerBlock.intraIdx);
          srcP += CollConfig::CHUNK_SIZE;
          dstP += CollConfig::CHUNK_SIZE;
        }
      }
      else {
        for (int i = 0; i < chunks; ++i) {
          flag++;
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          superCopy<PurlinAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace,
            peerBlock.blockSetSize, peerBlock.intraIdx);
          srcP += CollConfig::CHUNK_SIZE;
          dstP += CollConfig::CHUNK_SIZE;
        }
      }
      if (bytes > chunkCutoff) {
        flag++;
        const auto residue = bytes - chunkCutoff;
        dstP = dst + (CollConfig::CHUNK_SIZE * chunks);
        srcP = srcBase + (CollConfig::CHUNK_SIZE * chunks);
        if (!threadIdx.x) {
          if constexpr (outputLayout == DataLayout::transposedV) {
            if (chunks == 0) {
              const auto cv = vSignal->waitUntilAtLeast(flag);
              *srcOffset = cv.data;
              cuda::std::ignore = vSignal->loadAcquire();
            }
            else {
              waitUntilAtLeast(signal, flag);
            }
          }
          else {
            waitUntilAtLeast(signal, flag);
          }
        }
        __syncthreads();
        if constexpr (outputLayout == DataLayout::transposedV) {
          if (chunks == 0) {
            sourceOffset = *srcOffset;
            srcP += sourceOffset;
            __syncthreads();
          }
        }
        superCopy<PurlinAtom>(dstP, srcP, residue, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
      }
      const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
        outputLayout == DataLayout::transposedV ? cuda::ceil_div(globalMaxBytes, chunkSize) :
        (outputLayout == DataLayout::packedV ?
          cuda::ceil_div(ctx.vState.maxBytes, chunkSize) : flag - epochState.epoch));
      markEpoch(ctx, bIdx, nextEpoch);
    }
  }

  template<typename PurlinAtom, DataLayout inputLayout, typename BT = int>
  __device__ __forceinline__
  static void gatherLR(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& nextEpoch,
    const uint& senseBit,
    const size_t* __restrict__ const& sizes = nullptr,
    const size_t* __restrict__ const& inSizes = nullptr) {
    const auto stagingPrefix = (senseBit * ctx.world * purlin::PACKET_BUFFER_SIZE);
    constexpr auto bufferStride = purlin::PACKET_BUFFER_SIZE;
    const auto rankOffset = ctx.rank * purlin::PACKET_BUFFER_SIZE;
    auto* __restrict__ localStaging = ctx.stagingLR[ctx.rank] + stagingPrefix;
    // shared memory state
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(workspace);
    auto* __restrict__ offsets = reinterpret_cast<size_t*>(staging + MAX_RANKS_PER_DOMAIN);
    auto* __restrict__ sizesP = offsets + MAX_RANKS_PER_DOMAIN;
    auto* __restrict__ inSizesP = sizesP + MAX_RANKS_PER_DOMAIN;
    auto* __restrict__ inOffsetsP = inSizesP + MAX_RANKS_PER_DOMAIN;
    static_assert(inputLayout != DataLayout::scatteredV || (PurlinAtom::COPY_SMEM_SIZE >= sizeof(void*) * 5));
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += PurlinAtom::THREADS) {
      staging[peer] = ctx.stagingLR[peer] + (stagingPrefix + rankOffset);
      if constexpr (inputLayout == DataLayout::packedV) {
        sizesP[peer] = sizes[peer];
      }
      else if constexpr (inputLayout == DataLayout::scatteredV) {
        sizesP[peer] = sizes[peer];
        inSizesP[peer] = inSizes[peer];
      }
    }
    if constexpr (inputLayout == DataLayout::packedV) {
      auto* __restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte*>(sizesP + MAX_RANKS_PER_DOMAIN);
      prefixSum<PurlinAtom::THREADS>(sizes, offsets, scanWorkspace, ctx.world);
    }
    else if constexpr (inputLayout == DataLayout::scatteredV) {
      auto* __restrict__ scanWorkspace = reinterpret_cast<cuda::std::byte*>(inOffsetsP + MAX_RANKS_PER_DOMAIN);
      prefixSum<PurlinAtom::THREADS>(sizes, offsets, scanWorkspace, ctx.world);
      prefixSum<PurlinAtom::THREADS>(inSizes, inOffsetsP, scanWorkspace, ctx.world);
    }
    const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
    __syncthreads();
    const auto isInPlace = inputLayout == DataLayout::packed ? src == (dst + ctx.rank * bytes) : false; // TODO
    const LRArgs gArgs{
      .src = src,
      .staging = staging,
      .localStaging = localStaging,
      .dst = dst,
      .flag = nextEpoch,
      .bufferStride = bufferStride,
      .bytes = bytes,
      .maxBytes = inputLayout == DataLayout::scatteredV ? ctx.vState.maxOutBytes : ctx.vState.maxBytes,
      .inSizes = inSizes,
      .sizes = sizesP,
      .inOffsets = inOffsetsP,
      .offsets = offsets,
      .blocks = blocks,
      .tIdx = static_cast<int>(tid),
      .world = ctx.world,
      .rank = ctx.rank,
      .isInPlace = isInPlace,
    };
    fascia::gather<typename PurlinAtom::BaseConfig, inputLayout>(gArgs);
    __syncthreads();
    markEpoch(ctx, bIdx, nextEpoch);
    markUnusedEpochs<PurlinAtom>(ctx, blocks, blocks, nextEpoch, tid);
  }

  template<
    typename PurlinAtom,
    typename CollConfig,
    DataLayout inputLayout,
    DataLayout outputLayout,
    typename BT
  >
  __device__ __forceinline__
  static void gatherNonChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const EpochState& epochState,
    const int& collBlocks,
    const size_t* __restrict__ const& sizes = nullptr,
    const size_t* __restrict__ const& inSizes = nullptr) {
    static_assert(CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked);
    if constexpr ((inputLayout == DataLayout::packed && outputLayout == DataLayout::packed) ||
      (inputLayout == DataLayout::packedV && outputLayout == DataLayout::packedV)) {
      // AllGather
      constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      if (bIdx < CollConfig::PUT_BLOCKS) {
        const auto globalBytes = bytes;
        const auto [bytesPut, putStartOffset] = partition<CollConfig::PUT_BLOCKS, alignmentBytes>(globalBytes, bIdx);
        const auto* __restrict__ srcP = src + putStartOffset;
        auto* __restrict__ dstBase = ctx.staging[ctx.rank] + epochState.trStagingPrefix;
        auto* __restrict__ dstP = dstBase + putStartOffset;
        PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        if (threadIdx.x / WARP_SIZE == 0) {
          const auto laneId = threadIdx.x % WARP_SIZE;
          int shouldNotify = CollConfig::PUT_BLOCKS == 1 ? 1 : 0;
          if (!threadIdx.x) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == CollConfig::PUT_BLOCKS;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            signalAllPeers(ctx.signals, ctx.rank, ctx.world, epochState.nextEpoch, laneId);
            __syncwarp();
          }
        }
        const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
        markEpoch(ctx, bIdx, epochState.nextEpoch);
        markUnusedEpochs<PurlinAtom, CollConfig::PUT_BLOCKS>(ctx, collBlocks, epochState.nextEpoch, tid);
        return;
      }
      const auto cBIdx = bIdx - CollConfig::PUT_BLOCKS;
      const auto consumerBlocks = static_cast<int>(blocks) - CollConfig::PUT_BLOCKS;
      const auto skewed = inputLayout != DataLayout::packed &&
        isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l);
      static_assert(inputLayout != DataLayout::packedV ||
        PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
      auto* __restrict__ sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
      auto* __restrict__ offsets = sizesP + MAX_RANKS_PER_DOMAIN;
      if constexpr (inputLayout == DataLayout::packedV) {
        for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
          sizesP[i] = sizes[i];
        }
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
        __syncthreads();
      }
      const auto peerBlock = skewed ? mapWeightedPeerBlock(cBIdx, consumerBlocks, sizesP, workspace, ctx.world)
      : mapPeerBlock(cBIdx, consumerBlocks / ctx.world);
      const auto peerBytes = inputLayout == DataLayout::packedV ? sizes[peerBlock.peer] : bytes;
      const auto offset = inputLayout == DataLayout::packedV ? offsets[peerBlock.peer] : bytes * peerBlock.peer;
      gatherConsumer<PurlinAtom, CollConfig, outputLayout>(
        dst + offset,
        peerBytes,
        workspace,
        ctx,
        epochState,
        bIdx,
        peerBlock,
        ctx.signals[ctx.rank],
        epochState.trStagingPrefix);
    }
    else {
      // All2All
      static_assert(inputLayout != DataLayout::packed);
      constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
      const int stagingBlocks = ctx.stagingBlocks;
      if (bIdx < stagingBlocks) {
        auto* sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        auto* shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
        auto* offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
        if constexpr (inputLayout == DataLayout::scatteredV) {
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
            sizesP[i] = inSizes[i];
            shiftedSizes[i] = inSizes[(i + ctx.rank + 1) % ctx.world];
          }
          prefixSum<PurlinAtom::THREADS>(inSizes, offsets, workspace, ctx.world);
        }
        __syncthreads();
        static_assert(inputLayout != DataLayout::scatteredV ||
          PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
        const auto skewed = inputLayout == DataLayout::scatteredV &&
          isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l);
        const auto peerBlock =  skewed ? mapWeightedPeerBlock(bIdx, stagingBlocks, shiftedSizes, workspace, ctx.world) :
        mapPeerBlock(bIdx, stagingBlocks / ctx.actualWorld, ctx.rank, ctx.world);
        const auto peer = skewed ? ((peerBlock.peer + ctx.rank + 1) % ctx.world) : peerBlock.peer;
        // no chunking
        const auto myBytes = inputLayout == DataLayout::scattered ? bytes : inSizes[peer];
        const auto [bytesPut, putStartOffset] = partition<alignmentBytes>(myBytes, peerBlock.blockSetSize, peerBlock.intraIdx);
        const auto shiftOffset = inputLayout == DataLayout::scattered ? bytes * peer : offsets[peer];
        const auto* __restrict__ srcP = src + (putStartOffset + shiftOffset);
        auto* __restrict__ dstBase = ctx.staging[ctx.rank] + (epochState.trStagingPrefix + shiftOffset);
        auto* __restrict__ dstP = dstBase + putStartOffset;
        auto* __restrict__ putCounter = ctx.putCounter + peer;
        PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        if (threadIdx.x / WARP_SIZE == 0) {
          const auto laneId = threadIdx.x % WARP_SIZE;
          int shouldNotify = peerBlock.blockSetSize == 1 ? 1 : 0;
          if (!threadIdx.x) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == peerBlock.blockSetSize;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            if (!laneId) {
              if constexpr (inputLayout == DataLayout::scattered) {
                auto* __restrict__ signal = ctx.signals[peer] + ctx.rank;
                signalOne(signal, epochState.nextEpoch);
              }
              else {
                const auto sigPrefix = (epochState.epoch % 2) * ctx.world;
                auto* __restrict__ signal = ctx.varOffsetSignals[peer] + (sigPrefix + ctx.rank);
                signal->writeRelease(shiftOffset, epochState.nextEpoch);
              }
            }
            __syncwarp();
          }
        }
        const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
        markEpoch(ctx, bIdx, epochState.nextEpoch);
        markUnusedEpochs<PurlinAtom>(ctx, collBlocks, stagingBlocks, epochState.nextEpoch, tid);
        return;
      }
      const auto totalPutBlocks = stagingBlocks + CollConfig::LOCAL_PUT_BLOCKS;
      if (bIdx < totalPutBlocks) {
        const auto myBytes = inputLayout == DataLayout::scattered ? bytes : sizes[ctx.rank];
        auto* inOffsets = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES) + MAX_RANKS_PER_DOMAIN;
        auto* outOffsets = inOffsets + MAX_RANKS_PER_DOMAIN;
        auto inOffset = bytes * ctx.rank;
        auto outOffset = bytes * ctx.rank;
        if constexpr (inputLayout == DataLayout::scatteredV) {
          prefixSum<PurlinAtom::THREADS>(inSizes, inOffsets, workspace, ctx.world);
          prefixSum<PurlinAtom::THREADS>(sizes, outOffsets, workspace, ctx.world);
          __syncthreads();
          inOffset = inOffsets[ctx.rank];
          outOffset = outOffsets[ctx.rank];
        }
        const auto lBIdx = bIdx - stagingBlocks;
        auto* __restrict__ srcP = src + inOffset;
        auto* __restrict__ dstP = dst + outOffset;
        superCopy<PurlinAtom, CollConfig::LOCAL_PUT_BLOCKS>(dstP, srcP, myBytes, workspace, lBIdx);
        markEpoch(ctx, bIdx, epochState.nextEpoch);
        return;
      }
      // consumers
      auto* sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
      auto* shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
      auto* offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
      if constexpr (inputLayout == DataLayout::scatteredV) {
        for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
          sizesP[i] = sizes[i];
          shiftedSizes[i] = sizes[(i + ctx.rank + 1) % ctx.world];
        }
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
        __syncthreads();
      }
      const auto cBIdx = bIdx - totalPutBlocks;
      const auto consumerBlocks = static_cast<int>(blocks - totalPutBlocks);
      const auto skewed = (inputLayout == DataLayout::scatteredV &&
          isSkewed(ctx.vState.totalOutBytes, ctx.vState.maxOutBytes, ctx.world_l));
      auto peerBlock = skewed ?
        mapWeightedPeerBlock(cBIdx, consumerBlocks, shiftedSizes, workspace, ctx.world) :
        mapPeerBlock(cBIdx, consumerBlocks / ctx.actualWorld, ctx.rank, ctx.world);
      peerBlock.peer = skewed ? (peerBlock.peer + ctx.rank + 1) % ctx.world : peerBlock.peer;
      const auto offset = inputLayout==DataLayout::scatteredV ? offsets[peerBlock.peer]: bytes * peerBlock.peer;
      const auto peerBytes = inputLayout == DataLayout::scatteredV ? sizes[peerBlock.peer] : bytes;
      auto* __restrict__ sigBase = inputLayout == DataLayout::scatteredV ? nullptr : ctx.signals[ctx.rank];
      gatherConsumer<PurlinAtom, CollConfig, outputLayout>(
        dst + offset,
        peerBytes,
        workspace,
        ctx,
        epochState,
        bIdx,
        peerBlock,
        sigBase,
        epochState.trStagingPrefix
      );
    }
  }

  template<
    typename PurlinAtom,
    typename CollConfig,
    DataLayout inputLayout,
    DataLayout outputLayout,
    typename BT
  >
  __device__ __forceinline__
  static void gatherChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const EpochState& epochState,
    const int& collBlocks,
    const size_t* __restrict__ const& sizes = nullptr,
    const size_t* __restrict__ const& inSizes = nullptr) {
    static_assert(CollConfig::COLLECTIVE_TYPE == CollectiveType::chunked);
    static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
    constexpr auto alignmentBytes = PurlinAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    if constexpr ((inputLayout == DataLayout::packed && outputLayout == DataLayout::packed) ||
      (inputLayout == DataLayout::packedV && outputLayout == DataLayout::packedV)) {
      // AllGather
      const auto chunks = static_cast<int>(bytes / CollConfig::CHUNK_SIZE);
      const auto cutoff = CollConfig::CHUNK_SIZE * chunks;
      if (bIdx < CollConfig::PUT_BLOCKS) {
        constexpr auto blockSetSize = CollConfig::PUT_BLOCKS;
        auto flag = epochState.epoch;
        auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
          signals[i] = ctx.signals[i] + ctx.rank;
        }
        __syncthreads();
        const auto [bytesPut, putStartOffset] = partition<CollConfig::CHUNK_SIZE, alignmentBytes>
        (blockSetSize, bIdx);
        const auto* __restrict__ srcP = src + putStartOffset;
        auto* __restrict__ dstBase = ctx.staging[ctx.rank] + epochState.trStagingPrefix;
        auto* __restrict__ dstP = dstBase + putStartOffset;
        const int laneId = static_cast<int>(threadIdx.x % WARP_SIZE);
        auto* __restrict__ putCounter = ctx.putCounter;
        for (int chunk = 0; chunk < chunks; ++chunk) {
          PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
          __syncthreads();
          flag++;
          if (threadIdx.x / WARP_SIZE == 0) {
            int shouldNotify = blockSetSize == 1 ? 1 : 0;
            if (blockSetSize > 1 && !laneId) {
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunk)};
              shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
              if (shouldNotify) {
                s.store(0, cuda::memory_order_relaxed);
              }
            }
            __syncwarp();
            shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
            if (shouldNotify) {
              signalPointerList(signals, ctx.world, flag, laneId);
              __syncwarp();
            }
          }
          dstP += CollConfig::CHUNK_SIZE;
          srcP += CollConfig::CHUNK_SIZE;
        }
        if (bytes > cutoff) {
          const auto residue = bytes - cutoff;
          const auto [bytesPutLeft, putStartOffsetLeft] = partition<alignmentBytes>
          (residue, blockSetSize, bIdx);
          srcP = src + (CollConfig::CHUNK_SIZE * chunks + putStartOffsetLeft);
          dstP = dstBase + (CollConfig::CHUNK_SIZE * chunks + putStartOffsetLeft);
          PurlinAtom::copy(dstP, srcP, bytesPutLeft, workspace);
          __syncthreads();
          flag++;
          if (threadIdx.x / WARP_SIZE == 0) {
            int shouldNotify = blockSetSize == 1 ? 1 : 0;
            if (blockSetSize > 1 && !laneId) {
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunks)};
              shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
              if (shouldNotify) {
                s.store(0, cuda::memory_order_relaxed);
              }
            }
            __syncwarp();
            shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
            if (shouldNotify) {
              signalPointerList(signals, ctx.world, flag, laneId);
              __syncwarp();
            }
          }
        }
        const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
        constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
        const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
          inputLayout == DataLayout::packedV ?
          cuda::ceil_div(ctx.vState.maxBytes, chunkSize) : flag - epochState.epoch);
        markEpoch(ctx, bIdx, nextEpoch);
        markUnusedEpochs<PurlinAtom, CollConfig::PUT_BLOCKS>(ctx, collBlocks, nextEpoch, tid);
        return;
      }
      const auto cBIdx = bIdx - CollConfig::PUT_BLOCKS;
      const auto consumerBlocks = static_cast<int>(blocks) - CollConfig::PUT_BLOCKS;
      const auto skewed = inputLayout != DataLayout::packed &&
        isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l);
      static_assert(inputLayout != DataLayout::packedV ||
        PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
      auto* __restrict__ sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
      auto* __restrict__ offsets = sizesP + MAX_RANKS_PER_DOMAIN;
      if constexpr (inputLayout == DataLayout::packedV) {
        for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
          sizesP[i] = sizes[i];
        }
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
        __syncthreads();
      }
      const auto peerBlock = skewed ? mapWeightedPeerBlock(cBIdx, consumerBlocks, sizesP, workspace, ctx.world)
      : mapPeerBlock(cBIdx, consumerBlocks / ctx.world);
      const auto peerBytes = inputLayout == DataLayout::packedV ? sizesP[peerBlock.peer] : bytes;
      const auto offset = inputLayout == DataLayout::packedV ? offsets[peerBlock.peer] : bytes * peerBlock.peer;
      gatherConsumer<PurlinAtom, CollConfig, outputLayout>(
        dst + offset,
        peerBytes,
        workspace,
        ctx,
        epochState,
        bIdx,
        peerBlock,
        ctx.signals[ctx.rank],
        epochState.trStagingPrefix
      );
    }
    else {
      constexpr auto inPlace = false;
      // All2All
      const int stagingBlocks = ctx.stagingBlocks;
      if (bIdx < stagingBlocks) {
        auto* sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
        auto* shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
        auto* offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
        if constexpr (inputLayout == DataLayout::scatteredV) {
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
            sizesP[i] = inSizes[i];
            shiftedSizes[i] = inSizes[(i + ctx.rank + 1) % ctx.world];
          }
          prefixSum<PurlinAtom::THREADS>(inSizes, offsets, workspace, ctx.world);
        }
        __syncthreads();
        static_assert(inputLayout != DataLayout::scatteredV ||
          PurlinAtom::COPY_PIPELINE_SMEM_BYTES >= WEIGHTED_PEER_BLOCK_STATE_BYTES);
        const auto skewed = (inputLayout == DataLayout::scatteredV &&
        isSkewed(ctx.vState.totalBytes, ctx.vState.maxBytes, ctx.world_l));
        auto peerBlock = skewed ?
        mapWeightedPeerBlock(bIdx, stagingBlocks, shiftedSizes, workspace, ctx.world) :
        mapPeerBlock(bIdx, stagingBlocks / ctx.actualWorld, ctx.rank, ctx.world);
        if constexpr (inputLayout == DataLayout::scatteredV) {
          peerBlock.peer = skewed ? ((peerBlock.peer + ctx.rank + 1) % ctx.world) : peerBlock.peer;
        }
        const auto myBytes = inputLayout == DataLayout::scattered ? bytes : inSizes[peerBlock.peer];
        const auto chunks = static_cast<int>(myBytes / CollConfig::CHUNK_SIZE);
        const auto cutoff = CollConfig::CHUNK_SIZE * chunks;
        auto flag = epochState.epoch;
        auto* __restrict__ signal = ctx.signals[peerBlock.peer] + ctx.rank;
        const auto [bytesPut, putStartOffset] = partition<CollConfig::CHUNK_SIZE, alignmentBytes>
        (peerBlock.blockSetSize, peerBlock.intraIdx);
        const auto intraOffset = inputLayout == DataLayout::scattered ? bytes * peerBlock.peer :
        offsets[peerBlock.peer];
        const auto* __restrict__ srcP = src + (putStartOffset + intraOffset);
        auto* __restrict__ dstBase = ctx.staging[ctx.rank] + (epochState.trStagingPrefix + intraOffset);
        auto* __restrict__ dstP = dstBase + putStartOffset;
        const int laneId = static_cast<int>(threadIdx.x % WARP_SIZE);
        auto* __restrict__ putCounter = ctx.putCounter + peerBlock.peer * MAX_CHUNKS;
        const auto sigPrefix = (epochState.epoch % 2) * ctx.world;
        auto* __restrict__ vSignal = ctx.varOffsetSignals[peerBlock.peer] + (sigPrefix + ctx.rank);
        for (int chunk = 0; chunk < chunks; ++chunk) {
          PurlinAtom::copy(dstP, srcP, bytesPut, workspace);
          __syncthreads();
          flag++;
          if (threadIdx.x / WARP_SIZE == 0) {
            int shouldNotify = peerBlock.blockSetSize == 1 ? 1 : 0;
            if (peerBlock.blockSetSize > 1 && !laneId) {
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunk)};
              shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == peerBlock.blockSetSize;
              if (shouldNotify) {
                s.store(0, cuda::memory_order_relaxed);
              }
            }
            __syncwarp();
            shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
            if (shouldNotify) {
              if (!laneId) {
                if constexpr (inputLayout == DataLayout::scattered) {
                  signalOne(signal, flag);
                }
                else {
                  if (chunk == 0) {
                    vSignal->writeRelease(intraOffset, flag);
                  }
                  else {
                    signalOne(signal, flag);
                  }
                }
              }
              __syncwarp();
            }
          }
          dstP += CollConfig::CHUNK_SIZE;
          srcP += CollConfig::CHUNK_SIZE;
        }
        if (myBytes > cutoff) {
          const auto residue = myBytes - cutoff;
          const auto [bytesPutLeft, putStartOffsetLeft] = partition<alignmentBytes>
          (residue, peerBlock.blockSetSize, peerBlock.intraIdx);
          srcP = src + ((CollConfig::CHUNK_SIZE * chunks + putStartOffsetLeft) + intraOffset);
          dstP = dstBase + (CollConfig::CHUNK_SIZE * chunks + putStartOffsetLeft);
          PurlinAtom::copy(dstP, srcP, bytesPutLeft, workspace);
          __syncthreads();
          flag++;
          if (threadIdx.x / WARP_SIZE == 0) {
            int shouldNotify = peerBlock.blockSetSize == 1 ? 1 : 0;
            if (peerBlock.blockSetSize > 1 && !laneId) {
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunks)};
              shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == peerBlock.blockSetSize;
              if (shouldNotify) {
                s.store(0, cuda::memory_order_relaxed);
              }
            }
            __syncwarp();
            shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
            if (shouldNotify) {
              if (!laneId) {
                if constexpr (inputLayout == DataLayout::scattered) {
                  signalOne(signal, flag);
                }
                else {
                  if (chunks == 0) {
                    vSignal->writeRelease(intraOffset, flag);
                  }
                  else {
                    signalOne(signal, flag);
                  }
                }
              }
              __syncwarp();
            }
          }
        }
        const auto tid = bIdx * PurlinAtom::THREADS + threadIdx.x;
        constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
        const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
          inputLayout == DataLayout::scatteredV ?
          cuda::ceil_div(bytes, chunkSize) : flag - epochState.epoch);
        markEpoch(ctx, bIdx, nextEpoch);
        markUnusedEpochs<PurlinAtom>(ctx, collBlocks, stagingBlocks, nextEpoch, tid);
        return;
      }
      const auto totalPutBlocks = stagingBlocks + (inPlace ? 0 : CollConfig::LOCAL_PUT_BLOCKS);
      if (!inPlace && bIdx < totalPutBlocks) {
        const auto myBytes = inputLayout == DataLayout::scattered ? bytes : sizes[ctx.rank];
        auto* inOffsets = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES) + MAX_RANKS_PER_DOMAIN;
        auto* outOffsets = inOffsets + MAX_RANKS_PER_DOMAIN;
        auto inOffset = bytes * ctx.rank;
        auto outOffset = bytes * ctx.rank;
        if constexpr (inputLayout == DataLayout::scatteredV) {
          prefixSum<PurlinAtom::THREADS>(inSizes, inOffsets, workspace, ctx.world);
          prefixSum<PurlinAtom::THREADS>(sizes, outOffsets, workspace, ctx.world);
          __syncthreads();
          inOffset = inOffsets[ctx.rank];
          outOffset = outOffsets[ctx.rank];
        }
        const auto lBIdx = bIdx - stagingBlocks;
        auto* __restrict__ srcP = src + inOffset;
        auto* __restrict__ dstP = dst + outOffset;
        if constexpr (inputLayout == DataLayout::scatteredV) {
          constexpr auto stageBytes = static_cast<size_t>(PurlinAtom::STAGE_BYTES);
          constexpr auto partGranularity = stageBytes * CollConfig::LOCAL_PUT_BLOCKS;
          const auto paddedBytes = alignUp(myBytes, partGranularity);
          const auto [bytesP, startOffset] = partition<CollConfig::LOCAL_PUT_BLOCKS, static_cast<int>(stageBytes)>(paddedBytes, lBIdx);
          const auto actualBytes = startOffset >= myBytes ? size_t{0} :
            cuda::std::min(bytesP, myBytes - startOffset);
          PurlinAtom::copy(dstP + startOffset, srcP + startOffset, actualBytes, workspace);
        }
        else {
          superCopy<PurlinAtom, CollConfig::LOCAL_PUT_BLOCKS>(dstP, srcP, myBytes, workspace, lBIdx);
        }
        constexpr auto chunkSize = CollConfig::CHUNK_SIZE;
        const auto nextEpoch = chunkedNextEpoch(epochState.epoch,
          inputLayout == DataLayout::scatteredV ?
          cuda::ceil_div(bytes, chunkSize) :
          static_cast<size_t>(cuda::ceil_div(myBytes, chunkSize)));
        markEpoch(ctx, bIdx, nextEpoch);
        return;
      }
      // consumers
      const auto cBIdx = bIdx - totalPutBlocks;
      const auto consumerBlocks = static_cast<int>(blocks - totalPutBlocks);
      auto* sizesP = reinterpret_cast<size_t*>(workspace + PurlinAtom::COPY_PIPELINE_SMEM_BYTES);
      auto* shiftedSizes = sizesP + MAX_RANKS_PER_DOMAIN;
      auto* offsets = shiftedSizes + MAX_RANKS_PER_DOMAIN;
      if constexpr (inputLayout == DataLayout::scatteredV) {
        for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += PurlinAtom::THREADS) {
          sizesP[i] = sizes[i];
          shiftedSizes[i] = sizes[(i + ctx.rank + 1) % ctx.world];
        }
        prefixSum<PurlinAtom::THREADS>(sizes, offsets, workspace, ctx.world);
        __syncthreads();
      }
      const auto skewed = (inputLayout == DataLayout::scatteredV &&
          isSkewed(ctx.vState.totalOutBytes, ctx.vState.maxOutBytes, ctx.world_l));
      auto peerBlock = skewed ?
        mapWeightedPeerBlock(cBIdx, consumerBlocks, shiftedSizes, workspace, ctx.world) :
        mapPeerBlock(cBIdx, consumerBlocks / ctx.actualWorld, ctx.rank, ctx.world);
      if constexpr (inputLayout == DataLayout::scatteredV) {
        peerBlock.peer = skewed ? (peerBlock.peer + ctx.rank + 1) % ctx.world : peerBlock.peer;
      }
      const auto offset = inputLayout==DataLayout::scatteredV ? offsets[peerBlock.peer]: bytes * peerBlock.peer;
      const auto peerBytes = inputLayout == DataLayout::scatteredV ? sizes[peerBlock.peer] : bytes;
      gatherConsumer<PurlinAtom, CollConfig, outputLayout>(
        dst + offset,
        peerBytes,
        workspace,
        ctx,
        epochState,
        bIdx,
        peerBlock,
        ctx.signals[ctx.rank],
        epochState.trStagingPrefix, bytes
      );
    }
  }
}
#endif //PURLIN_GATHER_CUH
