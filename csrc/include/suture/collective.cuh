//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_COLLECTIVE_CUH
#define SUTURE_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"

namespace suture {
  struct PartitionResult {
    const size_t bytes;
    const size_t startOffset;
  };
  struct EpochState {
    const uint64_t epoch;
    const uint64_t nextEpoch;
    const uint senseBit;
    const size_t trStagingPrefix;
    const size_t lrStagingPrefix;
  };
  struct PeerBlock {
    const int peer;
    const int intraIdx;
    const int blockSetSize;
  };
  enum class CollectiveType {
    chunked,
    nonChunked
  };
  template<
    CollectiveType ct,
    int putBlocks,
    int gatherBlocks,
    size_t chunkSize
  >
  struct CollectiveConfig {
    static constexpr int PUT_BLOCKS = putBlocks;
    static constexpr int GATHER_BLOCKS = gatherBlocks;
    static constexpr size_t CHUNK_SIZE = chunkSize;
    static constexpr CollectiveType COLLECTIVE_TYPE = ct;
  };
  using CollectiveConfigLR = void;
  __device__ __forceinline__
  static auto makeEpochState(const Context& ctx, const int& bIdx) {
    const auto epoch = ctx.epochs[bIdx];
    const auto senseBit = static_cast<uint>(epoch % 2);
    return EpochState{
      .epoch = epoch,
      .nextEpoch = epoch + static_cast<uint64_t>(1),
      .senseBit = senseBit,
      .trStagingPrefix = STAGING_BUFFER_SIZE_ * senseBit,
      .lrStagingPrefix = senseBit * ctx.world * suture::PACKET_BUFFER_SIZE,
    };
  }
  __device__ __forceinline__
  static void markEpoch(const Context& ctx, const int bIdx, const uint64_t flag) {
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = flag;
    }
  }
  template<typename SutureAtom, typename CB, typename AB>
  __device__ __forceinline__
  static void markUnusedEpochs(const Context& ctx, const CB collBlocks,
    const AB activeBlocks, const uint64_t flag, const int tid) {
    const auto leftover = suture::MAX_NUM_CTAS - collBlocks;
    auto* __restrict__ epochs = ctx.epochs + collBlocks;
    for (int i = tid; i < leftover; i += (SutureAtom::THREADS * activeBlocks)) {
      epochs[i] = flag;
    }
  }
  template<typename SutureAtom, int activeBlocks, typename CB>
  __device__ __forceinline__
  static void markUnusedEpochs(const Context& ctx, const CB collBlocks,
    const uint64_t flag, const int tid) {
    markUnusedEpochs<SutureAtom>(ctx, collBlocks, activeBlocks, flag, tid);
  }
  __device__ __forceinline__
  static void waitUntilAtLeast(uint64_t* __restrict__ const& signal, const uint64_t flag) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
    auto isHere = sig.load(cuda::memory_order_relaxed) >= flag;
    while (!isHere) {
      isHere = sig.load(cuda::memory_order_relaxed) >= flag;
    }
    cuda::std::ignore = sig.load(cuda::memory_order_acquire);
  }
  template<typename SutureAtom>
  __device__ __forceinline__
  static void waitPeerArrivals(uint64_t* __restrict__ const& signalBase,
    const int world, const uint64_t flag, const int tid = static_cast<int>(threadIdx.x)) {
    for (int peer = tid; peer < world; peer += SutureAtom::THREADS) {
      waitUntilAtLeast(signalBase + peer, flag);
    }
  }
  template<typename SutureAtom>
  __device__ __forceinline__
  static void waitPointerList(uint64_t** __restrict__ const& signals,
    const int world, const uint64_t flag, const int tid = static_cast<int>(threadIdx.x)) {
    for (int peer = tid; peer < world; peer += SutureAtom::THREADS) {
      waitUntilAtLeast(signals[peer], flag);
    }
  }
  __device__ __forceinline__
  static void signalOne(uint64_t* __restrict__ const& signal, const uint64_t flag) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
    sig.store(flag, cuda::std::memory_order_release);
  }
  __device__ __forceinline__
  static void signalAllPeers(uint64_t** __restrict__ const& signals,
    const int rank, const int world, const uint64_t flag, const int laneId) {
    for (int peer = laneId; peer < world; peer += WARP_SIZE) {
      signalOne(signals[peer] + rank, flag);
    }
  }
  __device__ __forceinline__
  static void signalPointerList(uint64_t** __restrict__ const& signals,
    const int world, const uint64_t flag, const int laneId) {
    for (int peer = laneId; peer < world; peer += WARP_SIZE) {
      signalOne(signals[peer], flag);
    }
  }
  __device__ __forceinline__
  static auto mapPeerBlock(const int bIdx, const int blockSetSize) {
    return PeerBlock{
      .peer = bIdx / blockSetSize,
      .intraIdx = bIdx % blockSetSize,
      .blockSetSize = blockSetSize,
    };
  }
  template<int AlignmentBytes, typename BT = int>
  __device__ __forceinline__
  constexpr auto partition(const size_t& bytes, const int& blocks, const int& bIdx) {
    static_assert(cuda::std::is_integral_v<BT> || cuda::std::is_same_v<cuda::fast_mod_div<long int>, BT>);
    const long int scaledChunkSize = bytes / AlignmentBytes;
    const auto ctaBaseChunk = static_cast<size_t>(scaledChunkSize / blocks);
    const auto ctaResidue = static_cast<int>(scaledChunkSize % blocks);
    const auto ctaChunk = ctaBaseChunk + (bIdx < ctaResidue);
    const auto offsetElems = ctaBaseChunk * bIdx + cute::min(bIdx, ctaResidue);
    const auto startOffset = offsetElems * AlignmentBytes;
    const size_t bytesSliced = static_cast<size_t>(ctaChunk) * AlignmentBytes;
    return PartitionResult{
      .bytes = bytesSliced,
      .startOffset = startOffset
    };
  }
  template<size_t bytes, int blocks, int AlignmentBytes>
  __device__ __forceinline__
  constexpr auto partition(const int& bIdx) {
    return partition<AlignmentBytes>(bytes, blocks, bIdx);
  }
  template<size_t bytes, int AlignmentBytes, typename BT = int>
  __device__ __forceinline__
  constexpr auto partition(const BT& blocks, const int& bIdx) {
    return partition<AlignmentBytes>(bytes, blocks, bIdx);
  }
  template<int blocks, int AlignmentBytes>
  __device__ __forceinline__
  constexpr auto partition(const size_t& bytes, const int& bIdx) {
    return partition<AlignmentBytes>(bytes, blocks, bIdx);
  }
  // super block put
  template<typename SutureAtom, TransferType pt = TransferType::asynchronous, typename BT = int>
  __device__ __forceinline__
  static void superPut(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src, const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_integral_v<BT> || cuda::std::is_same_v<cuda::fast_mod_div<long int>, BT>);
    // assert(bytes % SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES)
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto [bytesP, startOffset] = partition<alignmentBytes>(bytes, blocks, bIdx);
    const auto* __restrict__ srcP = src + startOffset;
    auto* __restrict__ dstP = dst + startOffset;
    if constexpr (pt == TransferType::asynchronous) {
      SutureAtom::putAsync(dstP, srcP, bytesP, workspace);
    }
    else {
      SutureAtom::put(dstP, srcP, bytesP, workspace);
    }
  }
  // super block put
  template<typename SutureAtom, TransferType pt = TransferType::asynchronous, typename BT = int>
  __device__ __forceinline__
  static void superGet(cuda::std::byte* __restrict__ const& dst, // local
    const cuda::std::byte* __restrict__ const& src, // remote
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    superPut<SutureAtom, pt>(dst, src, bytes, workspace, blocks, bIdx);
  }

  template<typename SutureAtom, size_t bytes, TransferType pt = TransferType::asynchronous, typename BT = int>
  __device__ __forceinline__
  static void superGet(cuda::std::byte* __restrict__ const& dst, // local
    const cuda::std::byte* __restrict__ const& src, // remote
    cuda::std::byte* __restrict__ const& workspace,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    superPut<SutureAtom, pt>(dst, src, bytes, workspace, blocks, bIdx);
  }

  __host__ __forceinline__
  auto getRedRegime(const size_t& bytes, const int& world) {
    if (world == 8) {
      if (bytes <= 64 * 1024) {
        return Regime::latency;
      }
      return Regime::throughput;
    }
    if (world == 2) {
      return bytes <= RED_LATENCY_BOUND_THRESHOLD ? Regime::latency : Regime::throughput;
    }
    if (bytes <= RED_LATENCY_BOUND_THRESHOLD) {
      return Regime::latency;
    }
    return Regime::throughput;
  }
  __host__ __forceinline__
  auto getGatherRegime(const size_t& bytes, const int& world) {
    if (world == 8) {
      if (bytes <= 1024) {
        return Regime::latency;
      }
      return Regime::throughput;
    }
    if (world == 4) {
      if (bytes <= 64 * 1024) {
        return Regime::latency;
      }
      return Regime::throughput;
    }
    // TODO figure out 2
    if (bytes <= RED_LATENCY_BOUND_THRESHOLD) {
      return Regime::latency;
    }
    return Regime::throughput;
  }
  template<typename SutureAtom, DataLayout inputLayout, typename Element, typename BT>
  __device__ __forceinline__
  static void reduceLR(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& nextEpoch,
    const uint& senseBit) {
    const auto stagingPrefix = (senseBit * ctx.world * suture::PACKET_BUFFER_SIZE);
    constexpr auto bufferStride = suture::PACKET_BUFFER_SIZE;
    const auto rankOffset = ctx.rank * suture::PACKET_BUFFER_SIZE;
    auto* __restrict__ base = ctx.stagingLR[ctx.rank];
    auto* __restrict__ localStaging = base + stagingPrefix;
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(typedWorkspace);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      const auto peerBase = ctx.stagingLR[peer];
      staging[peer] = peerBase + (stagingPrefix + rankOffset);
    }
    __syncthreads();
    const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
    const LRArgs redArgs{
      .src = src,
      .staging = staging,
      .localStaging = localStaging,
      .dst = dst,
      .flag = nextEpoch,
      .bufferStride = bufferStride,
      .bytes = bytes,
      .blocks = blocks,
      .tIdx = static_cast<int>(tid),
      .world = ctx.world,
    };
    SutureAtom::template reduce<inputLayout>(redArgs, typedWorkspace);
    __syncthreads();
    markEpoch(ctx, bIdx, nextEpoch);
    markUnusedEpochs<SutureAtom>(ctx, blocks, blocks, nextEpoch, tid);
  }

  template<
    typename SutureAtom,
    int PUT_BLOCKS,
    DataLayout inputLayout,
    DataLayout outputLayout,
    typename Element,
    typename BT
  >
  __device__ __forceinline__
  static void reduceNonChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& nextEpoch,
    const size_t& stagingPrefix, const int& collBlocks) {
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    if (bIdx < PUT_BLOCKS) {
      const auto globalBytes = inputLayout == DataLayout::scattered ? bytes * ctx.world : bytes;
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      const auto [bytesPut, putStartOffset] = partition<PUT_BLOCKS, alignmentBytes>(globalBytes, bIdx);
      const auto* __restrict__ srcP = src + putStartOffset;
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + stagingPrefix;
      auto* __restrict__ dstP = dstBase + putStartOffset;
      SutureAtom::put(dstP, srcP, bytesPut, workspace);
      __syncthreads();
      if (threadIdx.x / WARP_SIZE == 0) {
        const auto laneId = threadIdx.x % WARP_SIZE;
        int shouldNotify = PUT_BLOCKS == 1 ? 1 : 0;
        if (!threadIdx.x) {
          cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
          shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == PUT_BLOCKS;
          if (shouldNotify) {
            s.store(0, cuda::memory_order_relaxed);
          }
        }
        __syncwarp();
        shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
        if (shouldNotify) {
          signalAllPeers(ctx.signals, ctx.rank, ctx.world, nextEpoch, laneId);
          __syncwarp();
        }
      }
      const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
      markEpoch(ctx, bIdx, nextEpoch);
      markUnusedEpochs<SutureAtom, PUT_BLOCKS>(ctx, collBlocks, nextEpoch, tid);
      return;
    }
    // reducers
    const auto reduceBIdx = bIdx - PUT_BLOCKS;
    const auto reduceBlocks = blocks - PUT_BLOCKS;
    const auto [bytesRed, redStartOffset] = partition<alignmentBytes>(bytes, reduceBlocks, reduceBIdx);
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(workspace + SutureAtom::RED_PIPELINE_SMEM_BYTES);
    auto* __restrict__ gatherSignals = reinterpret_cast<uint64_t**>(staging + MAX_RANKS_PER_DOMAIN);
    static_assert(sizeof(cuda::std::byte**) == sizeof(uint64_t**) && alignof(cuda::std::byte**) == alignof(uint64_t**));
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      const auto offset = stagingPrefix + redStartOffset;
      staging[peer] = ctx.staging[peer] + (offset + (inputLayout == DataLayout::scattered ? bytes * ctx.rank : 0));
      gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
    }
    cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
    const ReduceTRArgs redArgs{
      .sources = staging,
      .dst = dstP,
      .bytesRed = bytesRed,
      .world = ctx.world,
    };
    const auto warpId = threadIdx.x / WARP_SIZE;
    const auto laneId = threadIdx.x % WARP_SIZE;
    waitPeerArrivals<SutureAtom>(ctx.signals[ctx.rank], redArgs.world, nextEpoch);
    __syncthreads();
    SutureAtom::reduce(redArgs, typedWorkspace);
    if constexpr (outputLayout == DataLayout::scattered) {
      __syncthreads();
      // notify that chunk is done
      if (warpId == 0) {
        int shouldNotify = reduceBlocks == 1 ? 1 : 0;
        if (reduceBlocks > 1 && !laneId) {
          cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.redCounter};
          shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == reduceBlocks;
          if (shouldNotify) {
            s.store(0, cuda::memory_order_relaxed);
          }
        }
        __syncwarp();
        shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
        if (shouldNotify) {
          signalPointerList(gatherSignals, ctx.world, nextEpoch, laneId);
        }
      }
    }
    markEpoch(ctx, bIdx, nextEpoch);
  }

  template<
    typename SutureAtom,
    int PUT_BLOCKS,
    size_t CHUNK_SIZE,
    DataLayout inputLayout,
    DataLayout outputLayout,
    typename Element,
    typename BT
  >
  __device__ __forceinline__
  static void reduceChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& epoch,
    const size_t& stagingPrefix, const int& collBlocks) {
    static_assert(CHUNK_SIZE >= MIN_CHUNK_SIZE);
    const auto chunks = static_cast<int>(bytes / CHUNK_SIZE);
    const auto cutoff = CHUNK_SIZE * chunks;
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    //chunked-throughput regime
    if (bIdx < PUT_BLOCKS) {
      const auto blockSetSize = inputLayout == DataLayout::packed ? PUT_BLOCKS : PUT_BLOCKS / ctx.world;
      const auto peer = bIdx / blockSetSize;
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      auto flag = epoch;
      auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::COPY_PIPELINE_SMEM_BYTES);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
        signals[i] = ctx.signals[i] + ctx.rank;
      }
      __syncthreads();
      const auto intraBIdx = inputLayout == DataLayout::packed ? bIdx : bIdx % blockSetSize;
      const auto [bytesPut, putStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(blockSetSize, intraBIdx);
      const auto intraOffset = inputLayout == DataLayout::packed ? 0 : peer * bytes;
      const auto* __restrict__ srcP = src + (putStartOffset + intraOffset);
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + (stagingPrefix + intraOffset);
      auto* __restrict__ dstP = dstBase + putStartOffset;
      const auto laneId = threadIdx.x % WARP_SIZE;
      auto* __restrict__ putCounter = inputLayout == DataLayout::packed ?
      ctx.putCounter : ctx.putCounter + peer * MAX_CHUNKS;
      for (int chunk = 0; chunk < chunks; ++chunk) {
        SutureAtom::put(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        flag++;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = blockSetSize == 1 ? 1 : 0;
          if (!laneId && blockSetSize > 1) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunk)};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            if constexpr (inputLayout == DataLayout::packed) {
              signalPointerList(signals, ctx.world, flag, laneId);
            }
            else {
              if (!laneId) {
                signalOne(signals[peer], flag);
              }
            }
            __syncwarp();
          }
        }
        dstP += CHUNK_SIZE;
        srcP += CHUNK_SIZE;
      }
      if (bytes > cutoff) {
        const auto residue = bytes - cutoff;
        const auto [bytesPutLeft, putStartOffsetLeft] = partition<alignmentBytes>(residue, blockSetSize, intraBIdx);
        srcP = src + ((CHUNK_SIZE * chunks + putStartOffsetLeft) + intraOffset);
        dstP = dstBase + (CHUNK_SIZE * chunks + putStartOffsetLeft);
        SutureAtom::put(dstP, srcP, bytesPutLeft, workspace);
        __syncthreads();
        flag++;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = blockSetSize == 1 ? 1 : 0;
          if (!laneId && blockSetSize > 1) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(putCounter + chunks)};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            if constexpr (inputLayout == DataLayout::packed) {
              signalPointerList(signals, ctx.world, flag, laneId);
            }
            else {
              if (!laneId) {
                signalOne(signals[peer], flag);
              }
            }
            __syncwarp();
          }
        }
      }
      const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
      markEpoch(ctx, bIdx, flag);
      markUnusedEpochs<SutureAtom, PUT_BLOCKS>(ctx, collBlocks, flag, tid);
      return;
    }

    // reducer blocks
    const auto reduceBIdx = bIdx - PUT_BLOCKS;
    const auto reduceBlocks = blocks - PUT_BLOCKS;
    const auto [bytesRed, redStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(reduceBlocks, reduceBIdx);
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::RED_PIPELINE_SMEM_BYTES);
    auto* __restrict__ gatherSignals = signals + MAX_RANKS_PER_DOMAIN;
    static_assert(sizeof(cuda::std::byte**) == sizeof(uint64_t**) && alignof(cuda::std::byte**) == alignof(uint64_t**));
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(gatherSignals + MAX_RANKS_PER_DOMAIN);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      // signals
      signals[peer] = ctx.signals[ctx.rank] + peer;
      gatherSignals[peer] = ctx.gatherSignals[peer] + ctx.rank;
      // staging
      const auto offset = stagingPrefix + redStartOffset;
      staging[peer] = ctx.staging[peer] + (offset + (inputLayout == DataLayout::scattered ? bytes * ctx.rank : 0));
    }
    __syncthreads();
    auto flag = epoch;
    cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
    const auto warpId = threadIdx.x / WARP_SIZE;
    const auto laneId = threadIdx.x % WARP_SIZE;
    const auto tidS1 = (((warpId + (SutureAtom::WARPS - 1)) % SutureAtom::WARPS) * WARP_SIZE) + laneId;
    const auto tidS2 = SutureAtom::WARPS == 1 ? threadIdx.x :
    (((warpId + (SutureAtom::WARPS - 2)) % SutureAtom::WARPS) * WARP_SIZE) + laneId;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      flag++;
      const ReduceTRArgs redArgs{
        .sources = staging,
        .dst = dstP,
        .bytesRed = bytesRed,
        .world = ctx.world,
      };
      waitPointerList<SutureAtom>(signals, redArgs.world, flag, tidS2);
      __syncthreads();
      SutureAtom::reduce(redArgs, typedWorkspace);
      __syncthreads();
      if constexpr (outputLayout == DataLayout::scattered) {
        // notify that chunk is done
        if (warpId == 0) {
          int shouldNotify = reduceBlocks == 1 ? 1 : 0;
          if (reduceBlocks > 1 && !laneId) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.redCounter + chunk)};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == reduceBlocks;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            signalPointerList(gatherSignals, ctx.world, flag, laneId);
          }
        }
      }
      dstP += CHUNK_SIZE;
      for (int i = static_cast<int>(tidS1); i < ctx.world; i += SutureAtom::THREADS) {
        staging[i] += CHUNK_SIZE;
      }
    }
    if (bytes > cutoff) {
      flag++;
      const auto residue = bytes - cutoff;
      const auto [bytesRedLeft, redStartOffsetLeft] = partition<alignmentBytes>(residue, reduceBlocks, reduceBIdx);
      dstP = dst + (CHUNK_SIZE * chunks + redStartOffsetLeft);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
        auto* __restrict__ stagingBase = staging[i] - (chunks * CHUNK_SIZE + redStartOffset);
        staging[i] = stagingBase + (chunks * CHUNK_SIZE + redStartOffsetLeft);
      }
      const ReduceTRArgs redArgs{
        .sources = staging,
        .dst = dstP,
        .bytesRed = bytesRedLeft,
        .world = ctx.world,
      };
      waitPointerList<SutureAtom>(signals, redArgs.world, flag);
      __syncthreads();
      SutureAtom::reduce(redArgs, typedWorkspace);
      __syncthreads();
      if constexpr (outputLayout == DataLayout::scattered) {
        // notify that chunk is done
        if (warpId == 0) {
          int shouldNotify = reduceBlocks == 1 ? 1 : 0;
          if (reduceBlocks > 1 && !laneId) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.redCounter + chunks)};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == reduceBlocks;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            signalPointerList(gatherSignals, ctx.world, flag, laneId);
          }
        }
      }
    }
    markEpoch(ctx, bIdx, flag);
  }

  template<
    typename SutureAtom,
    typename CollConfig,
    typename Element,
    typename BT = int
  >
  __device__ __forceinline__
  static void reduceScatter(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    const auto epochState = makeEpochState(ctx, bIdx);
    static_assert(SutureAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    if constexpr (SutureAtom::REGIME == Regime::latency) {
      reduceLR<SutureAtom, DataLayout::scattered>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        reduceNonChunked<SutureAtom, CollConfig::PUT_BLOCKS, DataLayout::scattered, DataLayout::packed>
          (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.trStagingPrefix, blocks);
      }
      else {
        reduceChunked<SutureAtom, CollConfig::PUT_BLOCKS, CollConfig::CHUNK_SIZE, DataLayout::scattered, DataLayout::packed>
        (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.epoch, epochState.trStagingPrefix, blocks);
      }
    }
  }

  template<
    typename SutureAtom,
    typename CollConfig,
    typename Element,
    typename BT = int
  >
  __device__ __forceinline__
  static void allReduce(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    const auto epochState = makeEpochState(ctx, bIdx);
    static_assert(SutureAtom::REGIME == Regime::latency || !cuda::std::is_same_v<CollConfig, CollectiveConfigLR>);
    if constexpr (SutureAtom::REGIME == Regime::latency) {
      reduceLR<SutureAtom, DataLayout::packed>
      (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      const auto stagingPrefix = epochState.trStagingPrefix;
      // if (ctx.world == 2) {
      //   if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
      //     reduceNonChunked<
      //       SutureAtom,
      //       CollConfig::PUT_BLOCKS,
      //       DataLayout::packed,
      //       DataLayout::packed
      //     >
      //     (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, nextEpoch, stagingPrefix, blocks);
      //   }
      //   else {
      //     reduceChunked<
      //       SutureAtom,
      //       CollConfig::PUT_BLOCKS,
      //       CollConfig::CHUNK_SIZE,
      //       DataLayout::packed,
      //       DataLayout::packed
      //     >
      //     (dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epoch, stagingPrefix, blocks);
      //   }
      //   return;
      // }
      const auto localBytes = bytes / ctx.world_l;
      // RS+AG
      const auto reduceScatterBlocks = blocks - CollConfig::GATHER_BLOCKS;
      if (bIdx < reduceScatterBlocks) {
        auto* __restrict__ sDst = ctx.staging[ctx.rank] + (stagingPrefix + localBytes * ctx.rank);
        if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
          reduceNonChunked<
            SutureAtom,
            CollConfig::PUT_BLOCKS,
            DataLayout::scattered,
            DataLayout::scattered
          >
          (sDst, src, localBytes, typedWorkspace, ctx, reduceScatterBlocks, bIdx, epochState.nextEpoch, stagingPrefix, blocks);
        }
        else {
          reduceChunked<
            SutureAtom,
            CollConfig::PUT_BLOCKS,
            CollConfig::CHUNK_SIZE,
            DataLayout::scattered,
            DataLayout::scattered
          >
          (sDst, src, localBytes, typedWorkspace, ctx, reduceScatterBlocks, bIdx, epochState.epoch, stagingPrefix, blocks);
        }
        return;
      }
      // gather blocks
      const auto gBIdx = bIdx - reduceScatterBlocks;
      const auto blockSetSize = CollConfig::GATHER_BLOCKS / ctx.world;
      const auto peerBlock = mapPeerBlock(gBIdx, blockSetSize);
      auto* __restrict__ signal = ctx.gatherSignals[ctx.rank] + peerBlock.peer;
      const auto* __restrict__ srcBase = ctx.staging[peerBlock.peer] + (stagingPrefix + localBytes * peerBlock.peer);
      auto* __restrict__ dstBase = dst + localBytes * peerBlock.peer;
      const auto* __restrict__ srcP = srcBase;
      auto* __restrict__ dstP = dstBase;
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        if (!threadIdx.x) {
          waitUntilAtLeast(signal, epochState.nextEpoch);
        }
        __syncthreads();
        superGet<SutureAtom>(dstP, srcP, localBytes, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        markEpoch(ctx, bIdx, epochState.nextEpoch);
      }
      else {
        const auto chunks = static_cast<int>(localBytes / CollConfig::CHUNK_SIZE);
        const auto chunkCutoff = CollConfig::CHUNK_SIZE * chunks;
        auto flag = epochState.epoch;
        for (int i = 0; i < chunks; ++i) {
          flag++;
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          superGet<SutureAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
          srcP += CollConfig::CHUNK_SIZE;
          dstP += CollConfig::CHUNK_SIZE;
        }
        if (localBytes > chunkCutoff) {
          flag++;
          const auto residue = localBytes - chunkCutoff;
          dstP = dstBase + (CollConfig::CHUNK_SIZE * chunks);
          srcP = srcBase + (CollConfig::CHUNK_SIZE * chunks);
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          superGet<SutureAtom>(dstP, srcP, residue, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        }
        markEpoch(ctx, bIdx, flag);
      }
    }
  }

  template<typename SutureAtom, typename BT = int>
  __device__ __forceinline__
  static void gatherLR(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& nextEpoch,
    const uint& senseBit) {
    const auto stagingPrefix = (senseBit * ctx.world * suture::PACKET_BUFFER_SIZE);
    const auto isInPlace = src == (dst + ctx.rank * bytes);
    constexpr auto bufferStride = suture::PACKET_BUFFER_SIZE;
    const auto rankOffset = ctx.rank * suture::PACKET_BUFFER_SIZE;
    auto* __restrict__ localStaging = ctx.stagingLR[ctx.rank] + stagingPrefix;
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(workspace);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      staging[peer] = ctx.stagingLR[peer] + (stagingPrefix + rankOffset);
    }
    const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
    __syncthreads();
    const LRArgs gArgs{
      .src = src,
      .staging = staging,
      .localStaging = localStaging,
      .dst = dst,
      .flag = nextEpoch,
      .bufferStride = bufferStride,
      .bytes = bytes,
      .blocks = blocks,
      .tIdx = static_cast<int>(tid),
      .world = ctx.world,
      .rank = ctx.rank,
      .isInPlace = isInPlace,
    };
    fascia::gather<typename SutureAtom::BaseConfig>(gArgs);
    __syncthreads();
    markEpoch(ctx, bIdx, nextEpoch);
    markUnusedEpochs<SutureAtom>(ctx, blocks, blocks, nextEpoch, tid);
  }

  template<
    typename SutureAtom,
    typename CollConfig,
    typename BT = int
  >
  __device__ __forceinline__
  static void allGather(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    const auto epochState = makeEpochState(ctx, bIdx);
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;

    if constexpr (SutureAtom::REGIME == Regime::latency) {
      gatherLR<SutureAtom>(dst, src, bytes, workspace, ctx, blocks, bIdx, epochState.nextEpoch, epochState.senseBit);
    }
    else {
      const auto stagingPrefix = epochState.trStagingPrefix;
      const auto chunks = static_cast<int>(bytes / CollConfig::CHUNK_SIZE);
      const auto cutoff = CollConfig::CHUNK_SIZE * chunks;
      if constexpr (CollConfig::COLLECTIVE_TYPE == CollectiveType::nonChunked) {
        if (bIdx < CollConfig::PUT_BLOCKS) {
          // no chunking
          const auto [bytesPut, putStartOffset] = partition<CollConfig::PUT_BLOCKS, alignmentBytes>(bytes, bIdx);
          const auto* __restrict__ srcP = src + putStartOffset;
          auto* __restrict__ dstBase = ctx.staging[ctx.rank] + stagingPrefix;
          auto* __restrict__ dstP = dstBase + putStartOffset;
          SutureAtom::put(dstP, srcP, bytesPut, workspace);
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
          const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
          markEpoch(ctx, bIdx, epochState.nextEpoch);
          markUnusedEpochs<SutureAtom, CollConfig::PUT_BLOCKS>(ctx, blocks, epochState.nextEpoch, tid);
          return;
        }
        // consumers
        const auto blockSetSize = static_cast<int>(blocks - CollConfig::PUT_BLOCKS) / ctx.world;
        const auto cBIdx = bIdx - CollConfig::PUT_BLOCKS;
        const auto peerBlock = mapPeerBlock(cBIdx, blockSetSize);
        const auto* __restrict__ srcP = ctx.staging[peerBlock.peer] + stagingPrefix;
        auto* __restrict__ dstP = dst + bytes * peerBlock.peer;
        // wait for peer to set signal
        if (!threadIdx.x) {
          waitUntilAtLeast(ctx.signals[ctx.rank] + peerBlock.peer, epochState.nextEpoch);
        }
        __syncthreads();
        superGet<SutureAtom>(dstP, srcP, bytes, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        markEpoch(ctx, bIdx, epochState.nextEpoch);
      }
      else {
        static_assert(CollConfig::CHUNK_SIZE >= MIN_CHUNK_SIZE);
        if (bIdx < CollConfig::PUT_BLOCKS) {
          auto flag = epochState.epoch;
          auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::COPY_PIPELINE_SMEM_BYTES);
          for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
            signals[i] = ctx.signals[i] + ctx.rank;
          }
          __syncthreads();
          const auto [bytesPut, putStartOffset] = partition<CollConfig::CHUNK_SIZE, CollConfig::PUT_BLOCKS, alignmentBytes>(bIdx);
          const auto* __restrict__ srcP = src + putStartOffset;
          auto* __restrict__ dstBase = ctx.staging[ctx.rank] + stagingPrefix;
          auto* __restrict__ dstP = dstBase + putStartOffset;
          const auto laneId = threadIdx.x % WARP_SIZE;
          for (int chunk = 0; chunk < chunks; ++chunk) {
            SutureAtom::put(dstP, srcP, bytesPut, workspace);
            __syncthreads();
            flag++;
            if (threadIdx.x / WARP_SIZE == 0) {
              int shouldNotify = CollConfig::PUT_BLOCKS == 1 ? 1 : 0;
              if (CollConfig::PUT_BLOCKS > 1 && !laneId) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.putCounter + chunk)};
                shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == CollConfig::PUT_BLOCKS;
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
            const long int scaledChunkSizeLeft = residue / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
            const auto ctaBasePutChunkLeft = scaledChunkSizeLeft / CollConfig::PUT_BLOCKS;
            const auto ctaPutResidueLeft = static_cast<int>(scaledChunkSizeLeft % CollConfig::PUT_BLOCKS);
            const auto ctaPutChunkLeft = ctaBasePutChunkLeft + (bIdx < ctaPutResidueLeft);
            const auto putOffsetElemsLeft = ctaBasePutChunkLeft * bIdx + min(bIdx, ctaPutResidueLeft);
            const auto putStartOffsetLeft = putOffsetElemsLeft * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
            const size_t bytesPutLeft = static_cast<size_t>(ctaPutChunkLeft) * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;

            srcP = src + (CollConfig::CHUNK_SIZE * chunks + putStartOffsetLeft);
            dstP = dstBase + (CollConfig::CHUNK_SIZE * chunks + putStartOffsetLeft);
            SutureAtom::put(dstP, srcP, bytesPutLeft, workspace);
            __syncthreads();
            flag++;
            if (threadIdx.x / WARP_SIZE == 0) {
              int shouldNotify = CollConfig::PUT_BLOCKS == 1 ? 1 : 0;
              if (!threadIdx.x) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*(ctx.putCounter + chunks)};
                shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == CollConfig::PUT_BLOCKS;
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
          const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
          markEpoch(ctx, bIdx, flag);
          markUnusedEpochs<SutureAtom, CollConfig::PUT_BLOCKS>(ctx, blocks, flag, tid);
          return;
        }
        // consumers
        const auto blockSetSize = static_cast<int>(blocks - CollConfig::PUT_BLOCKS) / ctx.world;
        const auto cBIdx = bIdx - CollConfig::PUT_BLOCKS;
        const auto peerBlock = mapPeerBlock(cBIdx, blockSetSize);
        const auto* __restrict__ const srcBase = ctx.staging[peerBlock.peer] + stagingPrefix;
        const auto* __restrict__ srcP = srcBase;
        auto* __restrict__ const dstBase = dst + (bytes * peerBlock.peer);
        auto* __restrict__ dstP = dstBase;
        auto* __restrict__ signal = ctx.signals[ctx.rank] + peerBlock.peer;
        auto flag = epochState.epoch;
        for (int chunk = 0; chunk < chunks; ++chunk) {
          flag++;
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          superGet<SutureAtom, CollConfig::CHUNK_SIZE>(dstP, srcP, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
          srcP += CollConfig::CHUNK_SIZE;
          dstP += CollConfig::CHUNK_SIZE;
        }
        if (bytes > cutoff) {
          flag++;
          const auto residue = bytes - cutoff;
          dstP = dstBase + (CollConfig::CHUNK_SIZE * chunks);
          srcP = srcBase + (CollConfig::CHUNK_SIZE * chunks);
          if (!threadIdx.x) {
            waitUntilAtLeast(signal, flag);
          }
          __syncthreads();
          superGet<SutureAtom>(dstP, srcP, residue, workspace, peerBlock.blockSetSize, peerBlock.intraIdx);
        }
        markEpoch(ctx, bIdx, flag);
      }
    }
  }
}
#endif //SUTURE_COLLECTIVE_CUH
