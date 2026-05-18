//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_COLLECTIVE_CUH
#define SUTURE_COLLECTIVE_CUH
#include "base.cuh"
#include "context.cuh"
#include "packet.cuh"

namespace suture {
  struct PartitionResult {
    const size_t bytes;
    const size_t startOffset;
  };
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
  Regime getRedRegime(const size_t& bytes) {
    if (bytes <= RED_LATENCY_BOUND_THRESHOLD) {
      return Regime::latency;
    }
    return Regime::throughput;
  }
  __host__ __forceinline__
  Regime getGatherRegime(const size_t& bytes) {
    if (bytes <= AG_LATENCY_BOUND_THRESHOLD) {
      return Regime::latency;
    }
    return Regime::throughput;
  }
  template<typename SutureAtom, InputLayout iLayout, typename Element, typename BT>
  __device__ __forceinline__
  static void reduceLR0(cuda::std::byte* __restrict__ const& dst,
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
    auto* __restrict__ localStaging = ctx.stagingLR[ctx.rank] + stagingPrefix;
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(typedWorkspace);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      staging[peer] = ctx.stagingLR[peer] + (stagingPrefix + rankOffset);
    }
    const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
    __syncthreads();
    const LLArgs redArgs{
      .src = src,
      .staging = staging,
      .localStaging = localStaging,
      .dst = dst,
      .flag = nextEpoch,
      .bufferStride = bufferStride,
      .bytes = bytes,
      .blocks = blocks,
      .tIdx = static_cast<int>(tid),
      .world = ctx.world
    };
    SutureAtom::template reduce<iLayout>(redArgs, typedWorkspace);
    __syncthreads();
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = nextEpoch;
    }
    const auto leftover = suture::MAX_NUM_CTAS - blocks;
    auto* __restrict__ epochs = ctx.epochs + blocks;
    for (int i = tid; i < leftover; i += (SutureAtom::THREADS * blocks)) {
      epochs[i] = nextEpoch;
    }
  }

  template<typename SutureAtom, InputLayout iLayout, typename Element, typename BT>
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
    const auto isPutBlock = bIdx < ctx.maxPutBlocks;
    const int superBlockIdx = static_cast<int>(bIdx / ctx.superBlockSize);
    const int intraIdx = static_cast<int>(bIdx % ctx.superBlockSize);
    const auto peer = (superBlockIdx + ctx.rank + 1) % ctx.world;
    cuda::std::byte* __restrict__ dstP = nullptr;
    const cuda::std::byte* __restrict__ srcPut = nullptr;
    const cuda::std::byte* __restrict__ srcPutLocal = nullptr;
    size_t bytesPut = 0, bytesRed = 0;
    cuda::std::byte* __restrict__ stagingPut = nullptr;
    cuda::std::byte* __restrict__ stagingPutLocal = nullptr;
    cuda::std::byte* __restrict__ stagingRed = nullptr;
    constexpr auto dAB = sizeof(LRP16::RT); // data alignment bytes
    const auto scaledChunkSize = static_cast<long int>(bytes / dAB);
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
      auto* __restrict__ staging = ctx.stagingLR[peer] + stagingOffset;

      bytesPut = ctaChunk * dAB;
      srcPut = src + (startOffset + (iLayout == InputLayout::scattered ? peer * bytes : 0));
      const auto rankOffset = (ctx.rank * suture::PACKET_BUFFER_SIZE);
      stagingPut = staging + rankOffset;
      if (superBlockIdx == 0) {
        srcPutLocal = src + (startOffset + (iLayout == InputLayout::scattered ? ctx.rank * bytes : 0));
        stagingPutLocal = ctx.stagingLR[ctx.rank] + stagingOffset + rankOffset;
      }
    }
    // reduction offsets
    const auto ctaBaseRedChunk = scaledChunkSize / blocks;
    const auto ctaRedResidue = static_cast<int>(scaledChunkSize % blocks);
    const auto ctaRedChunk = ctaBaseRedChunk + (bIdx < ctaRedResidue);
    const auto redOffsetElems = ctaBaseRedChunk * bIdx + min(bIdx, ctaRedResidue);
    const auto redStartOffset = redOffsetElems * dAB;

    bytesRed = ctaRedChunk * dAB;
    dstP = dst + redStartOffset;
    stagingRed = ctx.stagingLR[ctx.rank] + (stagingPrefix + (redOffsetElems * pAB));

    const LRArgs redArgs{
      .dst = dstP,
      .srcPut = srcPut,
      .srcPutLocal = srcPutLocal,
      .stagingPut = stagingPut,
      .stagingPutLocal = stagingPutLocal,
      .stagingGet = stagingRed,
      .flag = nextEpoch,
      .bytesPut = bytesPut,
      .bytesRed = bytesRed,
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
  }

  template<typename SutureAtom, InputLayout iLayout, typename Element, typename BT>
  __device__ __forceinline__
  static void reduceNonChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& nextEpoch,
    const uint& senseBit) {
    const auto stagingPrefix = STAGING_BUFFER_SIZE_ * senseBit;
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    if (bIdx < RED_PUT_BLOCKS) {
      const auto globalBytes = bytes * ctx.world;
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      const auto [bytesPut, putStartOffset] = partition<RED_PUT_BLOCKS, alignmentBytes>(globalBytes, bIdx);
      const auto* __restrict__ srcP = src + putStartOffset;
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + stagingPrefix;
      auto* __restrict__ dstP = dstBase + putStartOffset;
      SutureAtom::put(dstP, srcP, bytesPut, workspace);
      __syncthreads();
      if (threadIdx.x / WARP_SIZE == 0) {
        int shouldNotify = RED_PUT_BLOCKS == 1 ? 1 : 0;
        if (!threadIdx.x) {
          cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
          shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == RED_PUT_BLOCKS;
          if (shouldNotify) {
            s.store(0, cuda::memory_order_relaxed);
          }
        }
        __syncwarp();
        shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
        if (shouldNotify) {
          for (int i = static_cast<int>(threadIdx.x % WARP_SIZE); i < ctx.world; i += WARP_SIZE) {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> signal{*(ctx.signals[i] + ctx.rank)};
            signal.store(nextEpoch, cuda::std::memory_order_release);
          }
          __syncwarp();
        }
      }
      if (!threadIdx.x) {
        ctx.epochs[bIdx] = nextEpoch;
      }
      const auto leftover = suture::MAX_NUM_CTAS - blocks;
      const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
      auto* __restrict__ epochs = ctx.epochs + blocks;
      for (int i = tid; i < leftover; i += (SutureAtom::THREADS * RED_PUT_BLOCKS)) {
        epochs[i] = nextEpoch;
      }
      return;
    }
    // reducers
    const auto reduceBIdx = bIdx - RED_PUT_BLOCKS;
    const auto reduceBlocks = blocks - RED_PUT_BLOCKS;
    const auto [bytesRed, redStartOffset] = partition<alignmentBytes>(bytes, reduceBlocks, reduceBIdx);
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(workspace + SutureAtom::RED_PIPELINE_SMEM_BYTES);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      const auto offset = stagingPrefix + redStartOffset;
      staging[peer] = ctx.staging[peer] + (offset + (iLayout == InputLayout::scattered ? bytes * ctx.rank : 0));
    }
    cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
    const ReduceTRArgs redArgs{
      .sources = staging,
      .dst = dstP,
      .bytesRed = bytesRed,
      .world = ctx.world,
    };
    for (int peer = static_cast<int>(threadIdx.x); peer < redArgs.world; peer += SutureAtom::THREADS) {
      auto* __restrict__ signal = ctx.signals[ctx.rank] + peer;
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
      auto isHere = sig.load(cuda::memory_order_relaxed) >= nextEpoch;
      while (!isHere) {
        isHere = sig.load(cuda::memory_order_relaxed) >= nextEpoch;
      }
      cuda::std::ignore = sig.load(cuda::memory_order_acquire);
    }
    __syncthreads();
    SutureAtom::reduce(redArgs, typedWorkspace);
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = nextEpoch;
    }
  }

  template<typename SutureAtom, int CHUNK_SIZE, InputLayout iLayout, typename Element, typename BT>
  __device__ __forceinline__
  static void reduceChunked(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx,
    const uint64_t& epoch,
    const uint& senseBit) {
    // assert(RED_PUT_BLOCKS % world == 0);
    const auto stagingPrefix = STAGING_BUFFER_SIZE_ * senseBit;
    const auto chunks = static_cast<int>(bytes / CHUNK_SIZE);
    const auto cutoff = CHUNK_SIZE * chunks;
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    //chunked-throughput regime
    if (bIdx < RED_PUT_BLOCKS) {
      const auto blockSetSize = iLayout == InputLayout::packed ? RED_PUT_BLOCKS : RED_PUT_BLOCKS / ctx.world;
      const auto peer = bIdx / blockSetSize;
      auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
      auto flag = epoch;
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> sense{*(ctx.groupSense + peer)};
      uint32_t localSense = sense.load(cuda::memory_order_relaxed);
      auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::COPY_PIPELINE_SMEM_BYTES);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
        signals[i] = ctx.signals[i] + ctx.rank;
      }
      __syncthreads();
      const auto intraBIdx = iLayout == InputLayout::packed ? bIdx : bIdx % blockSetSize;
      const auto [bytesPut, putStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(blockSetSize, intraBIdx);
      const auto intraOffset = iLayout == InputLayout::packed ? 0 : peer * bytes;
      const auto* __restrict__ srcP = src + (putStartOffset + intraOffset);
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + (stagingPrefix + intraOffset);
      auto* __restrict__ dstP = dstBase + putStartOffset;
      const auto laneId = threadIdx.x % WARP_SIZE;
      auto* __restrict__ putCounter = iLayout == InputLayout::packed ? ctx.putCounter : ctx.putCounter + peer;
      for (int chunk = 0; chunk < chunks; ++chunk) {
        SutureAtom::put(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        flag++;
        const uint32_t nextSense = localSense ^ 1;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = blockSetSize == 1 ? 1 : 0;
          if (!laneId && blockSetSize > 1) {
            // wait until groupSense matches localSense
            bool canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            while (!canProceed) {
              canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            }
            cuda::std::ignore = sense.load(cuda::memory_order_acquire);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
              sense.store(nextSense, cuda::memory_order_release);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            if constexpr (iLayout == InputLayout::packed) {
              for (int i = static_cast<int>(threadIdx.x % WARP_SIZE); i < ctx.world; i += WARP_SIZE) {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_system> signal{*signals[i]};
                signal.store(flag, cuda::std::memory_order_release);
              }
            }
            else {
              if (!laneId) {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_system> signal{*signals[peer]};
                signal.store(flag, cuda::std::memory_order_release);
              }
            }
            __syncwarp();
          }
        }
        localSense = nextSense;
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
        uint32_t nextSense = localSense ^ 1;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = blockSetSize == 1 ? 1 : 0;
          if (!laneId && blockSetSize > 1) {
            // wait until groupSense matches localSense
            bool canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            while (!canProceed) {
              canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            }
            cuda::std::ignore = sense.load(cuda::memory_order_acquire);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blockSetSize;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
              sense.store(nextSense, cuda::memory_order_release);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            if constexpr (iLayout == InputLayout::packed) {
              for (int i = static_cast<int>(threadIdx.x % WARP_SIZE); i < ctx.world; i += WARP_SIZE) {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_system> signal{*signals[i]};
                signal.store(flag, cuda::std::memory_order_release);
              }
            }
            else {
              if (!laneId) {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_system> signal{*signals[peer]};
                signal.store(flag, cuda::std::memory_order_release);
              }
            }
            __syncwarp();
          }
        }
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

    // reducer blocks
    const auto reduceBIdx = bIdx - RED_PUT_BLOCKS;
    const auto reduceBlocks = blocks - RED_PUT_BLOCKS;
    const auto [bytesRed, redStartOffset] = partition<CHUNK_SIZE, alignmentBytes>(reduceBlocks, reduceBIdx);
    auto* __restrict__ workspace = reinterpret_cast<cuda::std::byte*>(typedWorkspace);
    auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::RED_PIPELINE_SMEM_BYTES);
    static_assert(sizeof(cuda::std::byte**) == sizeof(uint64_t**) && alignof(cuda::std::byte**) == alignof(uint64_t**));
    auto* __restrict__ staging = reinterpret_cast<cuda::std::byte**>(signals + MAX_RANKS_PER_DOMAIN);
    for (int peer = static_cast<int>(threadIdx.x); peer < ctx.world; peer += SutureAtom::THREADS) {
      // signals
      signals[peer] = ctx.signals[ctx.rank] + peer;
      // staging
      const auto offset = stagingPrefix + redStartOffset;
      staging[peer] = ctx.staging[peer] + (offset + (iLayout == InputLayout::scattered ? bytes * ctx.rank : 0));
    }
    auto flag = epoch;
    cuda::std::byte* __restrict__ dstP = dst + redStartOffset;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      flag++;
      const ReduceTRArgs redArgs{
        .sources = staging,
        .dst = dstP,
        .bytesRed = bytesRed,
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
      SutureAtom::reduce(redArgs, typedWorkspace);
      __syncthreads();
      dstP += CHUNK_SIZE;
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
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
      SutureAtom::reduce(redArgs, typedWorkspace);
      __syncthreads();
    }
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = flag;
    }
  }

  template<typename SutureAtom, int CHUNK_SIZE, InputLayout iLayout, typename Element, typename BT>
  __device__ __forceinline__
  static void reduce(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks,
    const int& bIdx) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    // Assumptions
    // assert(blocks <= suture::MAX_NUM_CTAS);
    // assert(ctx.world > 1)
    const auto epoch = ctx.epochs[bIdx];
    const auto nextEpoch = epoch + static_cast<uint64_t>(1);
    const auto senseBit = static_cast<uint>(epoch % 2);
    if constexpr (SutureAtom::REGIME == Regime::latency) {
      reduceLR0<SutureAtom, iLayout>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, nextEpoch, senseBit);
    }
    else {
      // throughput regime
      if (bytes <= CHUNK_SIZE) {
        reduceNonChunked<SutureAtom, iLayout>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, nextEpoch, senseBit);
        return;
      }
      reduceChunked<SutureAtom, CHUNK_SIZE, iLayout>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx, epoch, senseBit);
    }
  }

  template<typename SutureAtom, typename Element, typename BT = int>
  __device__ __forceinline__
  static void reduceScatter(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    reduce<SutureAtom, RS_CHUNK_SIZE, InputLayout::scattered>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx);
  }

  template<typename SutureAtom, typename Element, typename BT = int>
  __device__ __forceinline__
  static void allReduce(
    cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    Element* __restrict__ const& typedWorkspace, // shared
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    reduce<SutureAtom, RED_CHUNK_SIZE, InputLayout::packed>(dst, src, bytes, typedWorkspace, ctx, blocks, bIdx);
  }

  template<typename SutureAtom, typename BT = int>
  __device__ __forceinline__
  static void allGather(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    cuda::std::byte* __restrict__ const& workspace,
    const Context& ctx,
    const BT& blocks = static_cast<int>(gridDim.x),
    const int& bIdx = static_cast<int>(blockIdx.x)) {
    static_assert(cuda::std::is_same_v<BT, cuda::fast_mod_div<long int>> || cuda::std::is_same_v<BT, int>);
    const auto epoch = ctx.epochs[bIdx];
    constexpr auto alignmentBytes = SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto senseBit = static_cast<uint>(epoch % 2);
    const auto isInPlace = src == (dst + ctx.rank * bytes);

    if (bytes <= AG_LATENCY_BOUND_THRESHOLD) {
      const auto nextEpoch = epoch + static_cast<uint64_t>(1);
      const auto isPutBlock = bIdx < ctx.maxPutBlocks;
      const auto superBlockIdx = static_cast<int>(bIdx / ctx.superBlockSize);
      const auto intraIdx = static_cast<int>(bIdx % ctx.superBlockSize);
      const auto peer = (superBlockIdx + ctx.rank + 1) % ctx.world;
      cuda::std::byte* __restrict__ dstP = nullptr;
      const cuda::std::byte* __restrict__ srcPut = nullptr;
      const cuda::std::byte* __restrict__ srcPutLocal = nullptr;
      size_t bytesPut = 0, bytesRed = 0;
      cuda::std::byte* __restrict__ stagingPut = nullptr;
      cuda::std::byte* __restrict__ stagingPutLocal = nullptr;
      cuda::std::byte* __restrict__ stagingRed = nullptr;
      constexpr auto dAB = sizeof(LRP16::RT); // data alignment bytes
      const auto scaledChunkSize = static_cast<long int>(bytes / dAB);
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
        auto* __restrict__ staging = ctx.stagingLR[peer] + stagingOffset;

        bytesPut = ctaChunk * dAB;
        srcPut = src + startOffset;
        const auto rankOffset = (ctx.rank * suture::PACKET_BUFFER_SIZE);
        stagingPut = staging + rankOffset;
        if (!isInPlace && superBlockIdx == 0) {
          srcPutLocal = src + startOffset;
          stagingPutLocal = ctx.stagingLR[ctx.rank] + stagingOffset + rankOffset;
        }
      }
      // reduction offsets
      const auto ctaBaseChunk = scaledChunkSize / blocks;
      const auto ctaResidue = static_cast<int>(scaledChunkSize % blocks);
      const auto ctaChunk = ctaBaseChunk + (bIdx < ctaResidue);
      const auto offsetElems = ctaBaseChunk * bIdx + cute::min(bIdx, ctaResidue);
      const auto startOffset = offsetElems * dAB;

      bytesRed = ctaChunk * dAB;
      dstP = dst + startOffset;
      stagingRed = ctx.stagingLR[ctx.rank] + (stagingPrefix + (offsetElems * pAB));

      const LRArgs gatherArgs{
        .dst = dstP,
        .srcPut = srcPut,
        .srcPutLocal = srcPutLocal,
        .stagingPut = stagingPut,
        .stagingPutLocal = stagingPutLocal,
        .stagingGet = stagingRed,
        .flag = nextEpoch,
        .bytesPerPeer = bytes,
        .bytesPut = bytesPut,
        .bytesRed = bytesRed,
        .putBlock = isPutBlock,
        .world = ctx.world,
        .isInPlace = isInPlace,
        .rank = ctx.rank
      };
      fascia::gather<typename SutureAtom::BaseConfig>(gatherArgs);
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
    const auto stagingPrefix = STAGING_BUFFER_SIZE_ * senseBit;
    const auto chunks = static_cast<int>(bytes / AG_CHUNK_SIZE);
    const auto cutoff = AG_CHUNK_SIZE * chunks;
    if (bytes <= AG_CHUNK_SIZE) {
      const auto nextEpoch = epoch + 1;
      if (bIdx < AG_PUT_BLOCKS) {
        // no chunking
        const auto [bytesPut, putStartOffset] = partition<AG_PUT_BLOCKS, alignmentBytes>(bytes, bIdx);
        const auto* __restrict__ srcP = src + putStartOffset;
        auto* __restrict__ dstBase = ctx.staging[ctx.rank] + stagingPrefix;
        auto* __restrict__ dstP = dstBase + putStartOffset;
        SutureAtom::put(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = AG_PUT_BLOCKS == 1 ? 1 : 0;
          if (!threadIdx.x) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == AG_PUT_BLOCKS;
            if (shouldNotify) {
              s.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncwarp();
          shouldNotify = __shfl_sync(0xffffffff, shouldNotify, 0);
          if (shouldNotify) {
            for (int i = static_cast<int>(threadIdx.x % WARP_SIZE); i < ctx.world; i += WARP_SIZE) {
              cuda::atomic_ref<uint64_t, cuda::thread_scope_system> signal{*(ctx.signals[i] + ctx.rank)};
              signal.store(nextEpoch, cuda::std::memory_order_release);
            }
            __syncwarp();
          }
        }
        if (!threadIdx.x) {
          ctx.epochs[bIdx] = nextEpoch;
        }
        const auto leftover = suture::MAX_NUM_CTAS - blocks;
        auto* __restrict__ epochs = ctx.epochs + blocks;
        const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
        for (int i = tid; i < leftover; i += (SutureAtom::THREADS * AG_PUT_BLOCKS)) {
          epochs[i] = nextEpoch;
        }
        return;
      }
      // consumers
      const auto consumerBIdx = bIdx - AG_PUT_BLOCKS;
      const int superBlockIdx = static_cast<int>(consumerBIdx / ctx.superBlockSize);
      const int intraIdx = static_cast<int>(consumerBIdx % ctx.superBlockSize);
      const auto peer = superBlockIdx;
      const auto* __restrict__ srcP = ctx.staging[peer] + stagingPrefix;
      auto* __restrict__ dstP = dst + bytes * peer;
      // wait for peer to set signal
      if (!threadIdx.x) {
        auto* __restrict__ signal = ctx.signals[ctx.rank] + peer;
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
        auto isHere = sig.load(cuda::memory_order_relaxed) >= nextEpoch;
        while (!isHere) {
          isHere = sig.load(cuda::memory_order_relaxed) >= nextEpoch;
        }
        cuda::std::ignore = sig.load(cuda::memory_order_acquire);
      }
      __syncthreads();
      superGet<SutureAtom>(dstP, srcP, bytes, workspace, ctx.superBlockSize, intraIdx);
      if (!threadIdx.x) {
        ctx.epochs[bIdx] = nextEpoch;
      }
      return;
    }
    if (bIdx < AG_PUT_BLOCKS) {
      auto flag = epoch;
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> sense{*ctx.groupSense};
      uint32_t localSense = sense.load(cuda::memory_order_relaxed);
      auto* __restrict__ signals = reinterpret_cast<uint64_t**>(workspace + SutureAtom::COPY_PIPELINE_SMEM_BYTES);
      for (int i = static_cast<int>(threadIdx.x); i < ctx.world; i += SutureAtom::THREADS) {
        signals[i] = ctx.signals[i] + ctx.rank;
      }
      __syncthreads();
      const auto [bytesPut, putStartOffset] = partition<AG_CHUNK_SIZE, AG_PUT_BLOCKS, alignmentBytes>(bIdx);
      const auto* __restrict__ srcP = src + putStartOffset;
      auto* __restrict__ dstBase = ctx.staging[ctx.rank] + stagingPrefix;
      auto* __restrict__ dstP = dstBase + putStartOffset;
      const auto laneId = threadIdx.x % WARP_SIZE;
      for (int chunk = 0; chunk < chunks; ++chunk) {
        SutureAtom::put(dstP, srcP, bytesPut, workspace);
        __syncthreads();
        flag++;
        uint32_t nextSense = localSense ^ 1;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = AG_PUT_BLOCKS == 1 ? 1 : 0;
          if (AG_PUT_BLOCKS > 1 && !laneId) {
            // wait until groupSense matches localSense
            bool canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            while (!canProceed) {
              canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            }
            cuda::std::ignore = sense.load(cuda::memory_order_acquire);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == AG_PUT_BLOCKS;
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
        localSense = nextSense;
        dstP += AG_CHUNK_SIZE;
        srcP += AG_CHUNK_SIZE;
      }
      if (bytes > cutoff) {
        const auto residue = bytes - cutoff;
        const long int scaledChunkSizeLeft = residue / SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        const auto ctaBasePutChunkLeft = scaledChunkSizeLeft / AG_PUT_BLOCKS;
        const auto ctaPutResidueLeft = static_cast<int>(scaledChunkSizeLeft % AG_PUT_BLOCKS);
        const auto ctaPutChunkLeft = ctaBasePutChunkLeft + (bIdx < ctaPutResidueLeft);
        const auto putOffsetElemsLeft = ctaBasePutChunkLeft * bIdx + min(bIdx, ctaPutResidueLeft);
        const auto putStartOffsetLeft = putOffsetElemsLeft * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;
        const size_t bytesPutLeft = static_cast<size_t>(ctaPutChunkLeft) * SutureAtom::GMEM_ACCESS_ALIGNMENT_BYTES;

        srcP = src + (AG_CHUNK_SIZE * chunks + putStartOffsetLeft);
        dstP = dstBase + (AG_CHUNK_SIZE * chunks + putStartOffsetLeft);
        SutureAtom::put(dstP, srcP, bytesPutLeft, workspace);
        __syncthreads();
        flag++;
        uint32_t nextSense = localSense ^ 1;
        if (threadIdx.x / WARP_SIZE == 0) {
          int shouldNotify = AG_PUT_BLOCKS == 1 ? 1 : 0;
          if (!threadIdx.x) {
            // wait until groupSense matches localSense
            bool canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            while (!canProceed) {
              canProceed = sense.load(cuda::memory_order_relaxed) == localSense;
            }
            cuda::std::ignore = sense.load(cuda::memory_order_acquire);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> s{*ctx.putCounter};
            shouldNotify = s.fetch_add(1, cuda::memory_order_acq_rel) + 1 == AG_PUT_BLOCKS;
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
      }
      if (!threadIdx.x) {
        ctx.epochs[bIdx] = flag;
      }
      const auto leftover = suture::MAX_NUM_CTAS - blocks;
      auto* __restrict__ epochs = ctx.epochs + blocks;
      const auto tid = bIdx * SutureAtom::THREADS + threadIdx.x;
      for (int i = tid; i < leftover; i += (SutureAtom::THREADS * AG_PUT_BLOCKS)) {
        epochs[i] = flag;
      }
      return;
    }
    // consumers
    const auto consumerBIdx = bIdx - AG_PUT_BLOCKS;
    const int superBlockIdx = static_cast<int>(consumerBIdx / ctx.superBlockSize);
    const int intraIdx = static_cast<int>(consumerBIdx % ctx.superBlockSize);
    const auto peer = superBlockIdx;
    const auto* __restrict__ const srcBase = ctx.staging[peer] + stagingPrefix;
    const auto* __restrict__ srcP = srcBase;
    auto* __restrict__ const dstBase = dst + (bytes * peer);
    auto* __restrict__ dstP = dstBase;
    auto* __restrict__ signal = ctx.signals[ctx.rank] + peer;
    auto flag = epoch;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      flag++;
      if (!threadIdx.x) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
        auto isHere = sig.load(cuda::memory_order_relaxed) >= flag;
        while (!isHere) {
          isHere = sig.load(cuda::memory_order_relaxed) >= flag;
        }
        cuda::std::ignore = sig.load(cuda::memory_order_acquire);
      }
      __syncthreads();
      superGet<SutureAtom, AG_CHUNK_SIZE>(dstP, srcP, workspace, ctx.superBlockSize, intraIdx);
      srcP += AG_CHUNK_SIZE;
      dstP += AG_CHUNK_SIZE;
    }
    if (bytes > cutoff) {
      flag++;
      const auto residue = bytes - cutoff;
      dstP = dstBase + (AG_CHUNK_SIZE * chunks);
      srcP = srcBase + (AG_CHUNK_SIZE * chunks);
      if (!threadIdx.x) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sig{*signal};
        auto isHere = sig.load(cuda::memory_order_relaxed) >= flag;
        while (!isHere) {
          isHere = sig.load(cuda::memory_order_relaxed) >= flag;
        }
        cuda::std::ignore = sig.load(cuda::memory_order_acquire);
      }
      __syncthreads();
      superGet<SutureAtom>(dstP, srcP, residue, workspace, ctx.superBlockSize, intraIdx);
    }
    if (!threadIdx.x) {
      ctx.epochs[bIdx] = flag;
    }
  }
}
#endif //SUTURE_COLLECTIVE_CUH
