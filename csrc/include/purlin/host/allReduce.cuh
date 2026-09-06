#ifndef PURLIN_ALLREDUCE_CUH
#define PURLIN_ALLREDUCE_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "codesign.cuh"
namespace purlin {
  template<typename PurlinAtom, typename Element, typename CollConfig,
    World2Bypass wb = World2Bypass::unknown, LRMode mode = LRMode::fullBuffer,
    ReduceOp ro = ReduceOp::add>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void allReduceKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx) {
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    const SnacArgs<cuda::fast_mod_div<long int>> args{
      .dst = kArgs.dst,
      .src = kArgs.src,
      .bytes = kArgs.bytes,
      .workspace = workspace,
      .blocks = kArgs.blocks,
      .collBlocks = static_cast<int>(kArgs.blocks),
    };
    purlin::allReduce<PurlinAtom, CollConfig, wb, mode, ro, Element>(args, ctx);
  }

  template<typename PurlinAtom, typename Element, LRMode mode = LRMode::fullBuffer, ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void launchAllReduceLatency(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const int blocks, cudaStream_t stream) {
    constexpr auto kS = 0;
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    allReduceKernel<PurlinAtom, Element, CollectiveConfigLR, World2Bypass::unknown, mode, ro>
      <<<blocks, PurlinAtom::THREADS, kS, stream>>>(kArgs, ctx);
  }

  template<typename PurlinAtom, typename Element, typename CollConfig, World2Bypass Bypass,
    ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void launchAllReduceThroughput(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const int& gatherBlocks, const int& maxReduceBlocks, cudaStream_t stream) {
    constexpr auto kS = snacSmemBytes<PurlinAtom>();
    // Zero-staging has no producer role, so every block that is not reserved
    // for the gather half reduces.
    constexpr auto putBlocks = residencyOf<CollConfig> == Staging::zero ? 0 : CollConfig::PUT_BLOCKS;
    const auto blocks = getTRBlocks<PurlinAtom>(
      bytes, putBlocks + gatherBlocks, maxReduceBlocks, ctx.world);
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ensureOptIn<allReduceKernel<PurlinAtom, Element, CollConfig, Bypass, LRMode::fullBuffer, ro>,
      kS>();
    allReduceKernel<PurlinAtom, Element, CollConfig, Bypass, LRMode::fullBuffer, ro>
      <<<blocks, PurlinAtom::THREADS, kS, stream>>>(kArgs, ctx);
  }

  template<int NArch, typename LRCfg, typename Element, LRMode mode = LRMode::fullBuffer,
    size_t MmMaxBytes = static_cast<size_t>(-1), ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void launchAllReduceLatencyAuto(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const int blocks, cudaStream_t stream) {
    // In the latency path, multimem only broadcasts packets with a multicast
    // store. It does not perform the reduction, so any reduction operator works.
    if constexpr (NArch >= 900) {
      if (ctx.mcStagingLR != nullptr && bytes <= MmMaxBytes) {
        launchAllReduceLatency<Atom<NArch, WithMultimem<LRCfg>>, Element, mode, ro>(
          src, dst, bytes, ctx, blocks, stream);
        return;
      }
    }
    launchAllReduceLatency<Atom<NArch, LRCfg>, Element, mode, ro>(
      src, dst, bytes, ctx, blocks, stream);
  }

  template<typename Element, int NArch, int World, ReduceOp ro = ReduceOp::add,
    Staging residency = Staging::staged>
  __host__ __forceinline__
  void allReduceTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    using Policy = host::AllReduceCodesign<NArch, World>;

    using LRConfig = Configuration<
            Policy::LR_THREADS,
      alignment,
      UNUSED,
      UNUSED,
      unrollFactor,
      host::getWorldUnroll<World>()
    >;
    using PurlinAtomLR = Atom<NArch, LRConfig>;

    const auto partitionWindow = ctx.world >= 4 &&
      bytes >= Policy::LR_PARTITION_MIN_BYTES && bytes <= Policy::LR_PARTITION_MAX_BYTES;
    if constexpr (sizeof(Element) <= sizeof(LRP::RT)) {
      const auto shardAlignment = static_cast<size_t>(ctx.world) * sizeof(LRP::RT);
      if (partitionWindow && bytes % shardAlignment == 0) {
        const auto blocksPerPeer = bytes >= Policy::LR_WIDE_MIN_BYTES ?
          Policy::LR_WIDE_BLOCKS_PER_PEER :
          (bytes <= Policy::LR_PARTITION_SMALL_MAX_BYTES ?
            Policy::LR_PARTITION_SMALL_BLOCKS_PER_PEER : Policy::LR_PARTITION_BLOCKS_PER_PEER);
        const auto blocks = (ctx.world - 1) * blocksPerPeer;
        if (bytes <= Policy::LR_PARTITION_SMALL_MAX_BYTES) {
          using SmallLRConfig = Configuration<
                        Policy::LR_PARTITION_SMALL_THREADS,
            alignment,
            UNUSED,
            UNUSED,
            unrollFactor,
            host::getWorldUnroll<World>()
          >;
          launchAllReduceLatencyAuto<NArch, SmallLRConfig, Element, LRMode::partitioned, static_cast<size_t>(-1), ro>(
            src, dst, bytes, ctx, blocks, stream);
        }
        else if (bytes >= Policy::LR_WIDE_MIN_BYTES) {
          using WideLRConfig = Configuration<
                        Policy::LR_WIDE_THREADS,
            alignment,
            UNUSED,
            UNUSED,
            unrollFactor,
            host::getWorldUnroll<World>()
          >;
          launchAllReduceLatencyAuto<NArch, WideLRConfig, Element, LRMode::partitioned, static_cast<size_t>(-1), ro>(
            src, dst, bytes, ctx, blocks, stream);
        }
        else {
          launchAllReduceLatencyAuto<NArch, LRConfig, Element, LRMode::partitioned, static_cast<size_t>(-1), ro>(
            src, dst, bytes, ctx, blocks, stream);
        }
        return;
      }
    }

    // A payload that cannot be split evenly stays on the full-buffer latency path.
    if (bytes <= Policy::LATENCY_THRESHOLD || partitionWindow) {
      const auto remotePeers = ctx.world > 1 ? ctx.world - 1 : 1;
      const auto blocks = ctx.world > 4 && bytes <= Policy::LR_DIRECT_MAX_BYTES ?
        remotePeers * Policy::LR_DIRECT_BLOCKS_PER_PEER :
        getLRBlocks<PurlinAtomLR::THREADS>(bytes);
      // The full-buffer path can multicast the whole payload. Beyond
      // LR_MM_MAX_BYTES, sending directly to the other ranks is faster.
      launchAllReduceLatencyAuto<NArch, LRConfig, Element, LRMode::fullBuffer, Policy::LR_MM_MAX_BYTES, ro>(
        src, dst, bytes, ctx, blocks, stream);
      return;
    }

    using TRConfig = Configuration<
            Policy::THREADS,
      alignment,
      Policy::PIPE_STAGES,
      Policy::STAGE_EXTENT,
      unrollFactor
    >;
    using PurlinAtomTR = Atom<NArch, TRConfig>;
    // Large chunked transfers can trade fewer consumers for a deeper pipeline
    // while keeping roughly the same amount of data in flight.
    using TRConfigChunked = Configuration<
            Policy::THREADS,
      alignment,
      (Policy::CHUNKED_PIPE_STAGES > 0 ? Policy::CHUNKED_PIPE_STAGES : Policy::PIPE_STAGES),
      Policy::STAGE_EXTENT,
      unrollFactor
    >;
    using PurlinAtomTRChunked = Atom<NArch, TRConfigChunked>;
    using NonChunkedConfig = CollectiveConfig<
      CollectiveType::nonChunked,
      Policy::NON_CHUNKED_PUT_BLOCKS,
      Policy::GATHER_BLOCKS,
      Policy::CHUNK_SIZE
    >;
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      Policy::GATHER_BLOCKS,
      Policy::CHUNK_SIZE
    >;
    using ChunkedMidConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      Policy::GATHER_BLOCKS,
      (Policy::CHUNK_SIZE_MID > 0 ? Policy::CHUNK_SIZE_MID : Policy::CHUNK_SIZE)
    >;
    using ChunkedLargeConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      Policy::GATHER_BLOCKS,
      (Policy::CHUNK_SIZE_LARGE > 0 ? Policy::CHUNK_SIZE_LARGE : Policy::CHUNK_SIZE)
    >;
    constexpr size_t nonChunkedMax = Policy::NON_CHUNKED_MAX_BYTES > 0 ?
      Policy::NON_CHUNKED_MAX_BYTES : Policy::CHUNK_SIZE;
    static_assert(Policy::MID_CHUNK_MIN_BYTES == static_cast<size_t>(-1) ||
      Policy::MID_CHUNK_MIN_BYTES <= Policy::LARGE_CHUNK_MIN_BYTES,
      "an enabled mid-chunk tier must sit at or below the large-chunk boundary");
    constexpr auto bypass = World == 2 ? World2Bypass::yes : World2Bypass::no;
    constexpr auto gatherBlocks = Policy::GATHER_BLOCKS == UNUSED ? 0 : Policy::GATHER_BLOCKS;
    // Every peer needs a gather block, or its shard is never fetched. Under
    // zero-staging that same per-peer wait is also what guarantees no peer is
    // still reading this rank's source when the kernel completes.
    if constexpr (bypass == World2Bypass::no) {
      if (gatherBlocks < static_cast<int>(ctx.world)) {
        throw std::runtime_error("allReduce needs one gather block per rank; GATHER_BLOCKS is " +
          std::to_string(gatherBlocks) + " for world " + std::to_string(static_cast<int>(ctx.world)));
      }
    }

    // Zero-staging. Two ranks reduce packed->packed and touch no staging at all.
    // Above two it is the fused reduce-then-gather, whose reduce half reads
    // peers' buffers and leaves its shard where the gather half can find it:
    // staging by default, or the destination itself when ctx.peerDst says the
    // destination is peer-visible, which drops staging from the path entirely.
    // Only src need be symmetric. ctx.mcSrc selects multimem; null keeps the
    // unicast bands.
    if constexpr (residency == Staging::zero) {
      const auto deep = Policy::MID_CHUNK_MIN_BYTES != static_cast<size_t>(-1) &&
        bytes >= Policy::MID_CHUNK_MIN_BYTES;
      constexpr auto deepConsumers = Policy::MM_CONSUMER_BLOCKS == AUTO ?
        Policy::MAX_CONSUMER_BLOCKS : Policy::MM_CONSUMER_BLOCKS;
      using ZeroStagedConfig = WithZeroStaging<NonChunkedConfig>;
      if constexpr (bypass == World2Bypass::no && multimemReducible<NArch, Element, ro>()) {
        // The reduce half load-reduces from the caller's alias and broadcasts
        // its shard into staging, so both mappings have to exist.
        if (ctx.mcSrc != nullptr && ctx.mcStagingTR != nullptr &&
            bytes % (static_cast<size_t>(ctx.world) * 16) == 0) {
          using AtomMM = Atom<NArch, WithMultimem<TRConfig, Policy::MM_DEPTH>>;
          launchAllReduceThroughput<AtomMM, Element, ZeroStagedConfig, bypass, ro>(
            src, dst, bytes, ctx, gatherBlocks,
            deep ? deepConsumers : Policy::MAX_CONSUMER_BLOCKS, stream);
          return;
        }
      }
      if (deep) {
        launchAllReduceThroughput<PurlinAtomTRChunked, Element, ZeroStagedConfig, bypass, ro>(
          src, dst, bytes, ctx, gatherBlocks, deepConsumers, stream);
      }
      else {
        launchAllReduceThroughput<PurlinAtomTR, Element, ZeroStagedConfig, bypass, ro>(
          src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
      }
      return;
    }

    // If the payload is larger than the staging area, reuse that area in windows.
    // The regular path uses one window per shard; the two-rank shortcut uses the
    // whole area as a single window.
    if (bytes > ctx.stagingTRSize) {
      // Each cyclic slot must drain before it can be reused, so larger slots can
      // work better here than in the resident chunked band.
      constexpr auto cyclicChunk = Policy::CYCLIC_CHUNK_SIZE > 0 ? Policy::CYCLIC_CHUNK_SIZE :
        (Policy::CHUNK_SIZE_LARGE > 0 ? Policy::CHUNK_SIZE_LARGE : Policy::CHUNK_SIZE);
      using ChunkedCyclicConfig = CollectiveConfig<
        CollectiveType::chunked,
        Policy::CHUNKED_PUT_BLOCKS,
        Policy::GATHER_BLOCKS,
        cyclicChunk,
        UNUSED,
        LAT_THRESHOLD_DEFAULT,
        StagingMode::cyclic
      >;
      const auto regions = bypass == World2Bypass::yes ? 1 : static_cast<int>(ctx.world);
      const auto cyclicCtx = cyclicContext(ctx, cyclicChunk, regions);
      if constexpr (bypass == World2Bypass::no && multimemReducible<NArch, Element, ro>()) {
        // NVLS can also reduce oversized payloads through the multicast mapping.
        // Its cyclic windows mirror the unicast layout, and evenly split shards
        // retain the required 16-byte alignment.
        if (ctx.mcStagingTR != nullptr && bytes % (static_cast<size_t>(ctx.world) * 16) == 0) {
          using AtomCyclicMM = Atom<NArch, WithMultimem<TRConfig, Policy::MM_DEPTH>>;
          constexpr auto residentMmConsumers = Policy::MM_CONSUMER_BLOCKS == AUTO ?
            Policy::MAX_CONSUMER_BLOCKS : Policy::MM_CONSUMER_BLOCKS;
          constexpr auto mmConsumers = Policy::CYCLIC_MM_CONSUMER_BLOCKS == AUTO ?
            residentMmConsumers : Policy::CYCLIC_MM_CONSUMER_BLOCKS;
          launchAllReduceThroughput<AtomCyclicMM, Element, ChunkedCyclicConfig, bypass, ro>(
            src, dst, bytes, cyclicCtx, gatherBlocks, mmConsumers, stream);
          return;
        }
      }
      launchAllReduceThroughput<PurlinAtomTRChunked, Element, ChunkedCyclicConfig, bypass, ro>(
        src, dst, bytes, cyclicCtx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
      return;
    }
    const auto dispatchThroughput = [&]<typename AtomTR>(
      const int fineReduceBlocks, const int largeReduceBlocks) {
      if (bytes <= nonChunkedMax) {
        launchAllReduceThroughput<AtomTR, Element, NonChunkedConfig, bypass, ro>(
          src, dst, bytes, ctx, gatherBlocks, fineReduceBlocks, stream);
      }
      else if (bytes >= Policy::LARGE_CHUNK_MIN_BYTES) {
        launchAllReduceThroughput<PurlinAtomTRChunked, Element, ChunkedLargeConfig, bypass, ro>(
          src, dst, bytes, ctx, gatherBlocks, largeReduceBlocks, stream);
      }
      else if (bytes >= Policy::MID_CHUNK_MIN_BYTES) {
        launchAllReduceThroughput<PurlinAtomTRChunked, Element, ChunkedMidConfig, bypass>(
          src, dst, bytes, ctx, gatherBlocks, fineReduceBlocks, stream);
      }
      else {
        launchAllReduceThroughput<PurlinAtomTRChunked, Element, ChunkedConfig, bypass, ro>(
          src, dst, bytes, ctx, gatherBlocks, fineReduceBlocks, stream);
      }
    };
    if constexpr (bypass == World2Bypass::no && multimemReducible<NArch, Element, ro>()) {
      // Use the NVLS multicast mapping only when it exists and every shard keeps
      // the 16-byte alignment required by multimem.
      if (ctx.mcStagingTR != nullptr && bytes % (static_cast<size_t>(ctx.world) * 16) == 0) {
        using AtomLarge = Atom<NArch, WithMultimem<TRConfig, Policy::MM_DEPTH>>;
        using AtomPaced = Atom<NArch, WithMultimem<TRConfig, Policy::PACED_MM_DEPTH>>;
        constexpr auto mmConsumers = Policy::MM_CONSUMER_BLOCKS == AUTO ?
          Policy::MAX_CONSUMER_BLOCKS : Policy::MM_CONSUMER_BLOCKS;
        if (bytes <= nonChunkedMax) {
          launchAllReduceThroughput<AtomPaced, Element, NonChunkedConfig, bypass, ro>(
            src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
        }
        else if (bytes >= Policy::LARGE_CHUNK_MIN_BYTES) {
          launchAllReduceThroughput<AtomLarge, Element, ChunkedLargeConfig, bypass, ro>(
            src, dst, bytes, ctx, gatherBlocks, mmConsumers, stream);
        }
        else if (bytes >= Policy::MID_CHUNK_MIN_BYTES) {
          launchAllReduceThroughput<AtomPaced, Element, ChunkedMidConfig, bypass, ro>(
            src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
        }
        else {
          launchAllReduceThroughput<AtomPaced, Element, ChunkedConfig, bypass, ro>(
            src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
        }
        return;
      }
    }
    dispatchThroughput.template operator()<PurlinAtomTR>(
      Policy::MAX_CONSUMER_BLOCKS, Policy::MAX_CONSUMER_BLOCKS);
  }

  // Staging::zero carries the caller guarantees documented on the Staging enum.
  template<int arch, typename Element, ReduceOp ro = ReduceOp::add,
    Staging residency = Staging::staged>
  __host__ __forceinline__
  void allReduce(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::allReduce", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    constexpr auto nArch = purlin::normalizeArch<arch>();
    switch (ctx.world) {
      case 2: allReduceTuned<Element, nArch, 2, ro, residency>(src, dst, bytes, ctx, stream); break;
      case 4: allReduceTuned<Element, nArch, 4, ro, residency>(src, dst, bytes, ctx, stream); break;
      case 8: allReduceTuned<Element, nArch, 8, ro, residency>(src, dst, bytes, ctx, stream); break;
      default: allReduceTuned<Element, nArch, host::FALLBACK, ro, residency>(src, dst, bytes, ctx, stream); break;
    }
  }
}
#endif // PURLIN_ALLREDUCE_CUH
