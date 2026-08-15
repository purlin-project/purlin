//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ALLREDUCE_CUH
#define PURLIN_ALLREDUCE_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "tuning.cuh"
namespace purlin {
  template<typename PurlinAtom, typename Element, typename CollConfig,
    World2Bypass wb = World2Bypass::unknown, bool partitioned = false>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void allReduceKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx) {
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
    purlin::allReduce<PurlinAtom, CollConfig, wb, partitioned>(
      kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx, kArgs.blocks);
  }

  template<typename PurlinAtom, typename Element, bool partitioned = false>
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
    allReduceKernel<PurlinAtom, Element, CollectiveConfigLR, World2Bypass::unknown, partitioned>
      <<<blocks, PurlinAtom::THREADS, kS, stream>>>(kArgs, ctx);
  }

  template<typename PurlinAtom, typename Element, typename CollConfig, World2Bypass Bypass>
  __host__ __forceinline__
  void launchAllReduceThroughput(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const int& gatherBlocks, const int& maxReduceBlocks, cudaStream_t stream) {
    constexpr auto kS = cuda::std::max(PurlinAtom::COPY_SMEM_SIZE, PurlinAtom::RED_SMEM_SIZE);
    constexpr auto putBlocks = CollConfig::PUT_BLOCKS;
    const auto blocks = getTRBlocks<PurlinAtom>(
      bytes, putBlocks + gatherBlocks, maxReduceBlocks, ctx.world);
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ensureOptIn<allReduceKernel<PurlinAtom, Element, CollConfig, Bypass>, kS>();
    allReduceKernel<PurlinAtom, Element, CollConfig, Bypass>
      <<<blocks, PurlinAtom::THREADS, kS, stream>>>(kArgs, ctx);
  }

  template<int NArch, typename LRCfg, typename Element, bool partitioned = false,
    size_t MmMaxBytes = static_cast<size_t>(-1)>
  __host__ __forceinline__
  void launchAllReduceLatencyAuto(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const int blocks, cudaStream_t stream) {
    if constexpr (NArch >= 900) {
      if (ctx.mcStagingLR != nullptr && bytes <= MmMaxBytes) {
        launchAllReduceLatency<Atom<NArch, WithMultimem<LRCfg>>, Element, partitioned>(
          src, dst, bytes, ctx, blocks, stream);
        return;
      }
    }
    launchAllReduceLatency<Atom<NArch, LRCfg>, Element, partitioned>(
      src, dst, bytes, ctx, blocks, stream);
  }

  template<typename Element, int NArch, int World>
  __host__ __forceinline__
  void allReduceTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    using Policy = host::AllReduceTuning<NArch, World>;

    using LRConfig = Configuration<
      Regime::latency,
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
            Regime::latency,
            Policy::LR_PARTITION_SMALL_THREADS,
            alignment,
            UNUSED,
            UNUSED,
            unrollFactor,
            host::getWorldUnroll<World>()
          >;
          launchAllReduceLatencyAuto<NArch, SmallLRConfig, Element, true>(
            src, dst, bytes, ctx, blocks, stream);
        }
        else if (bytes >= Policy::LR_WIDE_MIN_BYTES) {
          using WideLRConfig = Configuration<
            Regime::latency,
            Policy::LR_WIDE_THREADS,
            alignment,
            UNUSED,
            UNUSED,
            unrollFactor,
            host::getWorldUnroll<World>()
          >;
          launchAllReduceLatencyAuto<NArch, WideLRConfig, Element, true>(
            src, dst, bytes, ctx, blocks, stream);
        }
        else {
          launchAllReduceLatencyAuto<NArch, LRConfig, Element, true>(
            src, dst, bytes, ctx, blocks, stream);
        }
        return;
      }
    }

    // Keep non-divisible partition candidates on the unified full-buffer LR path.
    if (bytes <= Policy::LATENCY_THRESHOLD || partitionWindow) {
      const auto remotePeers = ctx.world > 1 ? ctx.world - 1 : 1;
      const auto blocks = ctx.world > 4 && bytes <= Policy::LR_DIRECT_MAX_BYTES ?
        remotePeers * Policy::LR_DIRECT_BLOCKS_PER_PEER :
        getLRBlocks<PurlinAtomLR::THREADS>(bytes);
      // The full-buffer packet broadcast multicasts the whole payload; past
      // LR_MM_MAX_BYTES the multicast store's bandwidth loses to world-1 unicasts.
      launchAllReduceLatencyAuto<NArch, LRConfig, Element, false, Policy::LR_MM_MAX_BYTES>(
        src, dst, bytes, ctx, blocks, stream);
      return;
    }

    using TRConfig = Configuration<
      Regime::throughput,
      Policy::THREADS,
      alignment,
      Policy::PIPE_STAGES,
      Policy::STAGE_EXTENT,
      unrollFactor
    >;
    using PurlinAtomTR = Atom<NArch, TRConfig>;
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
    const auto dispatchThroughput = [&]<typename AtomTR>(
      const int fineReduceBlocks, const int largeReduceBlocks) {
      if (bytes <= nonChunkedMax) {
        launchAllReduceThroughput<AtomTR, Element, NonChunkedConfig, bypass>(
          src, dst, bytes, ctx, gatherBlocks, fineReduceBlocks, stream);
      }
      else if (bytes >= Policy::LARGE_CHUNK_MIN_BYTES) {
        launchAllReduceThroughput<AtomTR, Element, ChunkedLargeConfig, bypass>(
          src, dst, bytes, ctx, gatherBlocks, largeReduceBlocks, stream);
      }
      else if (bytes >= Policy::MID_CHUNK_MIN_BYTES) {
        launchAllReduceThroughput<AtomTR, Element, ChunkedMidConfig, bypass>(
          src, dst, bytes, ctx, gatherBlocks, fineReduceBlocks, stream);
      }
      else {
        launchAllReduceThroughput<AtomTR, Element, ChunkedConfig, bypass>(
          src, dst, bytes, ctx, gatherBlocks, fineReduceBlocks, stream);
      }
    };
    if constexpr (bypass == World2Bypass::no && (NArch >= 900 && sizeof(Element) > 1)) {
      // NVLS: reduce through the multicast staging mapping when it exists and the
      // shard split preserves 16-byte multimem alignment.
      if (ctx.mcStagingTR != nullptr && bytes % (static_cast<size_t>(ctx.world) * 16) == 0) {
        using AtomLarge = Atom<NArch, WithMultimem<TRConfig, Policy::MM_DEPTH>>;
        using AtomPaced = Atom<NArch, WithMultimem<TRConfig, Policy::PACED_MM_DEPTH>>;
        constexpr auto mmConsumers = Policy::MM_CONSUMER_BLOCKS == AUTO ?
          Policy::MAX_CONSUMER_BLOCKS : Policy::MM_CONSUMER_BLOCKS;
        if (bytes <= nonChunkedMax) {
          launchAllReduceThroughput<AtomPaced, Element, NonChunkedConfig, bypass>(
            src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
        }
        else if (bytes >= Policy::LARGE_CHUNK_MIN_BYTES) {
          launchAllReduceThroughput<AtomLarge, Element, ChunkedLargeConfig, bypass>(
            src, dst, bytes, ctx, gatherBlocks, mmConsumers, stream);
        }
        else if (bytes >= Policy::MID_CHUNK_MIN_BYTES) {
          launchAllReduceThroughput<AtomPaced, Element, ChunkedMidConfig, bypass>(
            src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
        }
        else {
          launchAllReduceThroughput<AtomPaced, Element, ChunkedConfig, bypass>(
            src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
        }
        return;
      }
    }
    dispatchThroughput.template operator()<PurlinAtomTR>(
      Policy::MAX_CONSUMER_BLOCKS, Policy::MAX_CONSUMER_BLOCKS);
  }

  template<int arch, typename Element>
  __host__ __forceinline__
  void allReduce(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::allReduce", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (bytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<arch>();
    switch (ctx.world) {
      case 2: allReduceTuned<Element, nArch, 2>(src, dst, bytes, ctx, stream); break;
      case 4: allReduceTuned<Element, nArch, 4>(src, dst, bytes, ctx, stream); break;
      case 8: allReduceTuned<Element, nArch, 8>(src, dst, bytes, ctx, stream); break;
      default: allReduceTuned<Element, nArch, host::FALLBACK>(src, dst, bytes, ctx, stream); break;
    }
  }
}
#endif //PURLIN_ALLREDUCE_CUH
