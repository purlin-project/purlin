#ifndef PURLIN_ALL2ALL_CUH
#define PURLIN_ALL2ALL_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "codesign.cuh"

namespace purlin::A2A {
  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world,
    const int& actualWorld, const bool& consumerFloor = false) {
    int blocks = 0;
    auto blocksNeeded = static_cast<int>(cuda::std::min((bytes / PurlinAtom::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxBlocks)) * actualWorld);
    const auto smallBlocks = static_cast<int>(cuda::std::min(cuda::ceil_div(bytes,
      static_cast<size_t>(PurlinAtom::THREADS*PurlinAtom::BaseConfig::ALIGNMENT_BYTES)),
      static_cast<size_t>(maxBlocks)) * actualWorld);
    if (consumerFloor) {
      blocksNeeded = cuda::std::max(blocksNeeded, smallBlocks);
    }
    blocksNeeded = bytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
    cuda::round_down(cuda::std::min(blocksNeeded, 32), actualWorld) : blocksNeeded;
    blocks = putBlocks + blocksNeeded;
    if (blocksNeeded < actualWorld) {
      blocks = putBlocks + smallBlocks;
    }
    if (blocks > static_cast<int>(MAX_NUM_CTAS)) {
      throw std::runtime_error("grid of " + std::to_string(blocks) +
        " blocks exceeds MAX_NUM_CTAS (" + std::to_string(MAX_NUM_CTAS) +
        "); lower the put/consumer block tuning");
    }
    return blocks;
  }
}

namespace purlin {
  template<DataLayout InputLayout, typename PurlinAtom, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void all2allKernel(const __grid_constant__ Args kArgs,
    const size_t* __restrict__ inSplits, const size_t* __restrict__ outSplits,
    const __grid_constant__ Context ctx) {
    static_assert(InputLayout == DataLayout::scattered || InputLayout == DataLayout::scatteredV);
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    const SnacArgs<cuda::fast_mod_div<long int>> args{
      .dst = kArgs.dst,
      .src = kArgs.src,
      .bytes = kArgs.bytes,
      .workspace = workspace,
      .sizes = outSplits,
      .inSizes = inSplits,
      .blocks = kArgs.blocks,
      .collBlocks = static_cast<int>(kArgs.blocks),
    };
    if constexpr (InputLayout == DataLayout::scatteredV) {
      purlin::all2allV<PurlinAtom, CollConfig>(args, ctx);
    }
    else {
      purlin::all2all<PurlinAtom, CollConfig>(args, ctx);
    }
  }

  template<
    DataLayout InputLayout,
    typename PurlinAtom,
    typename CollConfig,
    size_t SmemSize
  >
  __host__ __forceinline__
  void launchAll2AllKernel(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes,
    const size_t* __restrict__ inSplits, const size_t* __restrict__ outSplits,
    const Context& ctx, const int& blocks, cudaStream_t stream) {
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ensureOptIn<all2allKernel<InputLayout, PurlinAtom, CollConfig>, SmemSize>();
    all2allKernel<InputLayout, PurlinAtom, CollConfig>
      <<<blocks, PurlinAtom::THREADS, SmemSize, stream>>>(kArgs, inSplits, outSplits, ctx);
  }

  template<DataLayout InputLayout, typename PurlinAtom, typename CollConfig>
  __host__ __forceinline__
  void launchAll2AllThroughput(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ inSplits, const size_t* __restrict__ outSplits, Context& ctx,
    const int& targetPutBlocks, const int& maxConsumerBlocks, cudaStream_t stream,
    const bool& consumerFloor = false) {
    const int actualWorld = ctx.actualWorld;
    const auto putBlocksPerPeer = cuda::std::bit_floor(static_cast<uint32_t>(
      cuda::round_down(targetPutBlocks, actualWorld) / actualWorld));
    const auto stagingBlocks = putBlocksPerPeer * actualWorld;
    const auto putBlocks = stagingBlocks + CollConfig::LOCAL_PUT_BLOCKS;
    ctx.stagingBlocks = static_cast<int>(stagingBlocks);
    if constexpr (InputLayout == DataLayout::scatteredV ||
      CollConfig::STAGING_MODE == StagingMode::cyclic) {
      auto slots = cyclicSlotCount(ctx.stagingTRSize, CollConfig::CHUNK_SIZE, ctx.world);
      if constexpr (CollConfig::CYCLIC_STREAM_CHUNK > 0) {
        constexpr auto ratio = static_cast<int>(CollConfig::CYCLIC_STREAM_CHUNK / CollConfig::CHUNK_SIZE);
        slots = slots >= ratio ? cuda::round_down(slots, ratio) : slots;
      }
      ctx.cyclicSlots = cuda::fast_mod_div<int>{slots};
    }
    const auto blocks = A2A::getBlocks<PurlinAtom>(
      dispatchBytes, putBlocks, maxConsumerBlocks, ctx.world, actualWorld, consumerFloor);
    constexpr auto kS = copySmemBytes<PurlinAtom>();
    launchAll2AllKernel<InputLayout, PurlinAtom, CollConfig, kS>
      (src, dst, bytes, inSplits, outSplits, ctx, blocks, stream);
  }

  template<DataLayout InputLayout, int NArch, int World>
  __host__ __forceinline__
  void all2allTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ inSplits, const size_t* __restrict__ outSplits,
    Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    using Policy = cuda::std::conditional_t<InputLayout == DataLayout::scatteredV,
    host::All2AllVCodesign<NArch, World>, host::All2AllCodesign<NArch, World>>;
    constexpr auto latencyThreshold = Policy::LATENCY_THRESHOLD;

    if constexpr (InputLayout == DataLayout::scattered) {
      if (dispatchBytes <= latencyThreshold) {
        using LRConfig = Configuration<
                    Policy::LR_THREADS,
          alignment,
          UNUSED,
          UNUSED,
          unrollFactor
        >;
        using PurlinAtomLR = Atom<NArch, LRConfig>;
        const auto blocks = getLRBlocks<PurlinAtomLR::THREADS>(dispatchBytes);
        constexpr auto kS = copySmemBytes<PurlinAtomLR, Regime::latency>();
        launchAll2AllKernel<InputLayout, PurlinAtomLR, CollectiveConfigLR, kS>
          (src, dst, bytes, inSplits, outSplits, ctx, blocks, stream);
        return;
      }
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
    using PurlinAtomChunked = Atom<NArch, TRConfigChunked>;
    using NonChunkedConfig = CollectiveConfig<
      CollectiveType::nonChunked,
      UNUSED,
      UNUSED,
      Policy::CHUNK_SIZE,
      Policy::LOCAL_PUT_BLOCKS,
      Policy::LATENCY_THRESHOLD
    >;
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      UNUSED,
      UNUSED,
      Policy::CHUNK_SIZE,
      Policy::LOCAL_PUT_BLOCKS,
      Policy::LATENCY_THRESHOLD
    >;

    if constexpr (InputLayout == DataLayout::scattered) {
      const auto footprint = bytes * static_cast<size_t>(static_cast<int>(ctx.world));
      if (footprint > ctx.stagingTRSize) {
        constexpr size_t cyclicChunkSize = Policy::CYCLIC_CHUNK_SIZE > 0 ?
          Policy::CYCLIC_CHUNK_SIZE : Policy::CHUNK_SIZE;
        using ChunkedCyclicConfig = CollectiveConfig<
          CollectiveType::chunked,
          UNUSED,
          UNUSED,
          cyclicChunkSize,
          Policy::LOCAL_PUT_BLOCKS,
          Policy::LATENCY_THRESHOLD,
          StagingMode::cyclic
        >;
        launchAll2AllThroughput<InputLayout, PurlinAtomChunked, ChunkedCyclicConfig>(
          src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx,
          Policy::CHUNKED_PUT_BLOCKS, Policy::MAX_CONSUMER_BLOCKS, stream);
        return;
      }
    }

    if constexpr (InputLayout == DataLayout::scatteredV) {
      static_assert(Policy::CHUNK_SIZE_LARGE == 0 || Policy::CHUNK_SIZE_LARGE == Policy::CHUNK_SIZE,
        "a2aV bands must share one chunk size");
      using VChunkedConfig = CollectiveConfig<
        CollectiveType::chunked,
        UNUSED,
        UNUSED,
        Policy::CHUNK_SIZE,
        Policy::LOCAL_PUT_BLOCKS,
        Policy::LATENCY_THRESHOLD,
        StagingMode::resident,
        Policy::PER_STREAM_THRESHOLD,
        Policy::CYCLIC_STREAM_CHUNK,
        Policy::WEIGHTED_MAPPING_MIN_BYTES
      >;
      using ChunkedLargeConfig = CollectiveConfig<
        CollectiveType::chunked,
        UNUSED,
        UNUSED,
        (Policy::CHUNK_SIZE_LARGE > 0 ? Policy::CHUNK_SIZE_LARGE : Policy::CHUNK_SIZE),
        Policy::LOCAL_PUT_BLOCKS,
        Policy::LATENCY_THRESHOLD,
        StagingMode::resident,
        Policy::PER_STREAM_THRESHOLD,
        Policy::CYCLIC_STREAM_CHUNK,
        Policy::WEIGHTED_MAPPING_MIN_BYTES
      >;
      constexpr auto largeConsumers = Policy::LARGE_CONSUMER_BLOCKS == AUTO ?
        Policy::MAX_CONSUMER_BLOCKS : Policy::LARGE_CONSUMER_BLOCKS;
      constexpr auto largePuts = Policy::LARGE_PUT_BLOCKS == AUTO ?
        Policy::CHUNKED_PUT_BLOCKS : Policy::LARGE_PUT_BLOCKS;
      const auto putBlocks = ctx.vState.maxBytes >= Policy::LARGE_CHUNK_MIN_BYTES ?
        largePuts : Policy::CHUNKED_PUT_BLOCKS;
      if (dispatchBytes >= Policy::LARGE_CHUNK_MIN_BYTES) {
        launchAll2AllThroughput<InputLayout, PurlinAtomChunked, ChunkedLargeConfig>(
          src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx,
          putBlocks, largeConsumers, stream, Policy::CONSUMER_FLOOR);
      }
      else {
        launchAll2AllThroughput<InputLayout, PurlinAtomTR, VChunkedConfig>(
          src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx,
          putBlocks, Policy::MAX_CONSUMER_BLOCKS, stream, Policy::CONSUMER_FLOOR);
      }
    }
    else if (dispatchBytes <= Policy::CHUNK_SIZE) {
      launchAll2AllThroughput<InputLayout, PurlinAtomTR, NonChunkedConfig>(
        src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx,
        Policy::NON_CHUNKED_PUT_BLOCKS, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
    else {
      launchAll2AllThroughput<InputLayout, PurlinAtomChunked, ChunkedConfig>(
        src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx,
        Policy::CHUNKED_PUT_BLOCKS, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
  }

  template<DataLayout InputLayout, int NArch>
  __host__ __forceinline__
  void dispatchAll2All(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ inSplits, const size_t* __restrict__ outSplits,
    Context& ctx, cudaStream_t stream) {
    switch (ctx.world) {
      case 2:
        all2allTuned<InputLayout, NArch, 2>
          (src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx, stream);
        break;
      case 4:
        all2allTuned<InputLayout, NArch, 4>
          (src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx, stream);
        break;
      case 8:
        all2allTuned<InputLayout, NArch, 8>
          (src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx, stream);
        break;
      default:
        all2allTuned<InputLayout, NArch, host::FALLBACK>
          (src, dst, bytes, dispatchBytes, inSplits, outSplits, ctx, stream);
        break;
    }
  }

  template<int arch>
  __host__ __forceinline__
  void all2all(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::all2all", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (bytes == 0 || ctx.world == 1) return;
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchAll2All<DataLayout::scattered, nArch>
      (src, dst, bytes, bytes, nullptr, nullptr, ctx, stream);
  }

  template<int arch>
  __host__ __forceinline__
  void all2allV(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t* __restrict__ const& inSplits,
    const size_t* __restrict__ const& outSplits, Context& ctx, cudaStream_t stream) {
    const auto bytes = ctx.vState.bytes;
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::all2allV", nvtx3::payload{static_cast<uint64_t>(ctx.vState.totalBytes)}};
#endif
    if (ctx.world == 1) return;
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchAll2All<DataLayout::scatteredV, nArch>
      (src, dst, bytes, ctx.vState.maxOutBytes, inSplits, outSplits, ctx, stream);
  }
}
#endif // PURLIN_ALL2ALL_CUH
