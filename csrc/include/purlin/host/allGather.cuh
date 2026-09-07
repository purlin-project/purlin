#ifndef PURLIN_ALLGATHER_CUH
#define PURLIN_ALLGATHER_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "codesign.cuh"
namespace purlin::AG {
  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world) {
    int blocks = 0;
    auto blocksNeeded = static_cast<int>(cuda::std::min((bytes / PurlinAtom::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxBlocks)) * world);
    // Keep the block count divisible by the world size so every peer gets the
    // same number of consumers.
    blocksNeeded = bytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
    cuda::round_down(cuda::std::min(blocksNeeded, 32), world) : blocksNeeded;
    blocks = putBlocks + blocksNeeded;
    if (blocksNeeded < world) {
      // Small transfers do not have enough work to fill the pipeline.
      blocks = putBlocks + (cuda::std::min(cuda::ceil_div(bytes,
        static_cast<size_t>(PurlinAtom::THREADS*PurlinAtom::BaseConfig::ALIGNMENT_BYTES)),
        static_cast<size_t>(maxBlocks)) * world);
    }
    return blocks;
  }
}

namespace purlin {
  template<DataLayout InputLayout, typename PurlinAtom, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void allGatherKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx,
    const size_t* __restrict__ sizes) {
    static_assert(InputLayout == DataLayout::packed || InputLayout == DataLayout::packedV);
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    const SnacArgs<cuda::fast_mod_div<long int>> args{
      .dst = kArgs.dst,
      .src = kArgs.src,
      .bytes = kArgs.bytes,
      .workspace = workspace,
      .sizes = sizes,
      .blocks = kArgs.blocks,
      .collBlocks = static_cast<int>(kArgs.blocks),
    };
    if constexpr (InputLayout == DataLayout::packedV) {
      purlin::allGatherV<PurlinAtom, CollConfig>(args, ctx);
    }
    else {
      purlin::allGather<PurlinAtom, CollConfig>(args, ctx);
    }
  }

  template<DataLayout InputLayout, typename PurlinAtom, typename CollConfig, size_t SmemSize>
  __host__ __forceinline__
  void launchAllGatherKernel(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const size_t* __restrict__ sizes, const int& blocks, cudaStream_t stream) {
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ensureOptIn<allGatherKernel<InputLayout, PurlinAtom, CollConfig>, SmemSize>();
    allGatherKernel<InputLayout, PurlinAtom, CollConfig>
      <<<blocks, PurlinAtom::THREADS, SmemSize, stream>>>(kArgs, ctx, sizes);
  }

  template<DataLayout InputLayout, typename PurlinAtom, typename CollConfig>
  __host__ __forceinline__
  void launchAllGatherThroughput(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const Context& ctx, const size_t* __restrict__ sizes, const int& maxConsumerBlocks, cudaStream_t stream) {
    constexpr auto kS = copySmemBytes<PurlinAtom>();
    constexpr auto putBlocks = CollConfig::PUT_BLOCKS;
    const auto blocks = AG::getBlocks<PurlinAtom>(dispatchBytes, putBlocks, maxConsumerBlocks, ctx.world);
    launchAllGatherKernel<InputLayout, PurlinAtom, CollConfig, kS>
      (src, dst, bytes, ctx, sizes, blocks, stream);
  }

  template<DataLayout InputLayout, int NArch, int World>
  __host__ __forceinline__
  void allGatherTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ sizes, const Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    using Policy = cuda::std::conditional_t<InputLayout == DataLayout::packedV,
      host::AllGatherVCodesign<NArch, World>, host::AllGatherCodesign<NArch, World>>;

    if (dispatchBytes <= Policy::LATENCY_THRESHOLD) {
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
      launchAllGatherKernel<InputLayout, PurlinAtomLR, CollectiveConfigLR, kS>
        (src, dst, bytes, ctx, sizes, blocks, stream);
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
    using PurlinAtomChunked = Atom<NArch, TRConfigChunked>;
    using NonChunkedConfig = CollectiveConfig<
      CollectiveType::nonChunked,
      Policy::NON_CHUNKED_PUT_BLOCKS,
      UNUSED,
      Policy::CHUNK_SIZE,
      UNUSED
    >;
    // We tested sending small variable contributions through the packet path.
    // It made no measurable difference: a small contribution does not tie up
    // consumer groups or staging rounds. The experiment is documented in the
    // per-stream brief (2026-08-27).
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      UNUSED,
      Policy::CHUNK_SIZE,
      UNUSED
    >;

    // If one contribution is larger than the staging area, reuse the area as a
    // window of chunk slots. Every rank's consumers drain each window in turn.
    if (dispatchBytes > ctx.stagingTRSize) {
      // Cyclic slots drain one at a time, so this band can prefer a different
      // slot size than the resident chunked band. A value of 0 keeps the
      // resident chunk size, which is what every policy did before this hook.
      constexpr size_t cyclicChunkSize = Policy::CYCLIC_CHUNK_SIZE > 0 ?
        Policy::CYCLIC_CHUNK_SIZE : Policy::CHUNK_SIZE;
      using ChunkedCyclicConfig = CollectiveConfig<
        CollectiveType::chunked,
        Policy::CHUNKED_PUT_BLOCKS,
        UNUSED,
        cyclicChunkSize,
        UNUSED,
        LAT_THRESHOLD_DEFAULT,
        StagingMode::cyclic
      >;
      const auto cyclicCtx = cyclicContext(ctx, cyclicChunkSize, 1);
      constexpr auto cyclicConsumers = Policy::CHUNKED_CONSUMER_BLOCKS == AUTO ?
        Policy::MAX_CONSUMER_BLOCKS : Policy::CHUNKED_CONSUMER_BLOCKS;
      launchAllGatherThroughput<InputLayout, PurlinAtomChunked, ChunkedCyclicConfig>(
        src, dst, bytes, dispatchBytes, cyclicCtx, sizes, cyclicConsumers, stream);
      return;
    }

    constexpr auto altConsumers = Policy::ALT_CONSUMER_BLOCKS == AUTO ?
      Policy::MAX_CONSUMER_BLOCKS : Policy::ALT_CONSUMER_BLOCKS;
    const bool useAlternative = Policy::ALT_THREADS > 0 &&
      dispatchBytes >= Policy::ALT_MIN_BYTES && dispatchBytes <= Policy::ALT_MAX_BYTES;
    if constexpr (Policy::ALT_THREADS > 0) {
      if (useAlternative) {
        using AltTRConfig = Configuration<
                    Policy::ALT_THREADS,
          alignment,
          Policy::PIPE_STAGES,
          Policy::STAGE_EXTENT,
          unrollFactor
        >;
        using AltPurlinAtomTR = Atom<NArch, AltTRConfig>;
        if (dispatchBytes <= Policy::CHUNK_SIZE) {
          launchAllGatherThroughput<InputLayout, AltPurlinAtomTR, NonChunkedConfig>(
            src, dst, bytes, dispatchBytes, ctx, sizes, altConsumers, stream);
        }
        else {
          launchAllGatherThroughput<InputLayout, AltPurlinAtomTR, ChunkedConfig>(
            src, dst, bytes, dispatchBytes, ctx, sizes, altConsumers, stream);
        }
        return;
      }
    }

    if (dispatchBytes <= Policy::CHUNK_SIZE) {
      launchAllGatherThroughput<InputLayout, PurlinAtomTR, NonChunkedConfig>(
        src, dst, bytes, dispatchBytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
    else if (Policy::DEEP_CHUNK_MIN_BYTES == 0 || dispatchBytes >= Policy::DEEP_CHUNK_MIN_BYTES) {
      constexpr auto chunkedConsumers = Policy::CHUNKED_CONSUMER_BLOCKS == AUTO ?
        Policy::MAX_CONSUMER_BLOCKS : Policy::CHUNKED_CONSUMER_BLOCKS;
      launchAllGatherThroughput<InputLayout, PurlinAtomChunked, ChunkedConfig>(
        src, dst, bytes, dispatchBytes, ctx, sizes, chunkedConsumers, stream);
    }
    else {
      // Transfers just above the chunk boundary keep the shallow, wide shape.
      // Both bands use the same chunk size and therefore the same protocol state.
      launchAllGatherThroughput<InputLayout, PurlinAtomTR, ChunkedConfig>(
        src, dst, bytes, dispatchBytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
  }

  template<DataLayout InputLayout, int NArch>
  __host__ __forceinline__
  void dispatchAllGather(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ sizes, const Context& ctx, cudaStream_t stream) {
    switch (ctx.world) {
      case 2:
        allGatherTuned<InputLayout, NArch, 2>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      case 4:
        allGatherTuned<InputLayout, NArch, 4>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      case 8:
        allGatherTuned<InputLayout, NArch, 8>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      default:
        allGatherTuned<InputLayout, NArch, host::FALLBACK>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
    }
  }

  template<int arch>
  __host__ __forceinline__
  void allGather(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::allGather", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (bytes == 0 || ctx.world == 1) return;
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchAllGather<DataLayout::packed, nArch>
      (src, dst, bytes, bytes, nullptr, ctx, stream);
  }

  template<int arch>
  __host__ __forceinline__
  void allGatherV(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t* __restrict__ sizes,
    const Context& ctx, cudaStream_t stream) {
    const auto bytes = ctx.vState.bytes;
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::allGatherV", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (ctx.world == 1) return;
    const auto maxBytes = ctx.vState.maxBytes;
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchAllGather<DataLayout::packedV, nArch>
      (src, dst, bytes, maxBytes, sizes, ctx, stream);
  }
}
#endif // PURLIN_ALLGATHER_CUH
