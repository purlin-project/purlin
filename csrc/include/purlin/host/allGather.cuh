//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ALLGATHER_CUH
#define PURLIN_ALLGATHER_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "tuning.cuh"
namespace purlin::AG {
  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world) {
    int blocks = 0;
    auto blocksNeeded = static_cast<int>(cuda::std::min((bytes / PurlinAtom::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxBlocks)) * world);
    // keep the clamp a world multiple so the per-peer consumer split stays exact
    blocksNeeded = bytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
    cuda::round_down(cuda::std::min(blocksNeeded, 32), world) : blocksNeeded;
    blocks = putBlocks + blocksNeeded;
    if (blocksNeeded < world) {
      // non-pipelined path
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
    if constexpr (InputLayout == DataLayout::packedV) {
      purlin::allGatherV<PurlinAtom, CollConfig>
        (kArgs.dst, kArgs.src, sizes, workspace, ctx, kArgs.blocks);
    }
    else {
      purlin::allGather<PurlinAtom, CollConfig>
        (kArgs.dst, kArgs.src, kArgs.bytes, workspace, ctx, kArgs.blocks);
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
    constexpr auto kS = PurlinAtom::COPY_SMEM_SIZE;
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
      host::AllGatherVTuning<NArch, World>, host::AllGatherTuning<NArch, World>>;

    if (dispatchBytes <= Policy::LATENCY_THRESHOLD) {
      using LRConfig = Configuration<
        Regime::latency,
        Policy::LR_THREADS,
        alignment,
        UNUSED,
        UNUSED,
        unrollFactor
      >;
      using PurlinAtomLR = Atom<NArch, LRConfig>;
      const auto blocks = getLRBlocks<PurlinAtomLR::THREADS>(dispatchBytes);
      constexpr auto kS = PurlinAtomLR::COPY_SMEM_SIZE;
      launchAllGatherKernel<InputLayout, PurlinAtomLR, CollectiveConfigLR, kS>
        (src, dst, bytes, ctx, sizes, blocks, stream);
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
      UNUSED,
      Policy::CHUNK_SIZE,
      UNUSED
    >;
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      UNUSED,
      Policy::CHUNK_SIZE,
      UNUSED
    >;

    // A contribution exceeding a staging half rings through it as one window of
    // chunk slots, drained by every rank's gather consumers.
    if (dispatchBytes > ctx.stagingTRSize) {
      using ChunkedRingConfig = CollectiveConfig<
        CollectiveType::chunked,
        Policy::CHUNKED_PUT_BLOCKS,
        UNUSED,
        Policy::CHUNK_SIZE,
        UNUSED,
        LAT_THRESHOLD_DEFAULT,
        StagingMode::ring
      >;
      const auto ringCtx = ringContext(ctx, Policy::CHUNK_SIZE, 1);
      launchAllGatherThroughput<InputLayout, PurlinAtomTR, ChunkedRingConfig>(
        src, dst, bytes, dispatchBytes, ringCtx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
      return;
    }

    const bool useAlternative = Policy::ALT_THREADS > 0 &&
      dispatchBytes >= Policy::ALT_MIN_BYTES && dispatchBytes <= Policy::ALT_MAX_BYTES;
    if constexpr (Policy::ALT_THREADS > 0) {
      if (useAlternative) {
        using AltTRConfig = Configuration<
          Regime::throughput,
          Policy::ALT_THREADS,
          alignment,
          Policy::PIPE_STAGES,
          Policy::STAGE_EXTENT,
          unrollFactor
        >;
        using AltPurlinAtomTR = Atom<NArch, AltTRConfig>;
        if (dispatchBytes <= Policy::CHUNK_SIZE) {
          launchAllGatherThroughput<InputLayout, AltPurlinAtomTR, NonChunkedConfig>(
            src, dst, bytes, dispatchBytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
        }
        else {
          launchAllGatherThroughput<InputLayout, AltPurlinAtomTR, ChunkedConfig>(
            src, dst, bytes, dispatchBytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
        }
        return;
      }
    }

    if (dispatchBytes <= Policy::CHUNK_SIZE) {
      launchAllGatherThroughput<InputLayout, PurlinAtomTR, NonChunkedConfig>(
        src, dst, bytes, dispatchBytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
    else {
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
    const auto maxBytes = ctx.vState.maxBytes;
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchAllGather<DataLayout::packedV, nArch>
      (src, dst, bytes, maxBytes, sizes, ctx, stream);
  }
}
#endif //PURLIN_ALLGATHER_CUH
