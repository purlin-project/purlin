//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_REDUCESCATTER_CUH
#define PURLIN_REDUCESCATTER_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "tuning.cuh"

namespace purlin {
  template<DataLayout InputLayout, typename PurlinAtom, typename Element, typename CollConfig,
    ReduceOp ro = ReduceOp::add>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void reduceScatterKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx,
    const size_t* __restrict__ sizes) {
    static_assert(InputLayout == DataLayout::scattered || InputLayout == DataLayout::scatteredV);
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
    if constexpr (InputLayout == DataLayout::scatteredV) {
      purlin::reduceScatterV<PurlinAtom, CollConfig, Element, ro>(args, ctx);
    }
    else {
      purlin::reduceScatter<PurlinAtom, CollConfig, Element, ro>(args, ctx);
    }
  }

  template<DataLayout InputLayout, typename PurlinAtom, typename Element, typename CollConfig, size_t SmemSize,
    ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void launchReduceScatterKernel(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const size_t* __restrict__ sizes, const int& blocks, cudaStream_t stream) {
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ensureOptIn<reduceScatterKernel<InputLayout, PurlinAtom, Element, CollConfig, ro>, SmemSize>();
    reduceScatterKernel<InputLayout, PurlinAtom, Element, CollConfig, ro>
      <<<blocks, PurlinAtom::THREADS, SmemSize, stream>>>(kArgs, ctx, sizes);
  }

  template<DataLayout InputLayout, typename PurlinAtom, typename Element, typename CollConfig,
    ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void rst(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const size_t* __restrict__ sizes, const int& maxReduceBlocks, cudaStream_t stream) {
    constexpr auto kS = snacSmemBytes<PurlinAtom>();
    constexpr auto putBlocks = CollConfig::PUT_BLOCKS;
    const auto blocks = getTRBlocks<PurlinAtom>(bytes, putBlocks, maxReduceBlocks, ctx.world);
    launchReduceScatterKernel<InputLayout, PurlinAtom, Element, CollConfig, kS, ro>
      (src, dst, bytes, ctx, sizes, blocks, stream);
  }

  template<DataLayout InputLayout, typename Element, int NArch, int World, ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void reduceScatterTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ sizes, const Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    using Policy = host::ReduceScatterTuning<NArch, World>;

    if (dispatchBytes <= Policy::LATENCY_THRESHOLD) {
      using LRConfig = Configuration<
                Policy::LR_THREADS,
        alignment,
        UNUSED,
        UNUSED,
        unrollFactor,
        host::getWorldUnroll<World>()
      >;
      using PurlinAtomLR = Atom<NArch, LRConfig>;
      const auto blocks = getLRBlocks<PurlinAtomLR::THREADS>(dispatchBytes);
      constexpr auto kS = redSmemBytes<PurlinAtomLR, Regime::latency>();
      launchReduceScatterKernel<InputLayout, PurlinAtomLR, Element, CollectiveConfigLR, kS, ro>
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
    using NonChunkedConfig = CollectiveConfig<
      CollectiveType::nonChunked,
      Policy::NON_CHUNKED_PUT_BLOCKS,
      UNUSED,
      Policy::CHUNK_SIZE
    >;
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      UNUSED,
      Policy::CHUNK_SIZE
    >;

    // A staged input exceeding a staging half cycles through per-shard windows,
    // each drained by its owning rank's reducers.
    const auto footprint = InputLayout == DataLayout::scatteredV ? ctx.vState.totalBytes :
      bytes * static_cast<size_t>(static_cast<int>(ctx.world));
    if (footprint > ctx.stagingTRSize) {
      using ChunkedCyclicConfig = CollectiveConfig<
        CollectiveType::chunked,
        Policy::CHUNKED_PUT_BLOCKS,
        UNUSED,
        Policy::CHUNK_SIZE,
        UNUSED,
        LAT_THRESHOLD_DEFAULT,
        StagingMode::cyclic
      >;
      const auto cyclicCtx = cyclicContext(ctx, Policy::CHUNK_SIZE, ctx.world);
      rst<InputLayout, PurlinAtomTR, Element, ChunkedCyclicConfig, ro>
        (src, dst, bytes, cyclicCtx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
      return;
    }
    // A separate non-chunked bound lets the chunk size shrink without dragging the
    // band edge down, and gives the edge headroom over reduceScatterV's maxBytes
    // (slightly above the nominal size) so V does not fall one band up.
    constexpr size_t nonChunkedMax = Policy::NON_CHUNKED_MAX_BYTES > 0 ?
      Policy::NON_CHUNKED_MAX_BYTES : Policy::CHUNK_SIZE;
    if constexpr (InputLayout == DataLayout::scattered && multimemReducible<NArch, Element, ro>()) {
      // The multimem reduce pulls every replica through the switch (W·S egress vs
      // the (W-1)·S of unicast reads), so it pays only where instruction efficiency
      // dominates: small worlds cap it via MM_MAX_BYTES (0 disables outright).
      constexpr auto mmMax = cuda::std::min(nonChunkedMax, Policy::MM_MAX_BYTES);
      if (ctx.mcStagingTR != nullptr && bytes % 16 == 0 && dispatchBytes <= mmMax) {
        using TRConfigMM = WithMultimem<TRConfig, Policy::MM_DEPTH>;
        constexpr auto mmConsumers = Policy::MM_CONSUMER_BLOCKS == AUTO ?
          Policy::MAX_CONSUMER_BLOCKS : Policy::MM_CONSUMER_BLOCKS;
        rst<InputLayout, Atom<NArch, TRConfigMM>, Element, NonChunkedConfig, ro>
          (src, dst, bytes, ctx, sizes, mmConsumers, stream);
        return;
      }
    }
    if (dispatchBytes <= nonChunkedMax) {
      rst<InputLayout, PurlinAtomTR, Element, NonChunkedConfig, ro>
        (src, dst, bytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
    else {
      rst<InputLayout, PurlinAtomTR, Element, ChunkedConfig, ro>
        (src, dst, bytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
  }

  template<DataLayout InputLayout, typename Element, int NArch, ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void dispatchReduceScatter(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ sizes, const Context& ctx, cudaStream_t stream) {
    switch (ctx.world) {
      case 2:
        reduceScatterTuned<InputLayout, Element, NArch, 2, ro>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      case 4:
        reduceScatterTuned<InputLayout, Element, NArch, 4, ro>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      case 8:
        reduceScatterTuned<InputLayout, Element, NArch, 8, ro>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      default:
        reduceScatterTuned<InputLayout, Element, NArch, host::FALLBACK, ro>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
    }
  }

  template<int arch, typename Element, ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void reduceScatter(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::reduceScatter", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchReduceScatter<DataLayout::scattered, Element, nArch, ro>
      (src, dst, bytes, bytes, nullptr, ctx, stream);
  }

  template<int arch, typename Element, ReduceOp ro = ReduceOp::add>
  __host__ __forceinline__
  void reduceScatterV(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t* __restrict__ sizes,
    Context& ctx, cudaStream_t stream) {
    const auto bytes = ctx.vState.bytes;
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::reduceScatterV", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    const auto maxBytes = ctx.vState.maxBytes;
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchReduceScatter<DataLayout::scatteredV, Element, nArch, ro>
      (src, dst, bytes, maxBytes, sizes, ctx, stream);
  }
}
#endif //PURLIN_REDUCESCATTER_CUH
