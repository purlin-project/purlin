//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_REDUCESCATTER_CUH
#define PURLIN_REDUCESCATTER_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "tuning.cuh"

namespace purlin::RS {
  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world) {
    int blocks = 0;
    auto blocksNeeded = cuda::std::min(bytes / PurlinAtom::RED_PIPELINE_BYTES,
        bytes / (world * PurlinAtom::STAGE_BYTES));
    blocksNeeded = static_cast<int>(cuda::std::min(blocksNeeded,static_cast<size_t>(maxBlocks)));
    blocks = putBlocks + blocksNeeded;
    if (blocksNeeded < 1) {
      // non-pipelined path
      blocks = putBlocks + cuda::std::min(cuda::ceil_div(bytes / world,
        PurlinAtom::THREADS*PurlinAtom::BaseConfig::ALIGNMENT_BYTES), static_cast<size_t>(maxBlocks));
    }
    return blocks;
  }
}

namespace purlin {
  template<typename PurlinAtom, typename Element, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void reduceScatterKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx) {
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
    purlin::reduceScatter<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx, kArgs.blocks);
  }

  template<typename PurlinAtom, typename Element, typename CollConfig>
  __host__ __forceinline__
  void launchReduceScatterThroughput(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const int& maxReduceBlocks, cudaStream_t stream) {
    constexpr auto kS = cuda::std::max(PurlinAtom::COPY_SMEM_SIZE, PurlinAtom::RED_SMEM_SIZE);
    constexpr auto putBlocks = CollConfig::PUT_BLOCKS;
    const auto blocks = RS::getBlocks<PurlinAtom>(bytes, putBlocks, maxReduceBlocks, ctx.world);
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ensureOptIn<reduceScatterKernel<PurlinAtom, Element, CollConfig>, kS>();
    reduceScatterKernel<PurlinAtom, Element, CollConfig>
      <<<blocks, PurlinAtom::THREADS, kS, stream>>>(kArgs, ctx);
  }

  template<typename Element, int NArch, int World>
  __host__ __forceinline__
  void reduceScatterTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    constexpr auto tArch = host::tuningArch<NArch>;
    using Policy = host::ReduceScatterTuning<tArch, World>;

    if (bytes <= Policy::LATENCY_THRESHOLD) {
      using LRConfig = Configuration<
        Regime::latency,
        Policy::LR_THREADS,
        alignment,
        UNUSED,
        UNUSED,
        unrollFactor
      >;
      using PurlinAtomLR = Atom<NArch, LRConfig>;
      const auto blocks = getLRBlocks<PurlinAtomLR::THREADS>(bytes);
      constexpr auto kS = PurlinAtomLR::RED_SMEM_SIZE;
      const Args kArgs{
        .src = src,
        .dst = dst,
        .bytes = bytes,
        .blocks = cuda::fast_mod_div<long int>{blocks}
      };
      ensureOptIn<reduceScatterKernel<PurlinAtomLR, Element, CollectiveConfigLR>, kS>();
      reduceScatterKernel<PurlinAtomLR, Element, CollectiveConfigLR>
        <<<blocks, PurlinAtomLR::THREADS, kS, stream>>>(kArgs, ctx);
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
      Policy::CHUNK_SIZE
    >;
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      UNUSED,
      Policy::CHUNK_SIZE
    >;
    if (bytes <= Policy::CHUNK_SIZE) {
      launchReduceScatterThroughput<PurlinAtomTR, Element, NonChunkedConfig>(
        src, dst, bytes, ctx, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
    else {
      launchReduceScatterThroughput<PurlinAtomTR, Element, ChunkedConfig>(
        src, dst, bytes, ctx, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
  }

  template<typename Element>
  __host__ __forceinline__
  constexpr void reduceScatter(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::reduceScatter", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (ctx.world * bytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<ARCH>();
#if defined(PURLIN_JIT_WORLD)
    static_assert(cuda::std::is_integral_v<decltype(PURLIN_JIT_WORLD)>);
    static_assert(PURLIN_JIT_WORLD == 2 || PURLIN_JIT_WORLD == 4 || PURLIN_JIT_WORLD == 8);
    reduceScatterTuned<Element, nArch, PURLIN_JIT_WORLD>(src, dst, bytes, ctx, stream);
#else
    const int world = ctx.world;
    switch (world) {
      case 2: reduceScatterTuned<Element, nArch, 2>(src, dst, bytes, ctx, stream); break;
      case 4: reduceScatterTuned<Element, nArch, 4>(src, dst, bytes, ctx, stream); break;
      default: reduceScatterTuned<Element, nArch, 8>(src, dst, bytes, ctx, stream); break;
    }
#endif
  }

  template<typename PurlinAtom, typename Element, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void reduceScatterVKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx,
    const size_t* __restrict__ sizes) {
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte vWorkspace[];
    auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(vWorkspace);
    purlin::reduceScatterV<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, sizes, typedWorkspace, ctx, kArgs.blocks);
  }
  template<typename Element>
  __host__ __forceinline__
  constexpr void reduceScatterV(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t* __restrict__ sizes,
    Context& ctx, cudaStream_t stream) {
    const auto bytes = ctx.vState.bytes;
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::reduceScatterV", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    const auto totalBytes = ctx.vState.totalBytes;
    const auto maxBytes = ctx.vState.maxBytes;
    if (totalBytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<ARCH>();
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    if (RS::getRegime(maxBytes, ctx.world) == Regime::latency) {
      using LRConfig = Configuration<
        Regime::latency,
        512, /*threads*/
        alignment,
        UNUSED,
        UNUSED,
        unrollFactor
      >;
      using PurlinAtomLR = Atom<nArch, LRConfig>;
      const auto blocks = getLRBlocks<PurlinAtomLR::THREADS>(maxBytes);
      constexpr auto kSLR = PurlinAtomLR::RED_SMEM_SIZE;
      const Args kArgs{
        .src = src,
        .dst = dst,
        .bytes = bytes,
        .blocks = cuda::fast_mod_div<long int>{blocks}
      };
      ensureOptIn<reduceScatterVKernel<PurlinAtomLR, Element, CollectiveConfigLR>, kSLR>();
      reduceScatterVKernel<PurlinAtomLR, Element, CollectiveConfigLR><<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>
      (kArgs, ctx, sizes);
      return;
    }
#if defined(PURLIN_JIT_WORLD)
    static_assert(cuda::std::is_integral_v<decltype(PURLIN_JIT_WORLD)>);
    constexpr int world = PURLIN_JIT_WORLD; // <- may help reduce compilation times
#else
    const int world = ctx.world;
#endif
    switch (world) {
      case 2: {
        constexpr auto threads = 256;
        constexpr auto pipeStages = 8;
        constexpr auto elementsPerThread = 2;
        using TRConfig = Configuration<
            Regime::throughput,
            threads,
            alignment,
            pipeStages,
            elementsPerThread,
            unrollFactor
        >;
        using PurlinAtomTR = Atom<nArch, TRConfig>;
        constexpr auto maxReduceBlocks = 16;
        constexpr auto CHUNK_SIZE = 4 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 16;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE
        >;
        constexpr auto kSTR = cuda::std::max(PurlinAtomTR::COPY_SMEM_SIZE, PurlinAtomTR::RED_SMEM_SIZE);
        if (maxBytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = RS::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<reduceScatterVKernel<PurlinAtomTR, Element, nonChunkedConfig>, kSTR>();
          reduceScatterVKernel<PurlinAtomTR, Element, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          const auto blocks = RS::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<reduceScatterVKernel<PurlinAtomTR, Element, chunkedConfig>, kSTR>();
          reduceScatterVKernel<PurlinAtomTR, Element, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
      }
        break;
      case 4: {
        constexpr auto threads = 128;
        constexpr auto pipeStages = 8;
        constexpr auto elementsPerThread = 2;
        using TRConfig = Configuration<
            Regime::throughput,
            threads,
            alignment,
            pipeStages,
            elementsPerThread,
            unrollFactor
        >;
        using PurlinAtomTR = Atom<nArch, TRConfig>;
        constexpr auto maxReduceBlocks = 32;
        constexpr auto CHUNK_SIZE = 2 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 16;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE
        >;
        constexpr auto kSTR = cuda::std::max(PurlinAtomTR::COPY_SMEM_SIZE, PurlinAtomTR::RED_SMEM_SIZE);
        if (maxBytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = RS::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<reduceScatterVKernel<PurlinAtomTR, Element, nonChunkedConfig>, kSTR>();
          reduceScatterVKernel<PurlinAtomTR, Element, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          const auto blocks = RS::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<reduceScatterVKernel<PurlinAtomTR, Element, chunkedConfig>, kSTR>();
          reduceScatterVKernel<PurlinAtomTR, Element, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
      }
        break;
      default: {
        constexpr auto threads = 128;
        constexpr auto pipeStages = 8;
        constexpr auto elementsPerThread = 2;
        using TRConfig = Configuration<
            Regime::throughput,
            threads,
            alignment,
            pipeStages,
            elementsPerThread,
            unrollFactor
        >;
        using PurlinAtomTR = Atom<nArch, TRConfig>;
        constexpr auto maxReduceBlocks = 32;
        constexpr auto CHUNK_SIZE = 2 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 32;
        if (CHUNKED_PUT_BLOCKS % ctx.world != 0) {
          throw std::runtime_error("As of yet, we expect " +
            std::to_string(CHUNKED_PUT_BLOCKS) + "% " + std::to_string(ctx.world) + " == 0");
        }
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE
        >;
        constexpr auto kSTR = cuda::std::max(PurlinAtomTR::COPY_SMEM_SIZE, PurlinAtomTR::RED_SMEM_SIZE);
        if (maxBytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = RS::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<reduceScatterVKernel<PurlinAtomTR, Element, nonChunkedConfig>, kSTR>();
          reduceScatterVKernel<PurlinAtomTR, Element, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          const auto blocks = RS::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<reduceScatterVKernel<PurlinAtomTR, Element, chunkedConfig>, kSTR>();
          reduceScatterVKernel<PurlinAtomTR, Element, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
      }
    }
  }
}
#endif //PURLIN_REDUCESCATTER_CUH
