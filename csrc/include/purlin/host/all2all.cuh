//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ALL2ALL_CUH
#define PURLIN_ALL2ALL_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "tuning.cuh"
namespace purlin::A2A {
  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world, 
    const int& actualWorld) {
    int blocks = 0;
    auto blocksNeeded = static_cast<int>(cuda::std::min((bytes / PurlinAtom::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxBlocks)) * actualWorld);
    blocksNeeded = bytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
    cuda::round_down(cuda::std::min(blocksNeeded, 32), actualWorld) : blocksNeeded;
    blocks = putBlocks + blocksNeeded;
    if (blocksNeeded < actualWorld) {
      // non-pipelined path
      blocks = putBlocks + (cuda::std::min(cuda::ceil_div(bytes,
        static_cast<size_t>(PurlinAtom::THREADS*PurlinAtom::BaseConfig::ALIGNMENT_BYTES)),
        static_cast<size_t>(maxBlocks)) * actualWorld);
    }
    return blocks;
  }
}
namespace purlin {
  template<typename PurlinAtom, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void all2allKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx) {
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    purlin::all2all<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, kArgs.bytes, workspace, ctx, kArgs.blocks);
  }

  template<typename PurlinAtom, typename CollConfig>
  __host__ __forceinline__
  void launchAll2AllThroughput(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, Context& ctx,
    const int& targetPutBlocks, const int& maxConsumerBlocks, cudaStream_t stream) {
    const int actualWorld = ctx.actualWorld;
    const auto putBlocksPerPeer = cuda::std::bit_floor(static_cast<uint32_t>(
      cuda::round_down(targetPutBlocks, actualWorld) / actualWorld));
    const auto stagingBlocks = putBlocksPerPeer * actualWorld;
    const auto putBlocks = stagingBlocks + CollConfig::LOCAL_PUT_BLOCKS;
    ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
    const auto blocks = A2A::getBlocks<PurlinAtom>(
      bytes, putBlocks, maxConsumerBlocks, ctx.world, actualWorld);
    constexpr auto kS = PurlinAtom::COPY_SMEM_SIZE;
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ensureOptIn<all2allKernel<PurlinAtom, CollConfig>, kS>();
    all2allKernel<PurlinAtom, CollConfig><<<blocks, PurlinAtom::THREADS, kS, stream>>>(kArgs, ctx);
  }

  template<int NArch, int World>
  __host__ __forceinline__
  void all2allTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    constexpr auto tArch = host::tuningArch<NArch>;
    using Policy = host::All2AllTuning<tArch, World>;

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
      constexpr auto kS = PurlinAtomLR::COPY_SMEM_SIZE;
      const Args kArgs{
        .src = src,
        .dst = dst,
        .bytes = bytes,
        .blocks = cuda::fast_mod_div<long int>{blocks}
      };
      ensureOptIn<all2allKernel<PurlinAtomLR, CollectiveConfigLR>, kS>();
      all2allKernel<PurlinAtomLR, CollectiveConfigLR><<<blocks, PurlinAtomLR::THREADS, kS, stream>>>(kArgs, ctx);
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
      UNUSED,
      UNUSED,
      Policy::CHUNK_SIZE,
      Policy::LOCAL_PUT_BLOCKS
    >;
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      UNUSED,
      UNUSED,
      Policy::CHUNK_SIZE,
      Policy::LOCAL_PUT_BLOCKS
    >;
    if (bytes <= Policy::CHUNK_SIZE) {
      launchAll2AllThroughput<PurlinAtomTR, NonChunkedConfig>(
        src, dst, bytes, ctx, Policy::NON_CHUNKED_PUT_BLOCKS, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
    else {
      launchAll2AllThroughput<PurlinAtomTR, ChunkedConfig>(
        src, dst, bytes, ctx, Policy::CHUNKED_PUT_BLOCKS, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
  }

  __host__ __forceinline__
  void all2all(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::all2all", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (ctx.world * bytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<ARCH>();
#if defined(PURLIN_JIT_WORLD)
    static_assert(cuda::std::is_integral_v<decltype(PURLIN_JIT_WORLD)>);
    static_assert(PURLIN_JIT_WORLD == 2 || PURLIN_JIT_WORLD == 4 || PURLIN_JIT_WORLD == 8);
    all2allTuned<nArch, PURLIN_JIT_WORLD>(src, dst, bytes, ctx, stream);
#else
    const int world = ctx.world;
    switch (world) {
      case 2: all2allTuned<nArch, 2>(src, dst, bytes, ctx, stream); break;
      case 4: all2allTuned<nArch, 4>(src, dst, bytes, ctx, stream); break;
      default: all2allTuned<nArch, 8>(src, dst, bytes, ctx, stream); break;
    }
#endif
  }

  template<typename PurlinAtom, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void all2allVKernel(const __grid_constant__ Args kArgs,
    const size_t* __restrict__ inSplits, const size_t* __restrict__ outSplits,
    const __grid_constant__ Context ctx) {
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte vWorkspace[];
    purlin::all2allV<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, inSplits, outSplits, vWorkspace, ctx, kArgs.blocks);
  }

  __host__ __forceinline__
  void all2allV(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t* __restrict__ const& inSplits,
    const size_t* __restrict__ const& outSplits, Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::all2allV", nvtx3::payload{static_cast<uint64_t>(ctx.vState.totalBytes)}};
#endif
    if (ctx.vState.totalBytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<ARCH>();
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
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
#if defined(PURLIN_JIT_WORLD)
    static_assert(cuda::std::is_integral_v<decltype(PURLIN_JIT_WORLD)>);
    constexpr int world = PURLIN_JIT_WORLD; // <- may help reduce compilation times
    constexpr int actualWorld = world - 1;
#else
    const int world = ctx.world;
    const int actualWorld = ctx.actualWorld;
#endif
    switch (world) {
      case 2: {
        constexpr auto maxSuperBlockSize = 32;
        constexpr auto CHUNK_SIZE = 2 * 1024 * 1024;
        constexpr auto LOCAL_PUT_BLOCKS = 8;
        constexpr int PUT_BLOCKS = 32;
        using collConfig = CollectiveConfig<
          CollectiveType::chunked,
          UNUSED,
          UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        const auto pb = cuda::std::bit_floor(static_cast<uint32_t>(
          cuda::round_down(PUT_BLOCKS, actualWorld) / actualWorld));

        const auto stagingBlocks = pb * actualWorld;
        const auto putBlocks = stagingBlocks + collConfig::LOCAL_PUT_BLOCKS;
        ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
        const auto blocks = A2A::getBlocks<PurlinAtomTR>(ctx.vState.maxOutBytes, putBlocks, maxSuperBlockSize, world, actualWorld);
        const Args kArgs{
          .src = src,
          .dst = dst,
          .blocks = cuda::fast_mod_div<long int>{blocks}
        };
        ensureOptIn<all2allVKernel<PurlinAtomTR, collConfig>, kSTR>();
        all2allVKernel<PurlinAtomTR, collConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
        (kArgs, inSplits, outSplits, ctx);
      }
        break;
      case 4: {
        constexpr auto maxSuperBlockSize = 8;
        constexpr auto CHUNK_SIZE = 1 * 1024 * 1024;
        constexpr auto LOCAL_PUT_BLOCKS = 8;
        constexpr int PUT_BLOCKS = 32;
        using collConfig = CollectiveConfig<
          CollectiveType::chunked,
          UNUSED,
          UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        const auto pb = cuda::std::bit_floor(static_cast<uint32_t>(
          cuda::round_down(PUT_BLOCKS, actualWorld) / actualWorld));

        const auto stagingBlocks = pb * actualWorld;
        const auto putBlocks = stagingBlocks + collConfig::LOCAL_PUT_BLOCKS;
        ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
        const auto blocks = A2A::getBlocks<PurlinAtomTR>(ctx.vState.maxOutBytes, putBlocks, maxSuperBlockSize, world, actualWorld);
        const Args kArgs{
          .src = src,
          .dst = dst,
          .blocks = cuda::fast_mod_div<long int>{blocks}
        };
        ensureOptIn<all2allVKernel<PurlinAtomTR, collConfig>, kSTR>();
        all2allVKernel<PurlinAtomTR, collConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
        (kArgs, inSplits, outSplits, ctx);
      }
        break;
      default: {
        constexpr auto maxSuperBlockSize = 4;
        constexpr auto CHUNK_SIZE = 1 * 1024 * 1024;
        constexpr auto LOCAL_PUT_BLOCKS = 4;
        constexpr int PUT_BLOCKS = 32;
        using collConfig = CollectiveConfig<
          CollectiveType::chunked,
          UNUSED,
          UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        const auto pb = cuda::std::bit_floor(static_cast<uint32_t>(
          cuda::round_down(PUT_BLOCKS, actualWorld) / actualWorld));

        const auto stagingBlocks = pb * actualWorld;
        const auto putBlocks = stagingBlocks + collConfig::LOCAL_PUT_BLOCKS;
        ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
        const auto blocks = A2A::getBlocks<PurlinAtomTR>(ctx.vState.maxOutBytes, putBlocks, maxSuperBlockSize, world, actualWorld);
        const Args kArgs{
          .src = src,
          .dst = dst,
          .blocks = cuda::fast_mod_div<long int>{blocks}
        };
        ensureOptIn<all2allVKernel<PurlinAtomTR, collConfig>, kSTR>();
        all2allVKernel<PurlinAtomTR, collConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
        (kArgs, inSplits, outSplits, ctx);
      }
    }
  }
}
#endif //PURLIN_ALL2ALL_CUH
