//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ALL2ALL_CUH
#define PURLIN_ALL2ALL_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
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
        maxBlocks) * actualWorld);
    }
    return blocks;
  }
}
namespace purlin {
  template<typename PurlinAtom, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void all2allKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx) {
    extern __shared__ __align__(PurlinAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
    purlin::all2all<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, kArgs.bytes, workspace, ctx, kArgs.blocks);
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
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    if (A2A::getRegime(bytes, ctx.world) == Regime::latency) {
      using LRConfig = Configuration<
        Regime::latency,
        512, /*threads*/
        alignment,
        UNUSED,
        UNUSED,
        unrollFactor
      >;
      using PurlinAtomLR = Atom<nArch, LRConfig>;
      const auto blocks = getLRBlocks<PurlinAtomLR::THREADS>(bytes);
      constexpr auto kSLR = PurlinAtomLR::COPY_SMEM_SIZE;
      const Args kArgs{
        .src = src,
        .dst = dst,
        .bytes = bytes,
        .blocks = cuda::fast_mod_div<long int>{blocks}
      };
      ensureOptIn<all2allKernel<PurlinAtomLR, CollectiveConfigLR>, kSLR>();
      all2allKernel<PurlinAtomLR, CollectiveConfigLR><<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>(kArgs, ctx);
      return;
    }
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
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 32;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          UNUSED,
          UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          UNUSED,
          UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        const auto nNonChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
        cuda::round_down(NON_CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
        const auto nChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
          cuda::round_down(CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
        if (bytes <= CHUNK_SIZE) {
          const auto stagingBlocks = nNonChunkedPB * actualWorld;
          const auto putBlocks = stagingBlocks + nonChunkedConfig::LOCAL_PUT_BLOCKS;
          ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
          const auto blocks = A2A::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world, actualWorld);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<all2allKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
          all2allKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
        else {
          const auto stagingBlocks = nChunkedPB * actualWorld;
          const auto putBlocks = stagingBlocks + chunkedConfig::LOCAL_PUT_BLOCKS;
          ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
          const auto blocks = A2A::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world, actualWorld);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<all2allKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
          all2allKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
      }
        break;
      case 4: {
        constexpr auto maxSuperBlockSize = 8;
        constexpr auto CHUNK_SIZE = 1 * 1024 * 1024;
        constexpr auto LOCAL_PUT_BLOCKS = 4;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 32;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          UNUSED,
          UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          purlin::UNUSED,
          purlin::UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        const auto nNonChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
        cuda::round_down(NON_CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
        const auto nChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
          cuda::round_down(CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
        if (bytes <= CHUNK_SIZE) {
          const auto stagingBlocks = nNonChunkedPB * actualWorld;
          const auto putBlocks = stagingBlocks + nonChunkedConfig::LOCAL_PUT_BLOCKS;
          ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
          const auto blocks = A2A::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world, actualWorld);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<all2allKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
          all2allKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
        else {
          const auto stagingBlocks = nChunkedPB * actualWorld;
          const auto putBlocks = stagingBlocks + chunkedConfig::LOCAL_PUT_BLOCKS;
          ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
          const auto blocks = A2A::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world, actualWorld);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<all2allKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
          all2allKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
      }
        break;
      default: {
        constexpr auto maxSuperBlockSize = 4;
        constexpr auto CHUNK_SIZE = 1 * 1024 * 1024;
        constexpr auto LOCAL_PUT_BLOCKS = 4;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 32;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          UNUSED,
          UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          purlin::UNUSED,
          purlin::UNUSED,
          CHUNK_SIZE,
          LOCAL_PUT_BLOCKS
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        const auto nNonChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
        cuda::round_down(NON_CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
        const auto nChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
          cuda::round_down(CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
        if (bytes <= CHUNK_SIZE) {
          const auto stagingBlocks = nNonChunkedPB * actualWorld;
          const auto putBlocks = stagingBlocks + nonChunkedConfig::LOCAL_PUT_BLOCKS;
          ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
          const auto blocks = A2A::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world, actualWorld);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<all2allKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
          all2allKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
        else {
          const auto stagingBlocks = nChunkedPB * actualWorld;
          const auto putBlocks = stagingBlocks + chunkedConfig::LOCAL_PUT_BLOCKS;
          ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
          const auto blocks = A2A::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world, actualWorld);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<all2allKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
          all2allKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
      }
    }
  }

  template<typename PurlinAtom, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void all2allVKernel(const __grid_constant__ Args kArgs,
    const size_t* __restrict__ inSplits, const size_t* __restrict__ outSplits,
    const __grid_constant__ Context ctx) {
    extern __shared__ __align__(PurlinAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
    purlin::all2allV<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, inSplits, outSplits, workspace, ctx, kArgs.blocks);
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
