//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ALLGATHER_CUH
#define PURLIN_ALLGATHER_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
namespace purlin::AG {
  __host__ __forceinline__
  constexpr auto getRegime(const size_t& bytesPerRank, const int& world) {
    switch (world) {
      case 4: {
        if (bytesPerRank <= 128 * 1024) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
        break;
      case 8: {
        if (bytesPerRank <= 1024) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
        break;
      default: {
        if (bytesPerRank <= 512 * 1024) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
    }
  }
  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world) {
    int blocks = 0;
    auto blocksNeeded = static_cast<int>(cuda::std::min((bytes / PurlinAtom::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxBlocks)) * world);
    blocksNeeded = bytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
    cuda::std::min(blocksNeeded, 32) : blocksNeeded;
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
  template<typename PurlinAtom, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void allGatherKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx) {
    extern __shared__ __align__(PurlinAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
    purlin::allGather<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, kArgs.bytes, workspace, ctx, kArgs.blocks);
  }

  __host__ __forceinline__
  void allGather(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::allGather", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (bytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<ARCH>();
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    if (AG::getRegime(bytes, ctx.world) == Regime::latency) {
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
      ensureOptIn<allGatherKernel<PurlinAtomLR, CollectiveConfigLR>, kSLR>();
      allGatherKernel<PurlinAtomLR, CollectiveConfigLR><<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>(kArgs, ctx);
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
#else
    const int world = ctx.world;
#endif

    switch (world) {
      case 2: {
        constexpr auto maxSuperBlockSize = 16;
        constexpr auto CHUNK_SIZE = 4 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 16;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        if (bytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
          allGatherKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          if (bytes < 32UL * 1024 * 1024) {
            const auto blocks = AG::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world);
            const Args kArgs{
              .src = src,
              .dst = dst,
              .bytes = bytes,
              .blocks = cuda::fast_mod_div<long int>{blocks}
            };
            ensureOptIn<allGatherKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
            allGatherKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
          }
          else {
            using TRConfig256 = Configuration<
              Regime::throughput,
              256 /*threads*/,
              alignment,
              pipeStages,
              elementsPerThread,
              unrollFactor
            >;
            using PurlinAtomTR256 = Atom<nArch, TRConfig256>;
            constexpr auto kSTR256 = PurlinAtomTR256::COPY_SMEM_SIZE;
            const auto blocks = AG::getBlocks<PurlinAtomTR256>(bytes, putBlocks, maxSuperBlockSize, world);
            const Args kArgs{
              .src = src,
              .dst = dst,
              .bytes = bytes,
              .blocks = cuda::fast_mod_div<long int>{blocks}
            };
            ensureOptIn<allGatherKernel<PurlinAtomTR256, chunkedConfig>, kSTR256>();
            allGatherKernel<PurlinAtomTR256, chunkedConfig><<<blocks, PurlinAtomTR256::THREADS, kSTR256, stream>>>(kArgs, ctx);
          }
        }
      }
        break;
      case 4: {
        constexpr auto maxSuperBlockSize = 8;
        constexpr auto CHUNK_SIZE = 4 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 32;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        if (bytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
          allGatherKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
          allGatherKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
      }
        break;
      default: {
        constexpr auto maxSuperBlockSize = 4;
        constexpr auto CHUNK_SIZE = 4 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 16;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        if (bytes <= CHUNK_SIZE) {
          if (bytes <= 64 * 1024) {
            static_assert(CHUNK_SIZE >= 64 * 1024);
            using TRConfig256 = Configuration<
              Regime::throughput,
              256 /*threads*/,
              alignment,
              pipeStages,
              elementsPerThread,
              unrollFactor
            >;
            using PurlinAtomTR256 = Atom<nArch, TRConfig256>;
            constexpr auto kSTR256 = PurlinAtomTR256::COPY_SMEM_SIZE;
            constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
            const auto blocks = AG::getBlocks<PurlinAtomTR256>(bytes, putBlocks, maxSuperBlockSize, world);
            const Args kArgs{
              .src = src,
              .dst = dst,
              .bytes = bytes,
              .blocks = cuda::fast_mod_div<long int>{blocks}
            };
            ensureOptIn<allGatherKernel<PurlinAtomTR256, nonChunkedConfig>, kSTR256>();
            allGatherKernel<PurlinAtomTR256, nonChunkedConfig><<<blocks, PurlinAtomTR256::THREADS, kSTR256, stream>>>(kArgs, ctx);
          }
          else {
            constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
            const auto blocks = AG::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world);
            const Args kArgs{
              .src = src,
              .dst = dst,
              .bytes = bytes,
              .blocks = cuda::fast_mod_div<long int>{blocks}
            };
            ensureOptIn<allGatherKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
            allGatherKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
          }
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
          allGatherKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, ctx);
        }
      }
    }
  }

  template<typename PurlinAtom, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void allGatherVKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx,
    const size_t* __restrict__ sizes) {
    extern __shared__ __align__(PurlinAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
    purlin::allGatherV<PurlinAtom, CollConfig>(kArgs.dst, kArgs.src, sizes, workspace, ctx, kArgs.blocks);
  }

  __host__ __forceinline__
  void allGatherV(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t* __restrict__ sizes,
    const Context& ctx, cudaStream_t stream) {
    const auto bytes = ctx.vState.bytes;
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::allGatherV", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    const auto maxBytes = ctx.vState.maxBytes;
    if (maxBytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<ARCH>();
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    if (AG::getRegime(maxBytes, ctx.world) == Regime::latency) {
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
      constexpr auto kSLR = PurlinAtomLR::COPY_SMEM_SIZE;
      const Args kArgs{
        .src = src,
        .dst = dst,
        .blocks = cuda::fast_mod_div<long int>{blocks}
      };
      ensureOptIn<allGatherVKernel<PurlinAtomLR, CollectiveConfigLR>, kSLR>();
      allGatherVKernel<PurlinAtomLR, CollectiveConfigLR><<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>
      (kArgs, ctx, sizes);
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
#else
    const int world = ctx.world;
#endif

    switch (world) {
      case 2: {
        constexpr auto maxSuperBlockSize = 16;
        constexpr auto CHUNK_SIZE = 4 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 16;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        if (maxBytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(maxBytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherVKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
          allGatherVKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(maxBytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherVKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
          allGatherVKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
      }
        break;
      case 4: {
        constexpr auto maxSuperBlockSize = 8;
        constexpr auto CHUNK_SIZE = 4 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 32;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        if (maxBytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(maxBytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherVKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
          allGatherVKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(maxBytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherVKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
          allGatherVKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
      }
        break;
      default: {
        constexpr auto maxSuperBlockSize = 4;
        constexpr auto CHUNK_SIZE = 4 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 16;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          UNUSED,
          CHUNK_SIZE,
          UNUSED
        >;
        constexpr auto kSTR = PurlinAtomTR::COPY_SMEM_SIZE;
        if (maxBytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(maxBytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherVKernel<PurlinAtomTR, nonChunkedConfig>, kSTR>();
          allGatherVKernel<PurlinAtomTR, nonChunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
        else {
          constexpr auto putBlocks = chunkedConfig::PUT_BLOCKS;
          const auto blocks = AG::getBlocks<PurlinAtomTR>(maxBytes, putBlocks, maxSuperBlockSize, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allGatherVKernel<PurlinAtomTR, chunkedConfig>, kSTR>();
          allGatherVKernel<PurlinAtomTR, chunkedConfig><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx, sizes);
        }
      }
    }
  }
}
#endif //PURLIN_ALLGATHER_CUH
