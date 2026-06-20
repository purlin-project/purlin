//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ALL2ALL_CUH
#define PURLIN_ALL2ALL_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
namespace purlin::A2A {
  __host__ __forceinline__
  Regime getRegime(const size_t& bytesPerRank, const int& world) {
    switch (world) {
      case 4: {
        if (bytesPerRank <= 256 * 1024) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
        break;
      case 8: {
        if (bytesPerRank <= 4 * 1024) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
        break;
      default: {
        if (bytesPerRank <= RED_LATENCY_BOUND_THRESHOLD) {
          return Regime::latency;
        }
        return Regime::throughput;
      }
    }
  }

  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world, 
    const int& actualWorld) {
    int blocks = 0;
    auto blocksNeeded = static_cast<int>(cute::min((bytes / PurlinAtom::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxBlocks)) * actualWorld);
    blocksNeeded = bytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
    cuda::round_down(cute::min(blocksNeeded, 32), actualWorld) : blocksNeeded;
    blocks = putBlocks + blocksNeeded;
    if (blocksNeeded < actualWorld) {
      // non-pipelined path
      blocks = putBlocks + (cute::min(cuda::ceil_div(bytes,
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
    const int actualWorld = ctx.actualWorld;
    const int world = ctx.world;

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
          const auto putBlocks = stagingBlocks + nonChunkedConfig::PUT_BLOCKS;
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
          const auto putBlocks = stagingBlocks + chunkedConfig::PUT_BLOCKS;
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
          const auto putBlocks = stagingBlocks + nonChunkedConfig::PUT_BLOCKS;
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
          const auto putBlocks = stagingBlocks + chunkedConfig::PUT_BLOCKS;
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
          const auto putBlocks = stagingBlocks + nonChunkedConfig::PUT_BLOCKS;
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
          const auto putBlocks = stagingBlocks + chunkedConfig::PUT_BLOCKS;
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
}
#endif //PURLIN_ALL2ALL_CUH
