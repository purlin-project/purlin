//
// Created by osayamen on 5/28/26.
//

#ifndef SUTURE_ALLREDUCE_CUH
#define SUTURE_ALLREDUCE_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
namespace suture::AR {
  __host__ __forceinline__
  constexpr auto getRegime(const size_t& bytesPerRank, const int& world) {
    const size_t threshold = ((8 / world) * 128 * 1024);
    if (bytesPerRank <= threshold) {
      return Regime::latency;
    }
    return Regime::throughput;
  }
  template<typename SutureAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world) {
    int blocks = 0;
    auto blocksNeeded = cute::min(bytes / SutureAtom::RED_PIPELINE_BYTES,
        bytes / (world * SutureAtom::STAGE_BYTES));
    blocksNeeded = static_cast<int>(min(blocksNeeded,static_cast<size_t>(maxBlocks)));
    blocks = putBlocks + blocksNeeded;
    if (blocksNeeded < 1) {
      // non-pipelined path
      blocks = putBlocks + cute::min(cuda::ceil_div(bytes / world,
        SutureAtom::THREADS*SutureAtom::BaseConfig::ALIGNMENT_BYTES), maxBlocks);
    }
    return blocks;
  }
}

namespace suture {
  template<typename SutureAtom, typename Element, typename CollConfig>
  __launch_bounds__(SutureAtom::THREADS, 1)
  __global__ void allReduceKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx) {
    extern __shared__ __align__(SutureAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
    auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
    suture::allReduce<SutureAtom, CollConfig>(kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx, kArgs.blocks);
  }

  template<typename Element>
  __host__ __forceinline__
  void allReduce(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(SUTURE_NVTX) && SUTURE_NVTX
    const SutureRange range{"suture::allReduce", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (bytes > STAGING_BUFFER_SIZE_) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = suture::normalizeArch<ARCH>();
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    if (AR::getRegime(bytes, ctx.world) == Regime::latency) {
      using LRConfig = Configuration<
        nArch,
        Regime::latency,
        512, /*threads*/
        alignment,
        UNUSED,
        UNUSED,
        unrollFactor
      >;
      using SutureAtomLR = Atom<nArch, LRConfig>;
      const auto blocks = getLRBlocks<SutureAtomLR::THREADS>(bytes);
      constexpr auto kSLR = SutureAtomLR::RED_SMEM_SIZE;
      const Args kArgs{
        .src = src,
        .dst = dst,
        .bytes = bytes,
        .blocks = cuda::fast_mod_div<long int>{blocks}
      };
      ensureOptIn<allReduceKernel<SutureAtomLR, Element, CollectiveConfigLR>, kSLR>();
      allReduceKernel<SutureAtomLR, Element, CollectiveConfigLR><<<blocks, SutureAtomLR::THREADS, kSLR, stream>>>
      (kArgs, ctx);
      return;
    }
    const int world = ctx.world;
    switch (world) {
      case 2: {
        constexpr auto threads = 256;
        constexpr auto pipeStages = 8;
        constexpr auto elementsPerThread = 2;
        using TRConfig = Configuration<
            nArch,
            Regime::throughput,
            threads,
            alignment,
            pipeStages,
            elementsPerThread,
            unrollFactor
        >;
        using SutureAtomTR = Atom<nArch, TRConfig>;
        constexpr auto maxReduceBlocks = 16;
        constexpr auto CHUNK_SIZE = 4 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 32;
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
        constexpr auto kSTR = cute::max(SutureAtomTR::COPY_SMEM_SIZE, SutureAtomTR::RED_SMEM_SIZE);
        if (bytes <= CHUNK_SIZE) {
          constexpr auto putBlocks =  nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = AR::getBlocks<SutureAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<SutureAtomTR, Element, nonChunkedConfig>, kSTR>();
          allReduceKernel<SutureAtomTR, Element, nonChunkedConfig><<<blocks, SutureAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
        else {
          constexpr auto putBlocks =  chunkedConfig::PUT_BLOCKS;
          const auto blocks = AR::getBlocks<SutureAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<SutureAtomTR, Element, chunkedConfig>, kSTR>();
          allReduceKernel<SutureAtomTR, Element, chunkedConfig><<<blocks, SutureAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
      }
        break;
      case 4: {
        constexpr auto threads = 256;
        constexpr auto pipeStages = 8;
        constexpr auto elementsPerThread = 1;
        using TRConfig = Configuration<
            nArch,
            Regime::throughput,
            threads,
            alignment,
            pipeStages,
            elementsPerThread,
            unrollFactor
        >;
        using SutureAtomTR = Atom<nArch, TRConfig>;
        constexpr auto maxReduceBlocks = 32;
        constexpr auto CHUNK_SIZE = 1 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 32;
        constexpr int GATHER_BLOCKS = 16;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          GATHER_BLOCKS,
          CHUNK_SIZE
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          GATHER_BLOCKS,
          CHUNK_SIZE
        >;
        constexpr auto kSTR = cute::max(SutureAtomTR::COPY_SMEM_SIZE, SutureAtomTR::RED_SMEM_SIZE);
        if (bytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS + GATHER_BLOCKS;
          const auto blocks = AR::getBlocks<SutureAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<SutureAtomTR, Element, nonChunkedConfig>, kSTR>();
          allReduceKernel<SutureAtomTR, Element, nonChunkedConfig><<<blocks, SutureAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
        else {
          constexpr auto putBlocks =  chunkedConfig::PUT_BLOCKS + GATHER_BLOCKS;
          const auto blocks = AR::getBlocks<SutureAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<SutureAtomTR, Element, chunkedConfig>, kSTR>();
          allReduceKernel<SutureAtomTR, Element, chunkedConfig><<<blocks, SutureAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
      }
        break;
      default: {
        constexpr auto threads = 256;
        constexpr auto pipeStages = 8;
        constexpr auto elementsPerThread = 1;
        using TRConfig = Configuration<
            nArch,
            Regime::throughput,
            threads,
            alignment,
            pipeStages,
            elementsPerThread,
            unrollFactor
        >;
        using SutureAtomTR = Atom<nArch, TRConfig>;
        constexpr auto maxReduceBlocks = 32;
        constexpr auto CHUNK_SIZE = 1 * 1024 * 1024;
        constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
        constexpr int CHUNKED_PUT_BLOCKS = 16;
        constexpr int GATHER_BLOCKS = 16;
        using nonChunkedConfig = CollectiveConfig<
          CollectiveType::nonChunked,
          NON_CHUNKED_PUT_BLOCKS,
          GATHER_BLOCKS,
          CHUNK_SIZE
        >;
        using chunkedConfig = CollectiveConfig<
          CollectiveType::chunked,
          CHUNKED_PUT_BLOCKS,
          GATHER_BLOCKS,
          CHUNK_SIZE
        >;
        constexpr auto kSTR = cute::max(SutureAtomTR::COPY_SMEM_SIZE, SutureAtomTR::RED_SMEM_SIZE);
        if (bytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS + GATHER_BLOCKS;
          const auto blocks = AR::getBlocks<SutureAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<SutureAtomTR, Element, nonChunkedConfig>, kSTR>();
          allReduceKernel<SutureAtomTR, Element, nonChunkedConfig><<<blocks, SutureAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
        else {
          constexpr auto putBlocks =  chunkedConfig::PUT_BLOCKS + GATHER_BLOCKS;
          const auto blocks = AR::getBlocks<SutureAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<SutureAtomTR, Element, chunkedConfig>, kSTR>();
          allReduceKernel<SutureAtomTR, Element, chunkedConfig><<<blocks, SutureAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
      }
    }
  }
}
#endif //SUTURE_ALLREDUCE_CUH
