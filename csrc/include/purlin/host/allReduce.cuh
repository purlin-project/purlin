//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ALLREDUCE_CUH
#define PURLIN_ALLREDUCE_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
namespace purlin::AR {
  __host__ __forceinline__
  constexpr auto getRegime(const size_t& bytesPerRank, const int& world) {
    const size_t threshold = ((8 / world) * 128 * 1024);
    if (bytesPerRank <= threshold) {
      return Regime::latency;
    }
    return Regime::throughput;
  }
  template<typename PurlinAtom>
  __host__ __forceinline__
  constexpr auto getBlocks(const size_t& bytes, const int& putBlocks, const int& maxBlocks, const int& world) {
    int blocks = 0;
    auto blocksNeeded = cuda::std::min(bytes / PurlinAtom::RED_PIPELINE_BYTES,
        bytes / (world * PurlinAtom::STAGE_BYTES));
    blocksNeeded = static_cast<int>(min(blocksNeeded,static_cast<size_t>(maxBlocks)));
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
  template<typename PurlinAtom, typename Element, typename CollConfig, World2Bypass wb = World2Bypass::unknown>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void allReduceKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx) {
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
    purlin::allReduce<PurlinAtom, CollConfig, wb>(kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx, kArgs.blocks);
  }

  template<typename Element>
  __host__ __forceinline__
  void allReduce(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::allReduce", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (bytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<ARCH>();
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    if (AR::getRegime(bytes, ctx.world) == Regime::latency) {
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
      constexpr auto kSLR = PurlinAtomLR::RED_SMEM_SIZE;
      const Args kArgs{
        .src = src,
        .dst = dst,
        .bytes = bytes,
        .blocks = cuda::fast_mod_div<long int>{blocks}
      };
      ensureOptIn<allReduceKernel<PurlinAtomLR, Element, CollectiveConfigLR>, kSLR>();
      allReduceKernel<PurlinAtomLR, Element, CollectiveConfigLR><<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>
      (kArgs, ctx);
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
        constexpr auto kSTR = cuda::std::max(PurlinAtomTR::COPY_SMEM_SIZE, PurlinAtomTR::RED_SMEM_SIZE);
        if (bytes <= CHUNK_SIZE) {
          constexpr auto putBlocks =  nonChunkedConfig::PUT_BLOCKS;
          const auto blocks = AR::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<PurlinAtomTR, Element, nonChunkedConfig, World2Bypass::yes>, kSTR>();
          allReduceKernel<PurlinAtomTR, Element, nonChunkedConfig, World2Bypass::yes><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
        else {
          constexpr auto putBlocks =  chunkedConfig::PUT_BLOCKS;
          const auto blocks = AR::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<PurlinAtomTR, Element, chunkedConfig, World2Bypass::yes>, kSTR>();
          allReduceKernel<PurlinAtomTR, Element, chunkedConfig, World2Bypass::yes><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
      }
        break;
      case 4: {
        constexpr auto threads = 256;
        constexpr auto pipeStages = 8;
        constexpr auto elementsPerThread = 1;
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
        constexpr auto kSTR = cuda::std::max(PurlinAtomTR::COPY_SMEM_SIZE, PurlinAtomTR::RED_SMEM_SIZE);
        if (bytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS + GATHER_BLOCKS;
          const auto blocks = AR::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<PurlinAtomTR, Element, nonChunkedConfig, World2Bypass::no>, kSTR>();
          allReduceKernel<PurlinAtomTR, Element, nonChunkedConfig, World2Bypass::no><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
        else {
          constexpr auto putBlocks =  chunkedConfig::PUT_BLOCKS + GATHER_BLOCKS;
          const auto blocks = AR::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<PurlinAtomTR, Element, chunkedConfig, World2Bypass::no>, kSTR>();
          allReduceKernel<PurlinAtomTR, Element, chunkedConfig, World2Bypass::no><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
      }
        break;
      default: {
        constexpr auto threads = 256;
        constexpr auto pipeStages = 8;
        constexpr auto elementsPerThread = 1;
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
        constexpr auto kSTR = cuda::std::max(PurlinAtomTR::COPY_SMEM_SIZE, PurlinAtomTR::RED_SMEM_SIZE);
        if (bytes <= CHUNK_SIZE) {
          constexpr auto putBlocks = nonChunkedConfig::PUT_BLOCKS + GATHER_BLOCKS;
          const auto blocks = AR::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<PurlinAtomTR, Element, nonChunkedConfig, World2Bypass::no>, kSTR>();
          allReduceKernel<PurlinAtomTR, Element, nonChunkedConfig, World2Bypass::no><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
        else {
          constexpr auto putBlocks =  chunkedConfig::PUT_BLOCKS + GATHER_BLOCKS;
          const auto blocks = AR::getBlocks<PurlinAtomTR>(bytes, putBlocks, maxReduceBlocks, world);
          const Args kArgs{
            .src = src,
            .dst = dst,
            .bytes = bytes,
            .blocks = cuda::fast_mod_div<long int>{blocks}
          };
          ensureOptIn<allReduceKernel<PurlinAtomTR, Element, chunkedConfig, World2Bypass::no>, kSTR>();
          allReduceKernel<PurlinAtomTR, Element, chunkedConfig, World2Bypass::no><<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>
          (kArgs, ctx);
        }
      }
    }
  }
}
#endif //PURLIN_ALLREDUCE_CUH
