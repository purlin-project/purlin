//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_ALLREDUCE_CUH
#define PURLIN_ALLREDUCE_CUH
#include <stdexcept>

#include "args.cuh"
#include "telemetry.cuh"
#include "tuning.cuh"
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

  template<typename PurlinAtom, typename Element, typename CollConfig, World2Bypass Bypass>
  __host__ __forceinline__
  void launchAllReduceThroughput(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const int& gatherBlocks, const int& maxReduceBlocks, cudaStream_t stream) {
    constexpr auto kS = cuda::std::max(PurlinAtom::COPY_SMEM_SIZE, PurlinAtom::RED_SMEM_SIZE);
    constexpr auto putBlocks = CollConfig::PUT_BLOCKS;
    const auto blocks = AR::getBlocks<PurlinAtom>(
      bytes, putBlocks + gatherBlocks, maxReduceBlocks, ctx.world);
    const Args kArgs{
      .src = src,
      .dst = dst,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    ensureOptIn<allReduceKernel<PurlinAtom, Element, CollConfig, Bypass>, kS>();
    allReduceKernel<PurlinAtom, Element, CollConfig, Bypass>
      <<<blocks, PurlinAtom::THREADS, kS, stream>>>(kArgs, ctx);
  }

  template<typename Element, int NArch, int World>
  __host__ __forceinline__
  void allReduceTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    constexpr auto tArch = host::tuningArch<NArch>;
    using Policy = host::AllReduceTuning<tArch, World>;

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
      ensureOptIn<allReduceKernel<PurlinAtomLR, Element, CollectiveConfigLR>, kS>();
      allReduceKernel<PurlinAtomLR, Element, CollectiveConfigLR>
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
      Policy::GATHER_BLOCKS,
      Policy::CHUNK_SIZE
    >;
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      Policy::GATHER_BLOCKS,
      Policy::CHUNK_SIZE
    >;
    constexpr auto bypass = World == 2 ? World2Bypass::yes : World2Bypass::no;
    constexpr auto gatherBlocks = Policy::GATHER_BLOCKS == UNUSED ? 0 : Policy::GATHER_BLOCKS;
    if (bytes <= Policy::CHUNK_SIZE) {
      launchAllReduceThroughput<PurlinAtomTR, Element, NonChunkedConfig, bypass>(
        src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
    else {
      launchAllReduceThroughput<PurlinAtomTR, Element, ChunkedConfig, bypass>(
        src, dst, bytes, ctx, gatherBlocks, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
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
#if defined(PURLIN_JIT_WORLD)
    static_assert(cuda::std::is_integral_v<decltype(PURLIN_JIT_WORLD)>);
    static_assert(PURLIN_JIT_WORLD == 2 || PURLIN_JIT_WORLD == 4 || PURLIN_JIT_WORLD == 8);
    allReduceTuned<Element, nArch, PURLIN_JIT_WORLD>(src, dst, bytes, ctx, stream);
#else
    const int world = ctx.world;
    switch (world) {
      case 2: allReduceTuned<Element, nArch, 2>(src, dst, bytes, ctx, stream); break;
      case 4: allReduceTuned<Element, nArch, 4>(src, dst, bytes, ctx, stream); break;
      default: allReduceTuned<Element, nArch, 8>(src, dst, bytes, ctx, stream); break;
    }
#endif
  }
}
#endif //PURLIN_ALLREDUCE_CUH
