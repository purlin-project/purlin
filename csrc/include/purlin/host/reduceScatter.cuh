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
  template<DataLayout InputLayout, typename PurlinAtom, typename Element, typename CollConfig>
  __launch_bounds__(PurlinAtom::THREADS, 1)
  __global__ void reduceScatterKernel(const __grid_constant__ Args kArgs, const __grid_constant__ Context ctx,
    const size_t* __restrict__ sizes) {
    static_assert(InputLayout == DataLayout::scattered || InputLayout == DataLayout::scatteredV);
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    auto* __restrict__ typedWorkspace = reinterpret_cast<Element*>(workspace);
    if constexpr (InputLayout == DataLayout::scatteredV) {
      purlin::reduceScatterV<PurlinAtom, CollConfig>
        (kArgs.dst, kArgs.src, sizes, typedWorkspace, ctx, kArgs.blocks);
    }
    else {
      purlin::reduceScatter<PurlinAtom, CollConfig>
        (kArgs.dst, kArgs.src, kArgs.bytes, typedWorkspace, ctx, kArgs.blocks);
    }
  }

  template<DataLayout InputLayout, typename PurlinAtom, typename Element, typename CollConfig, size_t SmemSize>
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
    ensureOptIn<reduceScatterKernel<InputLayout, PurlinAtom, Element, CollConfig>, SmemSize>();
    reduceScatterKernel<InputLayout, PurlinAtom, Element, CollConfig>
      <<<blocks, PurlinAtom::THREADS, SmemSize, stream>>>(kArgs, ctx, sizes);
  }

  template<DataLayout InputLayout, typename PurlinAtom, typename Element, typename CollConfig>
  __host__ __forceinline__
  void rst(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const Context& ctx,
    const size_t* __restrict__ sizes, const int& maxReduceBlocks, cudaStream_t stream) {
    constexpr auto kS = cuda::std::max(PurlinAtom::COPY_SMEM_SIZE, PurlinAtom::RED_SMEM_SIZE);
    constexpr auto putBlocks = CollConfig::PUT_BLOCKS;
    const auto blocks = RS::getBlocks<PurlinAtom>(bytes, putBlocks, maxReduceBlocks, ctx.world);
    launchReduceScatterKernel<InputLayout, PurlinAtom, Element, CollConfig, kS>
      (src, dst, bytes, ctx, sizes, blocks, stream);
  }

  template<DataLayout InputLayout, typename Element, int NArch, int World>
  __host__ __forceinline__
  void reduceScatterTuned(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ sizes, const Context& ctx, cudaStream_t stream) {
    constexpr auto alignment = 16;
    constexpr auto unrollFactor = 2;
    using Policy = host::ReduceScatterTuning<NArch, World>;

    if (dispatchBytes <= Policy::LATENCY_THRESHOLD) {
      using LRConfig = Configuration<
        Regime::latency,
        Policy::LR_THREADS,
        alignment,
        UNUSED,
        UNUSED,
        unrollFactor,
        host::getWorldUnroll<World>()
      >;
      using PurlinAtomLR = Atom<NArch, LRConfig>;
      const auto blocks = getLRBlocks<PurlinAtomLR::THREADS>(dispatchBytes);
      constexpr auto kS = PurlinAtomLR::RED_SMEM_SIZE;
      launchReduceScatterKernel<InputLayout, PurlinAtomLR, Element, CollectiveConfigLR, kS>
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
      Policy::CHUNK_SIZE
    >;
    using ChunkedConfig = CollectiveConfig<
      CollectiveType::chunked,
      Policy::CHUNKED_PUT_BLOCKS,
      UNUSED,
      Policy::CHUNK_SIZE
    >;
    if (dispatchBytes <= Policy::CHUNK_SIZE) {
      rst<InputLayout, PurlinAtomTR, Element, NonChunkedConfig>
        (src, dst, bytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
    else {
      rst<InputLayout, PurlinAtomTR, Element, ChunkedConfig>
        (src, dst, bytes, ctx, sizes, Policy::MAX_CONSUMER_BLOCKS, stream);
    }
  }

  template<DataLayout InputLayout, typename Element, int NArch>
  __host__ __forceinline__
  void dispatchReduceScatter(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst, const size_t& bytes, const size_t& dispatchBytes,
    const size_t* __restrict__ sizes, const Context& ctx, cudaStream_t stream) {
    switch (ctx.world) {
      case 2:
        reduceScatterTuned<InputLayout, Element, NArch, 2>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      case 4:
        reduceScatterTuned<InputLayout, Element, NArch, 4>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      case 8:
        reduceScatterTuned<InputLayout, Element, NArch, 8>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
      default:
        reduceScatterTuned<InputLayout, Element, NArch, host::FALLBACK>
          (src, dst, bytes, dispatchBytes, sizes, ctx, stream);
        break;
    }
  }

  template<int arch, typename Element>
  __host__ __forceinline__
  void reduceScatter(const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& dst,
    const size_t& bytes, const Context& ctx, cudaStream_t stream) {
#if defined(PURLIN_NVTX) && PURLIN_NVTX
    const PurlinRange range{"purlin::reduceScatter", nvtx3::payload{static_cast<uint64_t>(bytes)}};
#endif
    if (ctx.world * bytes > ctx.stagingTRSize) {
      throw std::runtime_error("Bytes exceeds limit");
    }
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchReduceScatter<DataLayout::scattered, Element, nArch>
      (src, dst, bytes, bytes, nullptr, ctx, stream);
  }

  template<int arch, typename Element>
  __host__ __forceinline__
  void reduceScatterV(const cuda::std::byte* __restrict__ const& src,
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
    constexpr auto nArch = purlin::normalizeArch<arch>();
    dispatchReduceScatter<DataLayout::scatteredV, Element, nArch>
      (src, dst, bytes, maxBytes, sizes, ctx, stream);
  }
}
#endif //PURLIN_REDUCESCATTER_CUH
