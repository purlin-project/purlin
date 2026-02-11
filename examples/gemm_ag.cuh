//
// Created by osayamen on 2/9/26.
//

#ifndef TACK_GEMM_AG_CUH
#define TACK_GEMM_AG_CUH
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/ptx>
#include <cuda/utility>

#include "gemm_ag.cuh"
#include "tile.cuh"

namespace tack::examples
{
#if (__CUDA_ARCH__ >= 1000) && (defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 12) && (defined(__CUDACC_VER_MINOR__) && __CUDACC_VER_MINOR__ >= 9)
  constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
  constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif
  constexpr int threads = 128;
  constexpr int bM = 128;
  constexpr int bN = 128;
  constexpr int bK = 64;
  constexpr int pipeStagesGEMM = ARCH >= 800 ? 2 : 1;
  constexpr int WARP_SIZE = 32;
  constexpr int SMEM_ALIGNMENT = 16;
  constexpr int Alignment = 16;
  using Element = __half;
  using TileGEMM = tile::CollectiveMainloop<ARCH, bM, bN, bK, Element, float, threads, pipeStagesGEMM>;

  template <int Size>
  __device__ __forceinline__
  void cp_async_global_to_shared(void* __restrict__ const& smem_ptr, const void* __restrict__ const& gmem_ptr) {
    static_assert(Size == 4 || Size == 8 || Size == 16,
                  "cp.async only supports Size in {4, 8, 16}");
    uint32_t sp = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
      :
      : "r"(sp), "l"(gmem_ptr), "n"(Size)
    );
  }

  // GMEM -> GMEM
  template <int pipeStages = 2, int stageExtent = 2>
  __device__ __forceinline__
  void put(cuda::std::byte* __restrict__ const& dst, const cuda::std::byte* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& workspace, const size_t& partition /*in bytes*/) {
    // Simplifying assumptions
    // partition % MAX_ACCESS_ALIGNMENT == 0
    if (partition <= threads * Alignment * pipeStages * stageExtent) {
      constexpr int VectorWidth = MAX_ACCESS_ALIGNMENT / sizeof(uint);
      using VT = cutlass::AlignedArray<uint, VectorWidth, MAX_ACCESS_ALIGNMENT>;
      static_assert(cuda::std::is_trivially_copyable_v<VT>);
      const int vP = static_cast<int>(partition / MAX_ACCESS_ALIGNMENT);
      auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
      // use direct loads as pipelining is not necessary
      #pragma unroll 2
      for (int i = static_cast<int>(threadIdx.x); i < vP; i += threads) {
        const auto v = cuda::ptx::ld(cuda::ptx::space_global, vS + i);
        cuda::ptx::st(cuda::ptx::space_global, vD + i, v);
      }
    }
    else {
      constexpr int VectorWidth = Alignment / sizeof(uint);
      using VT = cutlass::AlignedArray<uint, VectorWidth, Alignment>;
      static_assert(cuda::is_power_of_two(pipeStages) && pipeStages >= 1 && pipeStages <= 8);
      auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
      auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
      const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
      const int stages = static_cast<int>(partition / (threads * Alignment * stageExtent));
      cuda::static_for<pipeStages>([&vW, &vS](auto i) {
        cuda::static_for<stageExtent>([&i, &vW, &vS](auto j) {
          const int slot = ((i * stageExtent + j) * threads) + threadIdx.x;
          // async gmem -> smem
          cp_async_global_to_shared<Alignment>(vW + slot, vS + slot);
        });
        cute::cp_async_fence();
      });
      VT reginald[stageExtent];
      for (int i = pipeStages; i < stages; ++i) {
        cute::cp_async_wait<pipeStages - 1>();
        const int stage_out = i - pipeStages;
        const int cs = stage_out % pipeStages;
        cuda::static_for<stageExtent>([&i, &cs, &vW, &reginald, &vS](auto j) {
          const int csW = (cs * stageExtent + j) * threads + threadIdx.x;
          const long int slot = (i * stageExtent + j) * threads + threadIdx.x;
          // smem -> rmem
          reginald[j] = vW[csW];
          // async gmem -> smem prefetch
          cp_async_global_to_shared<Alignment>(vW + csW, vS + slot);
        });
        cuda::static_for<stageExtent>([&stage_out, &reginald, &vD](auto j) {
          const long int slot = (stage_out * stageExtent + j) * threads + threadIdx.x;
          // rmem -> gmem
          vD[slot] = reginald[j];
        });
        // commit async transfers from this stage
        cute::cp_async_fence();
      }
      // tail
      cuda::static_for<pipeStages>([&vW, &reginald, &vS, &vD, &stages](auto i) {
        const int stage = (stages - pipeStages) + i;
        const int cs = stage % pipeStages;
        cute::cp_async_wait<pipeStages - 1 - i>();
        cuda::static_for<stageExtent>([&i, &cs, &vW, &reginald, &vS, &stages](auto j) {
          const int csW = (cs * stageExtent + j) * threads + threadIdx.x;
          // smem -> rmem
          reginald[j] = vW[csW];
        });
        cuda::static_for<stageExtent>([&stage, &reginald, &vD](auto j) {
          const long int slot = (stage * stageExtent + j) * threads + threadIdx.x;
          // rmem -> gmem
          vD[slot] = reginald[j];
        });
      });
    }
  }

  enum Status : int {
    pending,
    ready
  };

  struct __align__(16) Args {
    const cuda::std::byte* const A; // [M / world, K]
    cuda::std::byte* const B; // [N, K], symmetric
    cuda::std::byte* const C; // [M / world, N]
    int* const signals; // [world, ctas], symmetric
    uint64_t* const epochs; // [ctas, world], symmetric
    const uint64_t epoch; // monotonic counter
    int* const putSync; // [world]
    const int M;
    const int localM; // M / world
    const int N;
    const int K;
    const int world;
    const int rank;
    const int tilesM; // localM / bM
    const int tilesN; // N / bN
    const int numTiles; // tilesM * tilesN
    const int chunksPerPeer; // (N / world) / bN
  };

  __device__ __forceinline__
  void arrive_at_epoch(const Args& args) {
    if (threadIdx.x < args.world && threadIdx.x != args.rank) {
      // notify peers
      auto* ep = static_cast<uint64_t*>(nvshmem_ptr(args.epochs + (blockIdx.x * args.world + args.rank), threadIdx.x));
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> e{*ep};
      cuda::std::ignore = e.fetch_add(1, cuda::memory_order_release); // I think relaxed is fine here
      // wait for notification
      auto* myEP = args.epochs + (blockIdx.x * args.world + threadIdx.x);
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> me{*myEP};
      // == epoch or == epoch + 1
      auto isNotified = me.load(cuda::memory_order_acquire) >= args.epoch;
      while (!isNotified) {
        isNotified = me.load(cuda::memory_order_acquire) >= args.epoch;
      }
    }
    else if (threadIdx.x == args.world) {
      // set our own tile signal
      auto* sp = args.signals + (args.rank * gridDim.x + blockIdx.x);
      cuda::atomic_ref<int, cuda::thread_scope_system> s{*sp};
      s.store(ready, cuda::memory_order_release);
    }
    __syncthreads();
  }

  __device__ __forceinline__
  void tiledGEMM(const Args& args, cuda::std::byte* __restrict__ const& workspace) {
    const cooperative_groups::thread_block bg = cooperative_groups::this_thread_block();
    const auto* __restrict aP = reinterpret_cast<const Element*>(args.A);
    for (int tileIdx = static_cast<int>(blockIdx.x); tileIdx < args.numTiles; tileIdx += static_cast<int>(gridDim.x)) {
      constexpr TileGEMM tileMainloop{};
      const auto tileCoord = tile::idx2Coord(args.tilesM, args.tilesN, tileIdx);
      const auto tN = cute::get<1>(tileCoord);
      // compute owning peer
      const auto peer = tN / args.chunksPerPeer;
      // ensure data is ready
      cooperative_groups::invoke_one(bg, [&args]() {
        auto* sP = args.signals + (gridDim.x * peer + blockIdx.x);
        cuda::atomic_ref<int, cuda::thread_scope_system> p{*sP};
        auto isReady = p.load(cuda::memory_order_acquire) == ready;
        while (!isReady) {
          isReady = p.load(cuda::std::memory_order_acquire) == ready;
        }
        p.store(pending, cuda::memory_order_relaxed);
      });
      __syncthreads();
      // compute tile
      auto accumulator = TileGEMM::BLAS::suggest_accumulator();
      const auto* __restrict bP = reinterpret_cast<const Element*>(args.B);
      tileMainloop(workspace, aP, bP, accumulator, args.localM, args.N, args.K, tileCoord);
    }
  }

  // TODO standalone AllGather
  __device__ __forceinline__
  void gemmAG(const Args& args, cuda::std::byte* __restrict__ const& workspace) {
    // assumptions:
    // __isShared(workspace)
    // (N / world) % bN == 0
    // (# peers <= ctas)
    // threads > world
    // chunkSize % scaleFactor == 0
    static_assert(cuda::std::is_same_v<cuda::std::underlying_type_t<Status>, cuda::std::remove_pointer_t<decltype(args.signals)>>);
    __shared__ int isElected;
    arrive_at_epoch(args);
    // compute indices
    const int numSuperBlocks = static_cast<int>(cuda::ceil_div(gridDim.x, args.world));
    const int superBlockIdx = static_cast<int>(blockIdx.x % numSuperBlocks);
    const int superBlockSize = static_cast<int>((gridDim.x / args.world) + (superBlockIdx < gridDim.x % args.world));
    const int intraIdx = static_cast<int>(blockIdx.x) % superBlockIdx;
    const auto mappedPeer = (superBlockIdx + args.rank + 1) % args.world;

    constexpr auto scaleFactor = MAX_ACCESS_ALIGNMENT / sizeof(Element);
    // total number of aligned elements
    const size_t chunkSize = static_cast<size_t>(args.N / args.world) * args.K;
    const size_t scaledChunkSize = chunkSize / scaleFactor;
    const size_t chunksPerSBlock = (scaledChunkSize / numSuperBlocks) + (superBlockIdx < scaledChunkSize % numSuperBlocks);
    const size_t ctaBaseChunk = chunksPerSBlock / superBlockSize;
    const int residue = static_cast<int>(chunksPerSBlock % superBlockSize);
    const size_t ctaChunk = ctaBaseChunk + (intraIdx < residue);
    // compute buffer offset
    const auto startOffset = ctaBaseChunk * intraIdx + min(intraIdx, residue);
    const auto* __restrict__ srcP = args.B + (chunkSize * args.rank * sizeof(Element)) + startOffset;
    auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(srcP, mappedPeer));
    // put data
    put(dstP, srcP, workspace, ctaChunk);
    __syncthreads();
    if (threadIdx.x == 0) {
      cuda::atomic_ref<int, cuda::thread_scope_device> ps{*(args.putSync + mappedPeer)};
      if (ps.fetch_add(1, cuda::memory_order_acq_rel) + 1 == superBlockSize) {
        ps.store(0, cuda::memory_order_relaxed); // cleanup for the subsequent epoch
        isElected = 1;
      }
      else {
        isElected = 0;
      }
    }
    __syncthreads();
    if (isElected) {
      auto* sP = static_cast<int*>(nvshmem_ptr(args.signals + args.rank * gridDim.x, mappedPeer));
      for (int i = static_cast<int>(threadIdx.x); i < gridDim.x; i += threads) {
        cuda::atomic_ref<int, cuda::thread_scope_system> p{*(sP + i)};
        p.store(ready, cuda::memory_order_release);
      }
    }
    // do GEMM
    tiledGEMM(args, workspace);
  }
  __launch_bounds__(threads, 1)
  __global__ void gemmAGKernel(const __grid_constant__ Args args) {
    extern __shared__ __align__(SMEM_ALIGNMENT) cuda::std::byte workspace[];
    if (args.world == 1) {
      tiledGEMM(args, workspace);
    }
    else {
      gemmAG(args, workspace);
    }
  }
}
#endif //TACK_GEMM_AG_CUH
