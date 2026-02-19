//
// Created by osayamen on 2/18/26.
//
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <stdexcept>

#include <cuda/cmath>
#include <cuda/utility>
#include <cuda/ptx>
#include <cutlass/array.h>
#include <cute/arch/copy_sm80.hpp>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>
#include <nvshmem.h>

#include "common.cuh"
#include "debug.cuh"

constexpr int WARP_SIZE = 32;
constexpr int threads = 128;
static_assert(threads >= 32 && threads % WARP_SIZE == 0);
constexpr int Alignment = 16;
constexpr int pipeStages = 2;
constexpr int stageExtent = 4;
#if (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 9)
constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif
struct __align__(16) AGArgs {
  cuda::std::byte* sendBuff = nullptr; // [size], symmetric
  uint64_t* const completions = nullptr; // [ctas], symmetric
  uint64_t* const arrivals = nullptr; // [ctas, world], symmetric
  uint64_t signal = 0; // single bit
  size_t size = 0; // per rank message size in bytes
  const int rank = 0;
  const int world = 1;
};

namespace tack
{
  __device__ __forceinline__
  void arrive(const AGArgs& args, const int& peer) {
    for (int i = static_cast<int>(threadIdx.x); i < args.world; i += threads) {
      auto* ma = static_cast<uint64_t*>(nvshmem_ptr(args.arrivals + (blockIdx.x * args.world + args.rank), i));
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ap{*ma};
      ap.store(args.signal, cuda::memory_order_relaxed);
    }
    cooperative_groups::invoke_one(cooperative_groups::this_thread_block(), [&args, &peer]() {
      auto* na = args.arrivals + (blockIdx.x * args.world + peer);

      // wait for notification
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> np{*na};
      auto isNotified = np.load(cuda::memory_order_relaxed) == args.signal;
      while (!isNotified) {
        isNotified = np.load(cuda::memory_order_relaxed) == args.signal;
      }
    });
    __syncthreads();
  }
  template <int Size>
  __device__ __forceinline__
  void cp_async_global_to_shared(void* __restrict__ const& smem_ptr, const void* __restrict__ const& gmem_ptr) {
    static_assert(Size == 4 || Size == 8 || Size == 16, "cp.async only supports Size in {4, 8, 16}");
    uint32_t sp = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
      :
      : "r"(sp), "l"(gmem_ptr), "n"(Size)
    );
  }

  // GMEM -> GMEM
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
      for (int i = static_cast<int>(threadIdx.x); i < vP; i += threads) {
        if constexpr (MAX_ACCESS_ALIGNMENT > 16) {
          const auto v = cuda::ptx::ld(cuda::ptx::space_global, vS + i);
          cuda::ptx::st(cuda::ptx::space_global, vD + i, v);
        }
        else {
          vD[i] = vS[i];
        }
      }
    }
    else {
      constexpr int VectorWidth = Alignment / sizeof(uint);
      using VT = cutlass::AlignedArray<uint, VectorWidth, Alignment>;
      static_assert(pipeStages >= 1);
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
      // residue
      const auto cutoff = stages * static_cast<size_t>(threads * Alignment * stageExtent);
      const auto cutoffElems = cutoff / Alignment;
      const auto residue = (partition - cutoff) / Alignment; // elements not bytes
      const auto* __restrict rvS = vS + cutoffElems;
      vD += cutoffElems;
      for (size_t i = threadIdx.x; i < residue; i += threads) {
        if constexpr (MAX_ACCESS_ALIGNMENT > 16) {
          const auto v = cuda::ptx::ld(cuda::ptx::space_global, rvS + i);
          cuda::ptx::st(cuda::ptx::space_global, vD + i, v);
        }
        else {
          vD[i] = rvS[i];
        }
        //const auto v = cuda::ptx::ld(cuda::ptx::space_global, rvS + i);
        //cuda::ptx::st(cuda::ptx::space_global, vD + i, v);
      }
    }
  }
  __device__ __forceinline__
  void wait(const AGArgs& args, const int& peer) {
    __syncthreads();
    cooperative_groups::invoke_one(cooperative_groups::this_thread_block(), [&args, peer] {
      auto* mySP = args.completions + blockIdx.x;
      auto* sP = static_cast<uint64_t*>(nvshmem_ptr(mySP, peer));
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> p{*sP};
      // notify peer
      p.store(args.signal, cuda::memory_order_release);
      // await
      auto received = p.load(cuda::memory_order_acquire) == args.signal;
      while (!received) {
        received = p.load(cuda::memory_order_acquire) == args.signal;
      }
    });
    __syncthreads();
  }
}

__launch_bounds__(threads, 1)
__global__ void ag(const __grid_constant__ AGArgs args) {
  __shared__ __align__(Alignment) cuda::std::byte workspace[threads * Alignment * pipeStages * stageExtent];
  // compute indices
  // # ctas >= actualWorld
  // size % MAX_ACCESS_ALIGNMENT == 0
  const auto actualWorld = args.world - 1;
  const int numSuperBlocks = actualWorld;
  const int superBlockIdx = static_cast<int>(blockIdx.x % numSuperBlocks);
  const int intraIdx = static_cast<int>(blockIdx.x) / numSuperBlocks;
  const int superBlockSize = static_cast<int>((gridDim.x / actualWorld) + (superBlockIdx < gridDim.x % actualWorld));
  const auto peer = (superBlockIdx + args.rank + 1) % args.world;

  // total number of aligned elements
  const size_t scaledChunkSize = args.size / MAX_ACCESS_ALIGNMENT;
  const size_t ctaBaseChunk = scaledChunkSize / superBlockSize;
  const int residue = static_cast<int>(scaledChunkSize % superBlockSize);
  const size_t ctaChunk = ctaBaseChunk + (intraIdx < residue);
  // compute buffer offset
  const auto startOffset = (ctaBaseChunk * intraIdx + min(intraIdx, residue)) * MAX_ACCESS_ALIGNMENT;
  const auto* __restrict__ srcP = args.sendBuff + startOffset;
  auto* __restrict__ dstP = static_cast<cuda::std::byte*>(nvshmem_ptr(args.sendBuff + startOffset, peer));
  const size_t bytes = ctaChunk * MAX_ACCESS_ALIGNMENT;

  tack::arrive(args, peer);
  tack::put(dstP, srcP, workspace, bytes);
  //nvshmemx_putmem_nbi_block(args.sendBuff + startOffset, srcP, bytes, peer);
  tack::wait(args, peer);
}

struct Options {
  size_t minBytes = 128;
  size_t maxBytes = 128 * 1024 * 1024;
  int warmup = 128;
  int runs = 256;
};

float median(std::vector<float> v) {
  if (v.empty()) throw std::invalid_argument("median: empty vector");

  const size_t n = v.size();
  const long mid = static_cast<long>(n / 2);

  // Put the element that would be at position mid in sorted order into v[mid]
  std::ranges::nth_element(v.begin(), v.begin() + mid, v.end());
  float m = v[mid];

  if (n % 2 == 0) {
    // For even n, need the lower middle too
    std::nth_element(v.begin(), v.begin() + (mid - 1), v.begin() + mid);
    m = 0.5f * (m + v[mid - 1]);
  }
  return m;
}

struct Times {
  double t_ms;
  double n_ms;
  double ep;
};
// Parse sizes like 4096, 4K, 16M, 1G
size_t parseSize(const std::string& s) {
  char unit = 0;
  double val = 0.0;
  if (sscanf(s.c_str(), "%lf%c", &val, &unit) >= 1) {
    size_t mult = 1;
    switch (unit) {
    case 'k': case 'K': mult = 1024ull; break;
    case 'm': case 'M': mult = 1024ull * 1024ull; break;
    case 'g': case 'G': mult = 1024ull * 1024ull * 1024ull; break;
    default: mult = 1; break;
    }
    if (unit == 0 || (unit != 'K' && unit != 'k' && unit != 'M' && unit != 'm' && unit != 'G' && unit != 'g')) {
      // no unit, already parsed in val
      return static_cast<size_t>(val);
    }
    return static_cast<size_t>(val * static_cast<double>(mult));
  }
  fprintf(stderr, "Invalid Size\n");
  std::exit(EXIT_FAILURE);
}

__host__
void agHost(const Options& opts) {
  cuda::std::byte* rcvBuff = nullptr; // [world, size], symmetric
  uint64_t* completions = nullptr; // [ctas], symmetric
  uint64_t* arrivals = nullptr; // [ctas, world], symmetric

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("world,localBytes,globalBytes,threads,blocks/SM,SMs,blocks,error(%%),warmup,runs,tack(ms),nccl(ms),tack(GB/s),nccl(GB/s)\n");
    if (world <= 1) {
      printf("pass\n");
      return;
    }
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  auto kernel = ag;
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, 0));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  const auto actualWorld = world - 1;
  const auto blocksUpper = min(cuda::ceil_div(opts.maxBytes, threads * Alignment) * actualWorld, static_cast<size_t>(num_sms * bps));
  completions = static_cast<uint64_t*>(nvshmem_calloc(blocksUpper, sizeof(uint64_t)));
  arrivals = static_cast<uint64_t*>(nvshmem_calloc(blocksUpper * world, sizeof(uint64_t)));
  rcvBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxBytes * world));
  auto* refBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxBytes * world));
  if (rcvBuff == nullptr || !cuda::is_aligned(rcvBuff, MAX_ACCESS_ALIGNMENT)) {
    throw std::runtime_error("rcvBuff is invalid");
  }
  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.minCTAs = blocksUpper;
  config.maxCTAs = blocksUpper;
  NCCL_CHECK(ncclCommInitRankConfig(&comm, world, id, rank, &config));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::random_device rd;
  AGArgs args{
    .sendBuff = nullptr,
    .completions = completions,
    .arrivals = arrivals,
    .signal = 1,
    .size = opts.minBytes,
    .rank = rank,
    .world = world
  };
  auto agk = [&](const auto& blocks, AGArgs& kArgs, const int& runs) {
    for (int i = 0; i < runs; ++i) {
      ag<<<blocks, threads, 0, stream>>>(kArgs);
      kArgs.signal += 1;
    }
  };
  auto nag = [&](auto* const& sb, auto* const& rb, const auto& count, const int& runs) {
    for (int i = 0; i < runs; ++i) {
      ncclAllGather(sb, rb, count, ncclUint8, comm, stream);
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  for (size_t bytes = opts.minBytes; bytes <= opts.maxBytes; bytes *= 2) {
    // fill buffer with random values
    const auto seed = rd();
    static_assert(MAX_ACCESS_ALIGNMENT % sizeof(float) == 0);
    const auto elems = bytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(rcvBuff) + (rank * elems);
    randUniform<ARCH>(tS, elems, seed, -1.f, 1.f, stream);
    auto* tSr = reinterpret_cast<float*>(refBuff) + (rank * elems);
    randUniform<ARCH>(tSr, elems, seed, -1.f, 1.f, stream);
    args.size = bytes;
    args.sendBuff = rcvBuff + (rank * bytes);
    const auto blocks = static_cast<uint>(min(cuda::ceil_div(bytes, threads * Alignment) * actualWorld,
      static_cast<size_t>(num_sms * bps)));
    // correctness run
    agk(blocks, args, 1);
    auto* sB = refBuff + (rank * bytes);
    nag(sB, refBuff, bytes, 1);
    auto ag_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(rcvBuff), {1, static_cast<matx::index_t>(elems * world)});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff), {1, static_cast<matx::index_t>(elems * world)});
    // bitwise check
    (ag_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);
    // benchmark tack
    agk(blocks, args, opts.warmup);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaEventRecord(start, stream);
    agk(blocks, args, opts.runs);
    cudaEventRecord(stop, stream);
    CHECK_CUDA(cudaEventSynchronize(stop));
    float t_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
    t_ms /= static_cast<float>(opts.runs);

    // benchmark NCCL AG
    nag(sB, refBuff, bytes, opts.warmup);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaEventRecord(start, stream);
    nag(sB, refBuff, bytes, opts.runs);
    cudaEventRecord(stop, stream);
    CHECK_CUDA(cudaEventSynchronize(stop));
    float n_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&n_ms, start, stop));
    n_ms /= static_cast<float>(opts.runs);

    times.ep = 1.0 - (static_cast<double>(ag_matches()) / static_cast<double>(tR.TotalSize()));
    times.t_ms = t_ms;
    times.n_ms = n_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = (world * static_cast<double>(bytes)) / 1e9;
      const auto tack_algBW = gb / (times.t_ms * 1e-3);
      const auto nccl_algBW = gb / (times.n_ms * 1e-3);
      printf("%d, %lu, %lu, %d, %d, %d, %d, %lf, %d, %d, %lf, %lf, %lf, %lf\n",
        world, bytes, world * bytes, threads, bps, num_sms, blocks, times.ep, opts.warmup, opts.runs, times.t_ms, times.n_ms, tack_algBW, nccl_algBW);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_free(arrivals);
  nvshmem_free(completions);
  nvshmem_free(rcvBuff);
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}
// ./ag <minBytes> <maxBytes> <warmup> <runs>
int main(const int argc, char** argv) {
  Options opts{};
  if (argc > 1) opts.minBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxBytes = parseSize(argv[2]);
  if (argc > 3) opts.warmup = std::stoi(argv[3]);
  if (argc > 4) opts.runs = std::stoi(argv[4]);
  if (!cuda::is_power_of_two(opts.minBytes) || !cuda::is_power_of_two(opts.maxBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minBytes % MAX_ACCESS_ALIGNMENT != 0 || opts.maxBytes % MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  agHost(opts);
}