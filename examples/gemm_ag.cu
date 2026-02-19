//
// Created by Osayamen on 1/8/26.
//

#include <cuda_fp16.h>
#include <cstdio>
#include <random>
#include <vector>

#include <mpi.h>
#include <matx.h>
#include <nvshmem.h>
#include <nccl.h>
#include "common.cuh"
#include "debug.cuh"
#include "gemm_ag.cuh"

// Our implementations:
// 1. Standalone AG + GEMM
// 2. Fused GEMM + AG (tack)
// Baselines:
// 1. NCCL AG + cuBLASLt GEMM
// 4. Fused GEMM + AG (NVSHMEM)
using MT = matx::matxFp16;
template <typename Element>
consteval auto get_nccl_type() {
  static_assert(
    cuda::std::is_same_v<Element, __half> ||
    cuda::std::is_same_v<Element, __nv_bfloat16> ||
    cuda::std::is_same_v<Element, float> ||
    cuda::std::is_same_v<Element, double>,
    "Unsupported Element type"
  );
  if constexpr (cuda::std::is_same_v<Element, double>) return ncclDouble;
  else if constexpr (cuda::std::is_same_v<Element, float>) return ncclFloat;
  else if constexpr (cuda::std::is_same_v<Element, __half>) return ncclHalf;
  else return ncclBfloat16;
}

struct Options {
  int localM = 128;
  int localN = 256;
  int K = 128;
  int warmup = 128;
  int runs = 128;
  float rtol = 2e-2;
  float atol = 2e-3;
};

__host__ __forceinline__
uint parse_u32(const char* s, const char* name, const uint lo = 1, const uint hi = std::numeric_limits<uint>::max()) {
  // Fast, non-allocating, rejects negatives automatically
  std::string_view sv{s};
  uint64_t v = 0;
  auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
  if (ec != std::errc{} || ptr != sv.data() + sv.size()) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  if (v < lo || v > hi) {
    throw std::invalid_argument(std::string(name) + " out of range");
  }
  return static_cast<uint>(v);
}

__host__ __forceinline__
float parse_f32(const char* s, const char* name) {
  try {
    size_t idx = 0;
    float v = std::stof(s, &idx);
    if (idx != std::string(s).size()) {
      throw std::invalid_argument("trailing chars");
    }
    return v;
  } catch (...) {
    throw std::invalid_argument(std::string(name) + " must be a float");
  }
}

__host__ __forceinline__
void kickStart(const Options& opts) {
  cuda::std::byte* a = nullptr;
  cuda::std::byte* b = nullptr;
  cuda::std::byte* bRef = nullptr;
  cuda::std::byte* c = nullptr;
  int* putSync = nullptr;
  int* superSync = nullptr;
  int* signals = nullptr;
  uint64_t* epochs = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("Rank, world, dtype, localM, localN, K, bM, bN, bK, threads, blocks/SM, SMs, blocks, rtol, atol, "
           "ag_error(%%), gemm_ag_error(%%), warmup, runs, tack_Time(ms), nccl_cublas_time(ms)\n");
  }

  const auto N = opts.localN * world;
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  std::random_device rd;
  constexpr auto min_v = -1.f;
  constexpr auto max_v = 1.f;
  static_assert(std::is_same_v<uint32_t, decltype(rd)::result_type>);
  CHECK_CUDA(cudaMallocAsync(&a, sizeof(Element) * opts.localM * static_cast<size_t>(opts.K), stream));
  randUniform<ARCH>(reinterpret_cast<Element*>(a), opts.localM * static_cast<size_t>(opts.K), rd(), min_v, max_v, stream);
  CHECK_CUDA(cudaMallocAsync(&c, sizeof(Element) * opts.localM * static_cast<size_t>(N), stream));
  CHECK_CUDA(cudaMallocAsync(&putSync, sizeof(int) * world, stream));
  CHECK_CUDA(cudaMemsetAsync(putSync, 0, sizeof(int) * world, stream));
  cuda::std::byte* cRef = nullptr;
  CHECK_CUDA(cudaMallocAsync(&cRef, sizeof(Element) * opts.localM * static_cast<size_t>(N), stream));
  // get number of CTAs.
  constexpr auto pipeStages = 4;
  constexpr auto stageExtent = 4;
  constexpr auto putSharedSize = Alignment * threads * pipeStages * stageExtent;
  constexpr auto gemmSharedSize = cutlass::round_up(sizeof(Element) *
    bK * pipeStagesGEMM * (bM + bN), Alignment);
  constexpr auto kernelSharedSize = cute::max(gemmSharedSize, putSharedSize);
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  if (kernelSharedSize > maxSharedMemory) {
    const auto errmsg = std::string("Required shared memory ").append(std::to_string(kernelSharedSize))
    .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory)).append(" Reduce tile shapes or input sizes.");
    throw std::runtime_error(errmsg);
  }
  auto kernel = tack::gemmAGKernel;
  // opt-in
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSharedSize));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, kernelSharedSize));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  const size_t chunkSize = opts.localN * static_cast<size_t>(opts.K) * sizeof(Element);
  if (opts.localM % bM != 0 || opts.localN % bN != 0) {
    throw std::invalid_argument("localM or localN is invalid");
  }
  if (opts.K % bK != 0 || opts.K < pipeStagesGEMM * bK) {
    throw std::invalid_argument("K is invalid");
  }
  if (chunkSize % MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("localN * K should be a multiple of " + std::to_string(MAX_ACCESS_ALIGNMENT));
  }
  const auto commBlocks = cute::ceil_div(opts.localN * opts.K, (threads * Alignment)) * world;
  // get max number of blocks
  const auto blocks = cute::min(bps * num_sms, cute::max((opts.localM / bM) * (opts.localN / bN), commBlocks));
  if (world > blocks) {
    // this is not a functional requirement, just a simplifying assumption
    throw std::invalid_argument("World should be <= " + std::to_string(blocks));
  }
  b = static_cast<cuda::std::byte*>(nvshmem_malloc(sizeof(Element) * N * static_cast<size_t>(opts.K)));
  bRef = static_cast<cuda::std::byte*>(nvshmem_malloc(sizeof(Element) * N * static_cast<size_t>(opts.K)));
  CHECK_CUDA(cudaMallocAsync(&superSync, sizeof(int) * blocks, stream));
  CHECK_CUDA(cudaMemsetAsync(superSync, 0, sizeof(int) * blocks, stream));
  // fill our input buffer with random values
  const auto bSeed = rd();
  auto* localB = reinterpret_cast<Element*>(b + chunkSize * rank);
  randUniform<ARCH>(localB, opts.localN * static_cast<size_t>(opts.K), bSeed, min_v, max_v, stream);
  auto* localBRef = reinterpret_cast<Element*>(bRef + chunkSize * rank);
  randUniform<ARCH>(localBRef, opts.localN * static_cast<size_t>(opts.K), bSeed, min_v, max_v, stream);
  epochs = static_cast<uint64_t*>(nvshmem_calloc(world, sizeof(uint64_t)));
  static_assert(tack::pending == 0);
  signals = static_cast<int*>(nvshmem_calloc(world * blocks, sizeof(int)));
  auto args = tack::GAGArgs{
    .A = a,
    .B = b,
    .C = c,
    .signals =  signals,
    .epochs = epochs,
    .superSync = superSync,
    .putSync = putSync,
    .epoch = 1,
    .chunkSize = chunkSize,
    .localM = opts.localM,
    .N = N,
    .K = opts.K,
    .world = world,
    .rank = rank,
    .tilesM = opts.localM / bM,
    .tilesN = N / bN,
    .numTiles = (opts.localM / bM) * (N / bN),
    .chunksPerPeer = opts.localN / bN
  };

  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }

  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  auto gag = [&](const int& runs) {
    nvtx3::scoped_range r{"TACK"};
    for (int i = 0; i < runs; ++i) {
      tack::gemmAGKernel<<<blocks, threads, kernelSharedSize, stream>>>(args);
      args.epoch += 1;
    }
  };
  gag(1);
  matx::cudaExecutor exec{stream};

  // call and bench reference (NCCL AG + cuBLASLt)
  constexpr auto ndt = get_nccl_type<Element>();
  const auto* sendBuff = args.B + args.rank * args.chunkSize;
  const auto tA = matx::make_tensor<MT>(reinterpret_cast<MT*>(args.A), {args.localM, args.K});
  const auto tBRef = matx::make_tensor<MT>(reinterpret_cast<MT*>(bRef), {args.N, args.K});
  auto tCRef = matx::make_tensor<MT>(reinterpret_cast<MT*>(cRef), {args.localM, args.N});
  auto tB = matx::make_tensor<MT>(reinterpret_cast<MT*>(args.B), {args.N, args.K});
  auto refK = [&](const int& runs) {
    nvtx3::scoped_range r{"NCCL+cuBLAS"};
    if (world > 1) {
      for (int i = 0; i < runs; ++i) {
        // do NCCL AG
        ncclAllGather(sendBuff, bRef, args.chunkSize, ndt, comm, exec.getStream());
        // do cuBLASLt GEMM via MatX
        //(tCRef = matx::matmul(tA, tBRef.PermuteMatrix())).run(exec);
      }
    }
    else {
      for (int i = 0; i < runs; ++i) {
        (tCRef = matx::matmul(tA, tBRef.PermuteMatrix())).run(exec);
      }
    }
  };
  refK(1);
  // check correctness of ag
  auto ag_matches = matx::make_tensor<long int>({});
  // bitwise check
  (ag_matches = matx::sum(matx::isclose(tB, tBRef, 0, 0))).run(exec);
  // check correctness of gemm+ag
  auto tC = matx::make_tensor<MT>(reinterpret_cast<MT*>(c), {opts.localM, N});
  auto num_matches = matx::make_tensor<long int>({});
  (num_matches = matx::sum(matx::isclose(tC, tCRef, opts.rtol, opts.atol))).run(exec);
  exec.sync();
  // calculate error percentages
  const auto ag_ep = (1.0 - (static_cast<double>(ag_matches()) / static_cast<double>(tB.TotalSize()))) * 100;
  const auto ep =  (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(tC.TotalSize()))) * 100;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  // bench refK
  refK(opts.warmup);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaEventRecord(start);
  refK(opts.runs);
  cudaEventRecord(stop, stream);
  CHECK_CUDA(cudaEventSynchronize(stop));
  float nccl_cublas_time = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&nccl_cublas_time, start, stop));
  nccl_cublas_time /= static_cast<float>(opts.runs);
  // bench tack
  gag(opts.warmup);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaEventRecord(start);
  gag(opts.runs);
  cudaEventRecord(stop, stream);
  CHECK_CUDA(cudaEventSynchronize(stop));
  float m_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&m_ms, start, stop));
  m_ms /= static_cast<float>(opts.runs);
  constexpr auto es = element_string<Element>();
  printf("%d, %d, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %lf, %d, %d, %f, %f\n",
           rank, world, es, opts.localM, opts.localN, opts.K, bM, bN, bK, threads, bps, num_sms, blocks, opts.rtol,
           opts.atol, ag_ep, ep, opts.warmup, opts.runs, m_ms, nccl_cublas_time);
  CHECK_CUDA(cudaEventDestroy(start));CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFreeAsync(a, stream));
  CHECK_CUDA(cudaFreeAsync(c, stream));
  CHECK_CUDA(cudaFreeAsync(cRef, stream));
  CHECK_CUDA(cudaFreeAsync(putSync, stream));
  CHECK_CUDA(cudaFreeAsync(superSync, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
  nvshmem_free(b);
  nvshmem_free(bRef);
  nvshmem_free(signals);
  nvshmem_free(epochs);
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}

// ./gemm_ag <localM> <localN> <K> <warmup> <runs> <rtol> <atol>
int main(const int argc, char** argv) {
  Options opts{};
  if (argc > 1) opts.localM = std::stoi(argv[1]);
  if (argc > 2) opts.localN = std::stoi(argv[2]);
  if (argc > 3) opts.K = std::stoi(argv[3]);
  if (argc > 4) opts.warmup = std::stoi(argv[4]);
  if (argc > 5) opts.runs = std::stoi(argv[5]);
  if (argc > 6) opts.rtol = std::stof(argv[6]);
  if (argc > 7) opts.atol = std::stof(argv[7]);
  kickStart(opts);
}