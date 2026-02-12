//
// Created by Osayamen on 1/8/26.
//

#include <cstdio>
#include <random>

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

// NCCL GEMM + AG
__host__ __forceinline__
float reference(const tack::GAGArgs& args, cuda::std::byte* const& cRef) {
  // do NCCL AG
  // do cuBLASLt GEMM via MatX
  return 0.0f;
}

struct Options {
  const int localM = 128;
  const int localN = 128;
  const int K = 128;
  const int warmup = 128;
  const int runs = 128;
  const float rtol = 2e-2;
  const float atol = 2e-3;
};

__host__ __forceinline__
void kickStart(const Options& opts) {
  cuda::std::byte* a = nullptr;
  cuda::std::byte* b = nullptr;
  cuda::std::byte* c = nullptr;
  int* putSync = nullptr;
  int* signals = nullptr;
  uint64_t* epochs = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("Rank, world, dtype, localM, localN, K, bM, bN, bK, threads, blocks/SM, SMs, blocks, rtol, atol, "
           "error(%%), warmup, runs, tack_Time(ms), nccl_cublas_time(ms)\n");
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
  randUniform<ARCH>(a, opts.localM * static_cast<size_t>(opts.K), rd(), min_v, max_v, stream);
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
  const size_t chunkSize = opts.localN * static_cast<size_t>(opts.K);
  if (opts.localM % bM != 0 || opts.localN % bN != 0) {
    throw std::invalid_argument("localM or localN is invalid");
  }
  if (chunkSize % MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("localN * K should be a multiple of " + std::to_string(MAX_ACCESS_ALIGNMENT));
  }
  const auto commBlocks = cute::ceil_div(opts.localN * opts.K, (threads * Alignment)) * world;
  const auto blocks = cute::min(bps * num_sms, cute::max((opts.localM / bM) * (opts.localN / bN), commBlocks));
  if (world > blocks) {
    // this is not a functional requirement, just a simplifying assumption
    throw std::invalid_argument("World should be <= " + std::to_string(blocks));
  }
  // get max number of blocks
  b = static_cast<cuda::std::byte*>(nvshmem_malloc(sizeof(Element) * N * static_cast<size_t>(opts.K)));
  // fill our input buffer with random values
  randUniform<ARCH>(b + chunkSize * rank, N * static_cast<size_t>(opts.K), rd(), min_v, max_v, stream);
  nvshmem_barrier_all();
  epochs = static_cast<uint64_t*>(nvshmem_calloc(world * blocks, sizeof(uint64_t)));
  signals = static_cast<int*>(nvshmem_calloc(world * blocks, sizeof(int)));
  const auto args = tack::GAGArgs{
    .A = a,
    .B = b,
    .C = c,
    .signals =  signals,
    .epochs = epochs,
    .epoch = 0,
    .putSync = putSync,
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
  if (rank == 0)
  {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }

  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  auto gag = [&](const uint& runs) {
    for (int i = 0; i < runs; ++i) {
      tack::gemmAGKernel<<<blocks, threads, kernelSharedSize, stream>>>(args);
    }
  };
  gag(1);
  nvshmem_barrier_all();
  // call and bench reference (NCCL AG + cuBLASLt)
  const auto nccl_cublas_time = reference(args, cRef);
  using MT = MXE<Element>;
  // check correctness
  matx::cudaExecutor exec{stream};
  auto tC = matx::make_tensor<MT>(reinterpret_cast<MT*>(c), {opts.localM, N});
  auto tCRef = matx::make_tensor<MT>(reinterpret_cast<MT*>(cRef), {opts.localM, N});
  auto num_matches = matx::make_tensor<long int>({});
  (num_matches = matx::sum(matx::isclose(tC, tCRef, opts.rtol, opts.atol))).run(exec);
  exec.sync();
  // calculate error percentage
  const auto ep =  (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(tC.TotalSize()))) * 100;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  gag(opts.warmup);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaEventRecord(start);
  gag(opts.runs);
  cudaEventRecord(stop, stream);
  CHECK_CUDA(cudaEventSynchronize(stop));
  float m_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&m_ms, start, stop));
  const float m_time_ms = m_ms / static_cast<float>(opts.runs);
  constexpr auto es = element_string<Element>();
  printf("%d, %d, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %f, %d, %d, %f, %f\n",
           rank, world, es, opts.localM, opts.localN, opts.K, bM, bN, bK, threads, bps, num_sms, blocks, opts.rtol,
           opts.atol, ep, opts.warmup, opts.runs, m_time_ms, nccl_cublas_time);
  CHECK_CUDA(cudaFreeAsync(a, stream));
  CHECK_CUDA(cudaFreeAsync(c, stream));
  CHECK_CUDA(cudaFreeAsync(cRef, stream));
  CHECK_CUDA(cudaFreeAsync(putSync, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
  nvshmem_free(b);
  nvshmem_free(signals);
  nvshmem_free(epochs);
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}
int main() {
  printf("Hello World\n");
}