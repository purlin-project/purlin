//
// Created by Osayamen on 1/8/26.
//

#include <cuda_fp16.h>
#include <cstdio>
#include <random>
#include <vector>

#include <mpi.h>
#include <matx.h>
#include <nccl.h>

#include "ag.cuh"
#include "common.cuh"
#include "debug.cuh"

using Element = __half;
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
  size_t localM = 128;
  size_t localN = 256;
  size_t K = 128;
  int warmup = 128;
  int runs = 128;
  float rtol = 2e-2;
  float atol = 2e-3;
};

__host__ __forceinline__
uint parse_integer(const char* s, const char* name, const uint lo = 1, const uint hi = std::numeric_limits<uint>::max()) {
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
  cuda::std::byte* cRef = nullptr;
  uint64_t* completions = nullptr; // [ctas], symmetric
  uint64_t* arrivals = nullptr; // [ctas, world], symmetric

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("Rank, world, dtype, localM, localN, K, threads, blocks/SM, SMs, blocks, rtol, atol, "
           "ag_error(%%), gemm_ag_error(%%), warmup, runs, tack_cublaslt_Time(ms), nccl_cublaslt_time(ms)\n");
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
  CHECK_CUDA(cudaMallocAsync(&cRef, sizeof(Element) * opts.localM * static_cast<size_t>(N), stream));
  b = static_cast<cuda::std::byte*>(nvshmem_malloc(sizeof(Element) * N * opts.K));
  bRef = static_cast<cuda::std::byte*>(nvshmem_malloc(sizeof(Element) * N * opts.K));
  // get number of CTAs.
  auto kernel = ag;
  constexpr auto kernelSharedSize = threads * Alignment * pipeStages * stageExtent;
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  if (kernelSharedSize > maxSharedMemory) {
    const auto errmsg = std::string("Required shared memory ").append(std::to_string(kernelSharedSize))
    .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
    throw std::runtime_error(errmsg);
  }
  // opt-in
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSharedSize));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, kernelSharedSize));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  const auto actualWorld = world - 1;
  const size_t chunkSize = opts.localN * static_cast<size_t>(opts.K) * sizeof(Element);
  const auto blocks = min(cuda::ceil_div(chunkSize, threads * Alignment) * actualWorld,
    static_cast<size_t>(num_sms * bps));

  completions = static_cast<uint64_t*>(nvshmem_calloc(blocks, sizeof(uint64_t)));
  arrivals = static_cast<uint64_t*>(nvshmem_calloc(blocks * world, sizeof(uint64_t)));

  if (chunkSize % MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("localN * K should be a multiple of " + std::to_string(MAX_ACCESS_ALIGNMENT));
  }
  if (actualWorld > static_cast<int>(blocks)) {
    // this is not a functional requirement, just a simplifying assumption
    throw std::invalid_argument("(World - 1) should be <= " + std::to_string(blocks));
  }
  // fill our input buffer with random values
  const auto bSeed = rd();
  auto* localB = reinterpret_cast<Element*>(b + chunkSize * rank);
  randUniform<ARCH>(localB, opts.localN * static_cast<size_t>(opts.K), bSeed, min_v, max_v, stream);
  auto* localBRef = reinterpret_cast<Element*>(bRef + chunkSize * rank);
  randUniform<ARCH>(localBRef, opts.localN * static_cast<size_t>(opts.K), bSeed, min_v, max_v, stream);

  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }

  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  AGArgs args{
    .sendBuff = b + rank * chunkSize,
    .completions = completions,
    .arrivals = arrivals,
    .signal = 1,
    .size = chunkSize,
    .rank = rank,
    .world = world
  };

  matx::cudaExecutor exec{stream};

  // call and bench reference (NCCL AG + cuBLASLt)
  constexpr auto ndt = get_nccl_type<Element>();
  const auto tA = matx::make_tensor<MT>(reinterpret_cast<MT*>(a),
    {static_cast<matx::index_t>(opts.localM), static_cast<matx::index_t>(opts.K)});
  auto tB = matx::make_tensor<MT>(reinterpret_cast<MT*>(b),
    {static_cast<matx::index_t>(N), static_cast<matx::index_t>(opts.K)});
  const auto tBRef = matx::make_tensor<MT>(reinterpret_cast<MT*>(bRef), tB.Shape());
  auto tC = matx::make_tensor<MT>(reinterpret_cast<MT*>(c),
    {static_cast<matx::index_t>(opts.localM), static_cast<matx::index_t>(N)});
  auto tCRef = matx::make_tensor<MT>(reinterpret_cast<MT*>(cRef), tC.Shape());

  auto gag = [&](const int& runs) {
    nvtx3::scoped_range r{"TACK"};
    if (world > 1) {
      for (int i = 0; i < runs; ++i) {
        ag<<<blocks, threads, kernelSharedSize, stream>>>(args);
        // do cuBLASLt GEMM via MatX
        (tC = matx::matmul(tA, tB.PermuteMatrix())).run(exec);
        args.signal += 1;
      }
    }
    else {
      for (int i = 0; i < runs; ++i) {
        (tC = matx::matmul(tA, tB.PermuteMatrix())).run(exec);
      }
    }
  };
  gag(1);
  CHECK_CUDA(cudaPeekAtLastError());
  const auto* sendBuff = bRef + rank * chunkSize;
  auto refK = [&](const int& runs) {
    nvtx3::scoped_range r{"NCCL+cuBLAS"};
    if (world > 1) {
      for (int i = 0; i < runs; ++i) {
        // do NCCL AG
        ncclAllGather(sendBuff, bRef, chunkSize, ncclUint8, comm, exec.getStream());
        // do cuBLASLt GEMM via MatX
        (tCRef = matx::matmul(tA, tBRef.PermuteMatrix())).run(exec);
      }
    }
    else {
      for (int i = 0; i < runs; ++i) {
        (tCRef = matx::matmul(tA, tBRef.PermuteMatrix())).run(exec);
      }
    }
  };
  refK(1);
  CHECK_CUDA(cudaPeekAtLastError());
  // check correctness of ag
  auto ag_matches = matx::make_tensor<long int>({});
  // bitwise check
  (ag_matches = matx::sum(matx::isclose(tB, tBRef, 0, 0))).run(exec);
  // check correctness of gemm+ag
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
  printf("%d, %d, %s, %lu, %lu, %lu, %d, %d, %d, %lu, %.1e, %.1e, %lf, %lf, %d, %d, %f, %f\n",
           rank, world, es, opts.localM, opts.localN, opts.K, threads, bps, num_sms, blocks, opts.rtol,
           opts.atol, ag_ep, ep, opts.warmup, opts.runs, m_ms, nccl_cublas_time);
  CHECK_CUDA(cudaEventDestroy(start));CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFreeAsync(a, stream));
  CHECK_CUDA(cudaFreeAsync(c, stream));
  CHECK_CUDA(cudaFreeAsync(cRef, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
  nvshmem_free(b);
  nvshmem_free(bRef);
  nvshmem_free(completions);
  nvshmem_free(arrivals);
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