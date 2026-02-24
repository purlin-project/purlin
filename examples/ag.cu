//
// Created by osayamen on 2/18/26.
//
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>

#include "ag.cuh"
#include "common.cuh"
#include "debug.cuh"

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
  constexpr auto kernelSharedSize = threads * Alignment * pipeStages * stageExtent;
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  if (kernelSharedSize > maxSharedMemory) {
    const auto errmsg = std::string("Required shared memory ").append(std::to_string(kernelSharedSize))
    .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
    throw std::runtime_error(errmsg);
  }
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSharedSize));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, kernelSharedSize));
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
      ag<<<blocks, threads, kernelSharedSize, stream>>>(kArgs);
      kArgs.signal += 1;
    }
  };
  auto nag = [&](auto* const& sb, auto* const& rb, const auto& count, const int& runs) {
    for (int i = 0; i < runs; ++i) {
      //ncclAllGather(sb, rb, count, ncclUint8, comm, stream);
      nvshmemx_fcollectmem_on_stream(NVSHMEM_TEAM_WORLD, rb, sb, count, stream);
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
