//
// Created by azureuser on 3/25/26.
//
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>

#include "../../include/suture/suture.cuh"
#include "../../include/suture/atomic_ar.cuh"
#include "../common.cuh"
#include "../debug.cuh"

constexpr auto NE = ncclFloat16;
// AllReduce reference kernel, not an optimal implementation
__global__ void rk(suture::RedElement** __restrict__ bufs, const int rank, const int world, const size_t elems) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= elems) {
    return;
  }
  auto* __restrict__ result = bufs[rank];
  using AccumType = cuda::std::common_type_t<suture::RedElement, float>;
  auto accumulator = static_cast<AccumType>(0.f);
  for (int i = 0; i < world; ++i) {
    constexpr Converter<AccumType, suture::RedElement> loadConv{};
    accumulator += loadConv(bufs[i][tid]);
  }
  constexpr Converter<suture::RedElement, AccumType> storeConv{};
  result[tid] = storeConv(accumulator);
}

__host__
void arHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  cuda::std::byte* rcvBuff = nullptr;
  uint64_t* completions = nullptr;
  uint64_t* arrivals = nullptr;
  uint8_t* senseBitsTR = nullptr;
  // latency-regime buffers
  uint8_t* senseBitsLR = nullptr;
  uint8_t* flags = nullptr;
  cuda::std::byte* staging = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("world,bytes,type,suture(ms),suture(GB/s),error_o(%%),error_n(%%),"
           "threads,pipeStages,stageExtent,unrollFactor,"
           "totalSMsOnGPU,superBlockSize,blocks,warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  const int kernelSharedSize = opts.maxLocalBytes > suture::AR_LATENCY_BOUND_THRESHOLD ?
  suture::kThreads * suture::TR_RED_ALIGNMENT * suture::kPipeStages * suture::kStageExtent : 0;
  if (opts.maxLocalBytes > suture::AR_LATENCY_BOUND_THRESHOLD) {
    int maxSharedMemory = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
    if (kernelSharedSize > maxSharedMemory) {
      const auto errmsg = std::string("Required shared memory ").append(std::to_string(kernelSharedSize))
      .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
      throw std::runtime_error(errmsg);
    }
    CHECK_CUDA(cudaFuncSetAttribute(allReduceTR, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSharedSize));
  }
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, allReduceTR, suture::kThreads, kernelSharedSize));
  int bps1 = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps1, allReduceLR, suture::kThreads, 0));
  bps = min(bps1, bps);
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  const auto actualWorld = world - 1;
  const auto maxActualSBSize = getSBZ<suture::AG_SUPER_BLOCK_THRESHOLD>(world, opts.maxLocalBytes);
  opts.maxSuperBlockSize = opts.maxSuperBlockSize <= 0 ? maxActualSBSize : min(opts.maxSuperBlockSize, maxActualSBSize);
  const auto requestedCTAs = opts.maxSuperBlockSize * actualWorld;
  const auto availableCTAs = bps * num_sms;
  const auto superBlockSize0 = requestedCTAs > availableCTAs ?
  (cuda::round_down(availableCTAs, actualWorld) / actualWorld) : opts.maxSuperBlockSize;
  const auto signalLength = world * opts.maxSuperBlockSize;
  completions = static_cast<uint64_t*>(nvshmem_calloc(signalLength, sizeof(uint64_t)));
  arrivals = static_cast<uint64_t*>(nvshmem_calloc(signalLength, sizeof(uint64_t)));
  CHECK_CUDA(cudaMallocAsync(&senseBitsLR, sizeof(uint8_t) * signalLength, stream));
  CHECK_CUDA(cudaMemsetAsync(senseBitsLR, 0, sizeof(uint8_t) * signalLength, stream));
  CHECK_CUDA(cudaMallocAsync(&senseBitsTR, sizeof(uint8_t) * signalLength, stream));
  CHECK_CUDA(cudaMemsetAsync(senseBitsTR, 0, sizeof(uint8_t) * signalLength, stream));
  rcvBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxLocalBytes));
  cuda::std::byte* refBuff = nullptr;
  CHECK_CUDA(cudaMallocAsync(&refBuff, opts.maxLocalBytes, stream));
  std::vector<cuda::std::byte*> dataBuffs(world, nullptr);
  for (auto & dataBuff : dataBuffs) {
    CHECK_CUDA(cudaMallocAsync(&dataBuff, opts.maxLocalBytes, stream));
  }
  srcBuff = dataBuffs[rank];
  void* devBs = nullptr;
  CHECK_CUDA(cudaMallocAsync(&devBs, sizeof(cuda::std::byte*) * world, stream));
  CHECK_CUDA(cudaMemcpyAsync(devBs, dataBuffs.data(), sizeof(cuda::std::byte*) * world, cudaMemcpyHostToDevice, stream));

  // *2 for double-buffering
  staging = static_cast<cuda::std::byte*>(nvshmem_calloc(2 * world * suture::PACKET_BUFFER_SIZE, sizeof(uint8_t)));
  if (staging == nullptr || !cuda::is_aligned(staging, suture::RED_MAX_ALIGNMENT)) {
    throw std::runtime_error("staging memory allocation failed");
  }
  const auto flagBytes = (2 * world * sizeof(uint8_t) * suture::AR_LATENCY_BOUND_THRESHOLD) / suture::LR_RED_ALIGNMENT;
  CHECK_CUDA(cudaMallocAsync(&flags, flagBytes, stream));
  CHECK_CUDA(cudaMemsetAsync(flags, 0, flagBytes, stream));
  if (rcvBuff == nullptr || !cuda::is_aligned(rcvBuff, suture::RED_MAX_ALIGNMENT)) {
    throw std::runtime_error("rcvBuff is invalid");
  }
  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::random_device rd;
  auto agk = [&](const auto& blocks, const ARArgs& kArgs, const size_t& bytes, const int& runs, const bool isLR) {
    if (isLR) {
      // low-latency
      for (int i = 0; i < runs; ++i) {
        cudaMemcpyAsync(kArgs.dst, kArgs.src, bytes, cudaMemcpyDeviceToDevice, stream);
        allReduceLR<<<blocks, suture::kThreads, 0, stream>>>(kArgs);
      }
    }
    else {
      for (int i = 0; i < runs; ++i) {
        cudaMemcpyAsync(kArgs.dst, kArgs.src, bytes, cudaMemcpyDeviceToDevice, stream);
        allReduceTR<<<blocks, suture::kThreads, kernelSharedSize, stream>>>(kArgs);
      }
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const cuda::fast_mod_div<int> world_v{world};
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    // fill buffer with random values
    uint seed;
    if (rank == 0) {
      seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    static_assert(suture::RED_MAX_ALIGNMENT % sizeof(suture::RedElement) == 0);
    const auto elems = bytes / sizeof(suture::RedElement);
    for (int i = 0; i < world; ++i) {
      const auto theirSeed = seed + i * 42;
      auto* cB = reinterpret_cast<suture::RedElement*>(dataBuffs[i]);
      randUniform<ARCH>(cB, elems, theirSeed, -1.f, 1.f, stream);
    }
    CHECK_CUDA(cudaMemcpyAsync(refBuff, srcBuff, bytes, cudaMemcpyDeviceToDevice, stream));
    const auto dataAlignment = bytes <= suture::AR_LATENCY_BOUND_THRESHOLD ? suture::LR_RED_ALIGNMENT : suture::TR_RED_ALIGNMENT;
    auto superBlockSize = static_cast<int>(min(cuda::ceil_div(bytes, suture::kThreads * dataAlignment),
      static_cast<size_t>(superBlockSize0)));
    if (world < 8 && superBlockSize > 16) {
      // A100
      superBlockSize = bytes < suture::AR_SUPER_BLOCK_THRESHOLD ? 16 : superBlockSize;
    }
    const size_t scaledChunkSize = bytes / dataAlignment;
    const cuda::fast_mod_div<int> superBlockSize_v{superBlockSize};
    const auto isLR = bytes <= suture::AR_LATENCY_BOUND_THRESHOLD;
    const ARArgs args{
      .src = srcBuff,
      .dst = rcvBuff,
      .completions = completions,
      .arrivals = arrivals,
      .senseBitsTR = senseBitsTR,
      .senseBitsLR = senseBitsLR,
      .staging = staging,
      .flagSense = flags,
      .ctaBaseChunk = scaledChunkSize / superBlockSize,
      .superBlockSize_v = superBlockSize_v,
      .world_v = world_v,
      .chunkResidue = static_cast<int>(scaledChunkSize % superBlockSize),
      .maxSuperBlockSize = opts.maxSuperBlockSize,
      .rank = rank,
      .world = world
    };
    const auto blocks = superBlockSize * actualWorld;
    // correctness run
    agk(blocks, args, bytes, 1, isLR);
    ncclAllReduce(refBuff, refBuff, elems, NE, ncclSum, comm, stream);
    auto ar_matches0 = matx::make_tensor<long int>({});
    using MRE = MXE<suture::RedElement>;
    auto tR = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(rcvBuff), {1, static_cast<matx::index_t>(elems)});
    auto tRef = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(refBuff), {1, static_cast<matx::index_t>(elems)});
    // correctness check against nccl
    (ar_matches0 = matx::sum(matx::isclose(tR, tRef, opts.rtol, opts.atol))).run(exec);

    constexpr uint rkThreads = 512;
    const auto rkBlocks = cuda::ceil_div(elems, rkThreads);
    rk<<<rkBlocks, rkThreads, 0, stream>>>(static_cast<suture::RedElement**>(devBs), rank, world, elems);
    auto tO = matx::make_tensor<MRE>(reinterpret_cast<MRE*>(srcBuff), {1, static_cast<matx::index_t>(elems)});
    // correctness check against oracle
    auto ar_matches1 = matx::make_tensor<long int>({});
    (ar_matches1 = matx::sum(matx::isclose(tR, tO, opts.rtol, opts.atol))).run(exec);
    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      agk(blocks, args, bytes, opts.runs, isLR);
      CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

      CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // warmup
      CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // time total launches = opts.runs * opts.graph_launches
      const int total_launches = opts.runs * opts.graph_launches;

      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < opts.graph_launches; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
      }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

      // per-iteration time (each launch is one iteration)
      t_ms = total_ms / static_cast<float>(total_launches);

      CHECK_CUDA(cudaGraphExecDestroy(graphExec));
      CHECK_CUDA(cudaGraphDestroy(graph));
    }
    else {
      // benchmark suture without graphs
      agk(blocks, args, bytes, opts.warmup, isLR);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      agk(blocks, args, bytes, opts.runs, isLR);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }

    times.ep = (1.0 - (static_cast<double>(ar_matches0()) / static_cast<double>(tR.TotalSize()))) * 100.0;
    times.oracle_ep = (1.0 - (static_cast<double>(ar_matches1()) / static_cast<double>(tR.TotalSize()))) * 100.0;
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = (static_cast<double>(bytes)) / 1e9;
      const auto suture_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %s, %lf, %lf, %lf, %lf, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
        world, bytes, element_string<suture::RedElement>(), times.t_ms, suture_algBW, times.oracle_ep, times.ep,
        suture::kThreads, suture::kPipeStages, suture::kStageExtent, suture::kUnrollFactor,
        num_sms, superBlockSize, blocks,  opts.graph_launches > 0 ? opts.runs : opts.warmup,
        opts.runs,opts.graph_launches);
    }
  }
  CHECK_CUDA(cudaFreeAsync(senseBitsLR, stream));
  CHECK_CUDA(cudaFreeAsync(senseBitsTR, stream));
  for (auto & dataBuff : dataBuffs) {
    CHECK_CUDA(cudaFreeAsync(dataBuff, stream));
  }
  CHECK_CUDA(cudaFreeAsync(flags, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_free(staging);
  nvshmem_free(arrivals);
  nvshmem_free(completions);
  nvshmem_free(rcvBuff);
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}
// ./ag <minLocalBytes> <maxLocalBytes> <graph_launches> <maxSuperBlockSize> <runs> <warmup> <rtol> <atol>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.maxSuperBlockSize = -1;
  opts.graph_launches = 16;
  opts.rtol = 2e-2;
  opts.atol = 2e-3;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.graph_launches = std::stoi(argv[3]);
  if (argc > 4) opts.maxSuperBlockSize = std::stoi(argv[4]);
  if (argc > 5) opts.runs = std::stoi(argv[5]);
  if (argc > 6) opts.warmup = std::stoi(argv[6]);
  if (argc > 7) opts.rtol = std::stof(argv[7]);
  if (argc > 8) opts.atol = std::stof(argv[8]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % suture::RED_MAX_ALIGNMENT != 0 || opts.maxLocalBytes % suture::RED_MAX_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(suture::RED_MAX_ALIGNMENT) + " bytes");
  }
  arHost(opts);
}