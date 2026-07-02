//
// Created by osayamen on 6/29/26.
//

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>
#include <nvshmem.h>

#include <purlin/core.cuh>
#include <purlin/host/all2all.cuh>

#include "util.cuh"

__host__ __forceinline__
auto makeSplitsForSource(const size_t baseBytes, const int src, const int world) {
  const auto peerBase = cuda::round_down(baseBytes / static_cast<size_t>(world), size_t{32});
  std::vector<size_t> splits(world, peerBase);
  splits[world - 1] += baseBytes - peerBase * static_cast<size_t>(world);
  if (peerBase >= static_cast<size_t>(64 * world)) {
    for (int dst = 0; dst < world - 1; ++dst) {
      const auto delta = static_cast<long long>(((src + dst) % 2) == 0 ? -128 : 128);
      splits[dst] = static_cast<size_t>(static_cast<long long>(splits[dst]) + delta);
      splits[world - 1] = static_cast<size_t>(static_cast<long long>(splits[world - 1]) - delta);
    }
  }
  return splits;
}

__host__ __forceinline__
auto makeInSplits(const size_t baseBytes, const int rank, const int world) {
  return makeSplitsForSource(baseBytes, rank, world);
}

__host__ __forceinline__
auto makeOutSplits(const size_t baseBytes, const int rank, const int world) {
  std::vector<size_t> splits(world);
  for (int peer = 0; peer < world; ++peer) {
    const auto peerSplits = makeSplitsForSource(baseBytes, peer, world);
    splits[peer] = peerSplits[rank];
  }
  return splits;
}

__host__ __forceinline__
auto makeOffsets(const std::vector<size_t>& sizes) {
  std::vector<size_t> offsets(sizes.size());
  size_t offset = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    offsets[i] = offset;
    offset += sizes[i];
  }
  return offsets;
}

__host__ __forceinline__
auto sumBytes(const std::vector<size_t>& sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), size_t{0});
}

__host__ __forceinline__
auto maxBytes(const std::vector<size_t>& sizes) {
  return *std::max_element(sizes.begin(), sizes.end());
}

__host__ __forceinline__
auto makeVState(const std::vector<size_t>& inSplits,
  const std::vector<size_t>& outSplits,
  const std::vector<size_t>& inOffsets,
  const int rank) {
  return purlin::VState{
    .maxOutBytes = maxBytes(outSplits),
    .maxBytes = maxBytes(inSplits),
    .totalBytes = sumBytes(inSplits),
    .totalOutBytes = sumBytes(outSplits),
    .offset = inOffsets[rank],
    .bytes = inSplits[rank],
  };
}

__host__ __forceinline__
void ncclAll2allVReference(const cuda::std::byte* __restrict__ const& src,
  cuda::std::byte* __restrict__ const& dst,
  const std::vector<size_t>& inSplits,
  const std::vector<size_t>& outSplits,
  const std::vector<size_t>& inOffsets,
  const std::vector<size_t>& outOffsets,
  const int rank,
  const int world,
  ncclComm_t comm,
  cudaStream_t stream) {
  if (inSplits[rank] > 0) {
    CHECK_CUDA(cudaMemcpyAsync(dst + outOffsets[rank], src + inOffsets[rank],
      inSplits[rank], cudaMemcpyDeviceToDevice, stream));
  }
  NCCL_CHECK(ncclGroupStart());
  for (int peer = 0; peer < world; ++peer) {
    if (peer == rank) {
      continue;
    }
    if (inSplits[peer] > 0) {
      NCCL_CHECK(ncclSend(src + inOffsets[peer], inSplits[peer], ncclUint8, peer, comm, stream));
    }
    if (outSplits[peer] > 0) {
      NCCL_CHECK(ncclRecv(dst + outOffsets[peer], outSplits[peer], ncclUint8, peer, comm, stream));
    }
  }
  NCCL_CHECK(ncclGroupEnd());
}

template<typename Op>
__host__ __forceinline__
auto timeOperation(cudaStream_t stream,
  cudaEvent_t start,
  cudaEvent_t stop,
  const RunOptions& opts,
  Op&& op) {
  float tMs = 0.0f;
  if (opts.graph_launches > 0) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < opts.runs; ++i) {
      op();
    }
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const int totalLaunches = opts.runs * opts.graph_launches;
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < opts.graph_launches; ++i) {
      CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));
    tMs = totalMs / static_cast<float>(totalLaunches);

    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
  }
  else {
    for (int i = 0; i < opts.warmup; ++i) {
      op();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < opts.runs; ++i) {
      op();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&tMs, start, stop));
    tMs /= static_cast<float>(opts.runs);
  }
  return tMs;
}

__host__
void a2avHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  cuda::std::byte* dstBuff = nullptr;
  cuda::std::byte* refBuff = nullptr;
  cuda::std::byte* fixedDstBuff = nullptr;
  size_t* devInSplits = nullptr;
  size_t* devOutSplits = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (world <= 1) {
    printf("Requires at least two processes!\n");
    return;
  }
  if (rank == 0) {
    printf("world,messageBytes,totalInBytes,totalOutBytes,maxInBytes,maxOutBytes,"
           "all2allV(ms),all2allV(GB/s),all2all(ms),all2all(GB/s),ncclAll2allV(ms),ncclAll2allV(GB/s),"
           "error_vs_nccl(%%),GPUName,warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId));

  const auto workspace = makeWorkspace(world, stream);
  auto ctx = purlin::initialize(rank, world, workspace, stream);

  const auto maxInSplits = makeInSplits(opts.maxLocalBytes, rank, world);
  const auto maxOutSplits = makeOutSplits(opts.maxLocalBytes, rank, world);
  const auto maxInTotalBytes = sumBytes(maxInSplits);
  const auto maxOutTotalBytes = sumBytes(maxOutSplits);
  const auto maxFixedPeerBytes = cuda::round_down(opts.maxLocalBytes / static_cast<size_t>(world),
    static_cast<size_t>(purlin::MAX_ACCESS_ALIGNMENT));
  const auto maxFixedBytes = maxFixedPeerBytes * static_cast<size_t>(world);
  const auto maxBufferBytes = std::max({maxInTotalBytes, maxOutTotalBytes, maxFixedBytes});
  if (maxInTotalBytes > ctx.stagingTRSize) {
    throw std::runtime_error("Total all2allV input bytes exceeds staging limit");
  }

  CHECK_CUDA(cudaMallocAsync(&srcBuff, maxBufferBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&dstBuff, maxBufferBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&refBuff, maxBufferBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&fixedDstBuff, maxBufferBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&devInSplits, sizeof(size_t) * world, stream));
  CHECK_CUDA(cudaMallocAsync(&devOutSplits, sizeof(size_t) * world, stream));

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
  matx::cudaExecutor exec{stream};
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    const auto inSplits = makeInSplits(bytes, rank, world);
    const auto outSplits = makeOutSplits(bytes, rank, world);
    const auto inOffsets = makeOffsets(inSplits);
    const auto outOffsets = makeOffsets(outSplits);
    const auto fixedPeerBytes = cuda::round_down(bytes / static_cast<size_t>(world),
      static_cast<size_t>(purlin::MAX_ACCESS_ALIGNMENT));
    const auto fixedTotalBytes = fixedPeerBytes * static_cast<size_t>(world);
    ctx.vState = makeVState(inSplits, outSplits, inOffsets, rank);

    CHECK_CUDA(cudaMemcpyAsync(devInSplits, inSplits.data(), sizeof(size_t) * world, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(devOutSplits, outSplits.data(), sizeof(size_t) * world, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(dstBuff, 0, maxBufferBytes, stream));
    CHECK_CUDA(cudaMemsetAsync(refBuff, 0, maxBufferBytes, stream));
    CHECK_CUDA(cudaMemsetAsync(fixedDstBuff, 0, maxBufferBytes, stream));

    const auto seed = rd();
    static_assert(purlin::MAX_ACCESS_ALIGNMENT % sizeof(float) == 0);
    const auto fillElems = maxBufferBytes / sizeof(float);
    randUniform<ARCH>(reinterpret_cast<float*>(srcBuff), fillElems, seed, -1.f, 1.f, stream);

    constexpr int MAX_BLOCKS = 128;
    const size_t timingBufBytes = MAX_BLOCKS * purlin::TIMING_SLOTS * sizeof(unsigned long long);
    unsigned long long* devTimingBuf = nullptr;
    unsigned long long hostTimingBuf[MAX_BLOCKS * purlin::TIMING_SLOTS];
    CHECK_CUDA(cudaMalloc(&devTimingBuf, timingBufBytes));

    auto purlinAll2allV = [&] {
      purlin::all2allV(srcBuff, dstBuff, devInSplits, devOutSplits, ctx, stream);
    };
    auto purlinAll2all = [&] {
      purlin::all2all(srcBuff, fixedDstBuff, fixedPeerBytes, ctx, stream);
    };
    auto ncclAll2allV = [&] {
      ncclAll2allVReference(srcBuff, refBuff, inSplits, outSplits, inOffsets, outOffsets, rank, world, comm, stream);
    };

    purlinAll2allV();
    ncclAll2allV();

    auto matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(dstBuff),
      {1, static_cast<matx::index_t>(ctx.vState.totalOutBytes / sizeof(float))});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff),
      {1, static_cast<matx::index_t>(ctx.vState.totalOutBytes / sizeof(float))});
    (matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const auto a2avMs = timeOperation(stream, start, stop, opts, purlinAll2allV);
    const auto a2aMs = timeOperation(stream, start, stop, opts, purlinAll2all);
    const auto ncclMs = timeOperation(stream, start, stop, opts, ncclAll2allV);

    // Collect per-block timing from one invocation of each kernel (non-graph, no warmup interference)
    auto collectTiming = [&](auto& op, const char* label) {
      ctx.timingBuf = nullptr;
      for (int w = 0; w < 8; ++w) { op(); }
      CHECK_CUDA(cudaStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);
      CHECK_CUDA(cudaMemset(devTimingBuf, 0, timingBufBytes));
      ctx.timingBuf = devTimingBuf;
      op();
      CHECK_CUDA(cudaStreamSynchronize(stream));
      ctx.timingBuf = nullptr;
      CHECK_CUDA(cudaMemcpy(hostTimingBuf, devTimingBuf, timingBufBytes, cudaMemcpyDeviceToHost));
      // Print LPUT blocks only (the bottleneck)
      const char* btNames[] = {"PROD", "LPUT", "CONS"};
      for (int b = 0; b < MAX_BLOCKS; ++b) {
        auto* t = hostTimingBuf + b * purlin::TIMING_SLOTS;
        if (t[0] == 0) continue;
        int bt = static_cast<int>(t[4]);
        if (bt == purlin::BT_LOCAL_PUT) {
          printf("TIMING %s rank=%d blk=%d type=LPUT gstart=%llu gend=%llu copy=%llu sm=%llu intra=%llu\n",
            label, rank, b, t[0], t[1], t[2], t[3], t[7]);
        }
      }
      // Print summary
      unsigned long long maxByType[3] = {0, 0, 0};
      unsigned long long sumByType[3] = {0, 0, 0};
      int countByType[3] = {0, 0, 0};
      for (int b = 0; b < MAX_BLOCKS; ++b) {
        auto* t = hostTimingBuf + b * purlin::TIMING_SLOTS;
        if (t[0] == 0) continue;
        int bt = static_cast<int>(t[4]);
        auto val = (bt == purlin::BT_LOCAL_PUT) ? (t[1] - t[0]) : t[0];
        sumByType[bt] += val;
        countByType[bt]++;
        if (val > maxByType[bt]) maxByType[bt] = val;
      }
      printf("TIMING_SUMMARY %s rank=%d", label, rank);
      for (int bt = 0; bt < 3; ++bt) {
        if (countByType[bt] > 0) {
          printf(" %s:max=%llu,avg=%llu,n=%d",
            btNames[bt], maxByType[bt], sumByType[bt]/countByType[bt], countByType[bt]);
        }
      }
      printf("\n");
    };
    collectTiming(purlinAll2allV, "V");
    collectTiming(purlinAll2all, "FIXED");

    CHECK_CUDA(cudaFree(devTimingBuf));

    double metrics[6] = {
      static_cast<double>(a2avMs),
      static_cast<double>(a2aMs),
      static_cast<double>(ncclMs),
      (1.0 - (static_cast<double>(matches()) /
        static_cast<double>(ctx.vState.totalOutBytes / sizeof(float)))) * 100.0,
      static_cast<double>(ctx.vState.totalOutBytes),
      static_cast<double>(fixedTotalBytes),
    };
    MPI_Allreduce(MPI_IN_PLACE, metrics, 6, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto a2avGb = metrics[4] / 1e9;
      const auto a2aGb = metrics[5] / 1e9;
      const auto a2avBw = a2avGb / (metrics[0] * 1e-3);
      const auto a2aBw = a2aGb / (metrics[1] * 1e-3);
      const auto ncclBw = a2avGb / (metrics[2] * 1e-3);
      printf("%d, %lu, %lu, %lu, %lu, %lu, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %s, %d, %d, %d\n",
        world, bytes, ctx.vState.totalBytes, ctx.vState.totalOutBytes, ctx.vState.maxBytes, ctx.vState.maxOutBytes,
        metrics[0], a2avBw, metrics[1], a2aBw, metrics[2], ncclBw, metrics[3], prop.name,
        opts.graph_launches > 0 ? opts.runs : opts.warmup, opts.runs, opts.graph_launches);
    }
  }

  CHECK_CUDA(cudaFreeAsync(srcBuff, stream));
  CHECK_CUDA(cudaFreeAsync(dstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  CHECK_CUDA(cudaFreeAsync(fixedDstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(devInSplits, stream));
  CHECK_CUDA(cudaFreeAsync(devOutSplits, stream));
  purlin::finalize(ctx, stream);
  destroyWorkspace(workspace, rank, stream);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}

// NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none mpirun -n <world> ./testA2AV <minBytes> <maxBytes> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.runs = 128;
  opts.warmup = 128;
  opts.graph_launches = 8;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.graph_launches = std::stoi(argv[3]);
  if (argc > 4) opts.runs = std::stoi(argv[4]);
  if (argc > 5) opts.warmup = std::stoi(argv[5]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0 ||
    opts.maxLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " +
      std::to_string(purlin::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  a2avHost(opts);
}
