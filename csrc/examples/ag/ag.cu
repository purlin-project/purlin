//
// Created by osayamen on 2/18/26.
//
#include <random>
#include <string>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>

#include <suture/suture.cuh>

#include "../common.cuh"
#include "../debug.cuh"

constexpr auto threads = 128;
constexpr auto unrollFactor = 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 4;
constexpr auto elementsPerThread = 8;

constexpr auto nArch = suture::normalizeArch<ARCH>();
using SutureConfig = suture::Configuration<
    nArch,
    threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor,
    suture::AUTO
>;

struct Args {
  const cuda::std::byte* const src;
  cuda::std::byte* const dst;
  const size_t bytes;
};

template<typename SutureAtom>
__global__ void allGatherKernel(const __grid_constant__ Args kArgs,
  const __grid_constant__ suture::SutureContext ctx) {
  extern __shared__ __align__(SutureAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
  suture::allGather<SutureAtom>(kArgs.dst, kArgs.src, kArgs.bytes, workspace, ctx);
}

__host__
void agHost(RunOptions& opts) {
  cuda::std::byte* rcvBuff = nullptr; // [world, size], symmetric

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank == 0) {
    printf("world,localBytes,globalBytes,suture(ms),suture(GB/s),error(%%),nArch,GPUName,threads,pipeStages,stageExtent,unrollFactor,"
           "SMsOnGPU,superBlockSize,blocks,warmup,runs,graph_launches\n");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  auto ctx = suture::initialize(rank, world, stream);

  using SutureAtom = suture::Atom<nArch, SutureConfig>;
  auto kernel = allGatherKernel<SutureAtom>;
  constexpr auto kernelSharedSize = SutureAtom::SMEM_SIZE;
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  if (kernelSharedSize > maxSharedMemory) {
    const auto errmsg = std::string("Required shared memory ").append(std::to_string(kernelSharedSize))
    .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
    throw std::runtime_error(errmsg);
  }
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSharedSize));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, SutureAtom::THREADS, kernelSharedSize));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  const auto actualWorld = world - 1;
  const auto maxActualSBSize = getSBZ<nArch, suture::AG_SUPER_BLOCK_THRESHOLD>(world, opts.maxLocalBytes);
  opts.maxSuperBlockSize = opts.maxSuperBlockSize <= 0 ? maxActualSBSize : min(opts.maxSuperBlockSize, maxActualSBSize);
  const auto requestedCTAs = opts.maxSuperBlockSize * actualWorld;
  const auto availableCTAs = bps * num_sms;
  const auto superBlockSize0 = requestedCTAs > availableCTAs ?
  (cuda::round_down(availableCTAs, actualWorld) / actualWorld) : opts.maxSuperBlockSize;
  rcvBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxLocalBytes * world));
  auto* refBuff = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxLocalBytes * world));
  if (rcvBuff == nullptr || !cuda::is_aligned(rcvBuff, suture::MAX_ACCESS_ALIGNMENT)) {
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
  auto agk = [&](const auto& blocks, const Args& kArgs, const suture::SutureContext& kCtx, const int& runs) {
    for (int i = 0; i < runs; ++i) {
      allGatherKernel<SutureAtom><<<blocks, SutureAtom::THREADS, kernelSharedSize, stream>>>(kArgs, kCtx);
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const cuda::fast_mod_div<int> world_v{world};
  for (size_t localBytes = opts.minLocalBytes; localBytes <= opts.maxLocalBytes; localBytes *= 2) {
    // fill buffer with random values
    const auto seed = rd();
    static_assert(suture::MAX_ACCESS_ALIGNMENT % sizeof(float) == 0);
    const auto elems = localBytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(rcvBuff) + (rank * elems);
    randUniform<ARCH>(tS, elems, seed, -1.f, 1.f, stream);
    auto* tSr = reinterpret_cast<float*>(refBuff) + (rank * elems);
    randUniform<ARCH>(tSr, elems, seed, -1.f, 1.f, stream);
    auto superBlockSize = static_cast<int>(min(cuda::ceil_div(localBytes, SutureAtom::THREADS * suture::MAX_ACCESS_ALIGNMENT),
      static_cast<size_t>(superBlockSize0)));
    if (world < 8 && superBlockSize > 16) {
      // A100
      superBlockSize = localBytes < suture::AG_SUPER_BLOCK_THRESHOLD ? 16 : superBlockSize;
    }
    const size_t scaledChunkSize = localBytes / suture::MAX_ACCESS_ALIGNMENT;
    const cuda::fast_mod_div<int> superBlockSize_v{superBlockSize};
    const Args kArgs{
      .src = rcvBuff + (rank * localBytes),
      .dst = rcvBuff,
      .bytes = localBytes
    };
    ctx.setSuperBlockSize(superBlockSize);
    const auto blocks = superBlockSize * actualWorld;
    // correctness run
    agk(blocks, kArgs, ctx, 1);
    auto* sB = refBuff + (rank * localBytes);
    ncclAllGather(sB, refBuff, localBytes, ncclUint8, comm, stream);
    auto ag_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(rcvBuff), {1, static_cast<matx::index_t>(elems * world)});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff), {1, static_cast<matx::index_t>(elems * world)});
    // bitwise check
    (ag_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);

    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      agk(blocks, kArgs, ctx, opts.runs);
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
      agk(blocks, kArgs, ctx,opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      agk(blocks, kArgs, ctx, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }

    times.ep = (1.0 - (static_cast<double>(ag_matches()) / static_cast<double>(tR.TotalSize()))) * 100.0;
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = (world * static_cast<double>(localBytes)) / 1e9;
      const auto suture_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %lu, %lf, %lf, %lf, %d, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
        world, localBytes, world * localBytes,times.t_ms, suture_algBW, times.ep, nArch,
        prop.name, SutureAtom::THREADS, pipeStages, elementsPerThread, unrollFactor, num_sms, superBlockSize, blocks,
        opts.graph_launches > 0 ? opts.runs : opts.warmup, opts.runs,opts.graph_launches);
    }
  }
  suture::finalize(ctx, stream);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_free(rcvBuff);
  nvshmem_free(refBuff);
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}
// ./ag <minLocalBytes> <maxLocalBytes> <maxSuperBlockSize> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.maxSuperBlockSize = -1;
  opts.graph_launches = 8;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.maxSuperBlockSize = std::stoi(argv[3]);
  if (argc > 4) opts.graph_launches = std::stoi(argv[4]);
  if (argc > 5) opts.runs = std::stoi(argv[5]);
  if (argc > 6) opts.warmup = std::stoi(argv[6]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % suture::MAX_ACCESS_ALIGNMENT != 0 || opts.maxLocalBytes % suture::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(suture::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  agHost(opts);
}
