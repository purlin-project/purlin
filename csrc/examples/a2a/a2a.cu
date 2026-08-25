//
// Created by osayamen on 5/24/26.
//
#include <random>
#include <string>
#include <stdexcept>

#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nccl.h>

#include <purlin/core.cuh>
#include <purlin/host/tuning.cuh>

#include <util.cuh>

constexpr auto threads = 128;
constexpr auto unrollFactor = 4;
constexpr auto alignment = 16;

constexpr auto pipeStages = 8;
constexpr auto elementsPerThread = 2;

constexpr auto nArch = purlin::normalizeArch<ARCH>();
constexpr auto worldUnroll = 2;

template<int NArch>
__host__ constexpr size_t all2allLatencyThreshold(const int world) {
  switch (world) {
    case 2: return purlin::host::All2AllTuning<NArch, 2>::LATENCY_THRESHOLD;
    case 4: return purlin::host::All2AllTuning<NArch, 4>::LATENCY_THRESHOLD;
    case 8: return purlin::host::All2AllTuning<NArch, 8>::LATENCY_THRESHOLD;
    default: return purlin::host::All2AllTuning<NArch, purlin::host::FALLBACK>::LATENCY_THRESHOLD;
  }
}

using TRConfig = purlin::Configuration<
        threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor
>;
constexpr auto t128Lower = 128 * 1024;
constexpr auto t128Higher = 1024 * 1024;
using TR128Config = purlin::Configuration<
        128, /*threads*/
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor
>;

using LRConfig = purlin::Configuration<
    512, /*threads*/
  alignment,
  purlin::UNUSED,
  purlin::UNUSED,
  unrollFactor,
  worldUnroll
>;

struct Args {
  const cuda::std::byte* const src;
  cuda::std::byte* const dst;
  const size_t bytes;
  const cuda::fast_mod_div<long int> blocks;
};
// 2MiB -> 4MiB <= globalBytes <= 16MiB
// 4MiB -> 32MiB <= globalBytes <= 128MiB
//constexpr size_t CHUNK_SIZE = 8 * 1024 * 1024;
constexpr size_t CHUNK_SIZE = 2 * 1024 * 1024;
constexpr int NON_CHUNKED_PUT_BLOCKS = 32;
constexpr int CHUNKED_PUT_BLOCKS = 32;
constexpr int LOCAL_PUT_BLOCKS = 8;
template<typename PurlinAtom, typename CollConfig>
__launch_bounds__(PurlinAtom::THREADS, 1)
__global__ void all2all(const __grid_constant__ Args kArgs, const __grid_constant__ purlin::Context ctx) {
  extern __shared__ __align__(SAMPLE_SMEM_ALIGNMENT) cuda::std::byte workspace[];
  const purlin::SnacArgs<cuda::fast_mod_div<long int>> args{
    .dst = kArgs.dst,
    .src = kArgs.src,
    .bytes = kArgs.bytes,
    .workspace = workspace,
    .blocks = kArgs.blocks,
    .collBlocks = static_cast<int>(kArgs.blocks),
  };
  purlin::all2all<PurlinAtom, CollConfig>(args, ctx);
}

__host__ __forceinline__
void all2allReference(const cuda::std::byte* src,
  cuda::std::byte* dst,
  const size_t bytes,
  const int rank,
  const int world,
  ncclComm_t comm,
  cudaStream_t stream) {
  CHECK_CUDA(cudaMemcpyAsync(dst + rank * bytes, src + rank * bytes, bytes, cudaMemcpyDeviceToDevice, stream));
  NCCL_CHECK(ncclGroupStart());
  for (int peer = 0; peer < world; ++peer) {
    if (peer == rank) {
      continue;
    }
    NCCL_CHECK(ncclSend(src + peer * bytes, bytes, ncclUint8, peer, comm, stream));
    NCCL_CHECK(ncclRecv(dst + peer * bytes, bytes, ncclUint8, peer, comm, stream));
  }
  NCCL_CHECK(ncclGroupEnd());
}

__host__
void a2aHost(RunOptions& opts) {
  cuda::std::byte* srcBuff = nullptr;
  cuda::std::byte* dstBuff = nullptr;
  cuda::std::byte* refBuff = nullptr;

  nvshmem_init();
  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (world <= 1) {
    printf("Requires at least two processes!\n");
  }
  if (rank == 0) {
    printf("world,localBytes,globalBytes,purlin(ms),purlin(GB/s),error(%%),nArch,GPUName,threads,"
           "pipeStages,stageExtent,unrollFactor,worldUnroll,"
           "SMsOnGPU,stagingBlocks,localPutBlocks,consumerBlocks,blocks,chunkSize(MiB),warmup,runs,graph_launches\n");
  }
  if (world > purlin::MAX_RANKS_PER_DOMAIN) {
    throw std::runtime_error(std::to_string(world) + "exceeds max allowed of " +
      std::to_string(purlin::MAX_RANKS_PER_DOMAIN) + "ranks");
  }
  if (NON_CHUNKED_PUT_BLOCKS % world != 0) {
    throw std::runtime_error("non-chunked put blocks: " + std::to_string(NON_CHUNKED_PUT_BLOCKS) +
      " must be a multiple of world");
  }
  if (CHUNKED_PUT_BLOCKS % world != 0) {
    throw std::runtime_error("chunked put blocks: " + std::to_string(CHUNKED_PUT_BLOCKS) +
      " must be a multiple of world");
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  const auto workspace = makeWorkspace(world, stream);
  auto ctx = purlin::initialize(rank, world, workspace, stream);
  using PurlinAtomLR = purlin::Atom<nArch, LRConfig>;
  using PurlinAtomTR = purlin::Atom<nArch, TRConfig>;
  using PurlinAtomTR128 = purlin::Atom<nArch, TR128Config>;
  using nonChunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::nonChunked,
    purlin::UNUSED,
    purlin::UNUSED,
    CHUNK_SIZE,
    LOCAL_PUT_BLOCKS
  >;
  using chunkedConfig = purlin::CollectiveConfig<
    purlin::CollectiveType::chunked,
    purlin::UNUSED,
    purlin::UNUSED,
    CHUNK_SIZE,
    LOCAL_PUT_BLOCKS
  >;
  constexpr auto kSTR = purlin::copySmemBytes<PurlinAtomTR>();
  constexpr auto kSTR128 = purlin::copySmemBytes<PurlinAtomTR128>();
  constexpr auto kSLR = purlin::copySmemBytes<PurlinAtomLR, purlin::Regime::latency>();
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  auto kernelTRNonChunked = all2all<PurlinAtomTR, nonChunkedConfig>;
  auto kernelTR128NonChunked = all2all<PurlinAtomTR128, nonChunkedConfig>;
  auto kernelTRChunked = all2all<PurlinAtomTR, chunkedConfig>;
  auto kernelLR = all2all<PurlinAtomLR, purlin::CollectiveConfigLR>;
  {
    if (kSTR > maxSharedMemory) {
      const auto errmsg = std::string("Required shared memory ").append(std::to_string(kSTR))
      .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
      throw std::runtime_error(errmsg);
    }
    CHECK_CUDA(cudaFuncSetAttribute(kernelTRNonChunked, cudaFuncAttributeMaxDynamicSharedMemorySize, kSTR));
    CHECK_CUDA(cudaFuncSetAttribute(kernelTR128NonChunked, cudaFuncAttributeMaxDynamicSharedMemorySize, kSTR128));
    CHECK_CUDA(cudaFuncSetAttribute(kernelTRChunked, cudaFuncAttributeMaxDynamicSharedMemorySize, kSTR));
  }
  {
    if (kSLR > maxSharedMemory) {
      const auto errmsg = std::string("Required shared memory ").append(std::to_string(kSLR))
      .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory));
      throw std::runtime_error(errmsg);
    }
    CHECK_CUDA(cudaFuncSetAttribute(kernelLR, cudaFuncAttributeMaxDynamicSharedMemorySize, kSLR));
  }
  opts.maxSuperBlockSize = opts.maxSuperBlockSize <= 0 ? (world == 2 ? 32 : (32 / world)) : opts.maxSuperBlockSize;
  const auto CTAsUpperLR = cuda::std::min(64U, cuda::std::bit_floor(static_cast<uint32_t>(num_sms)));

  CHECK_CUDA(cudaMallocAsync(&srcBuff, opts.maxLocalBytes * world, stream));
  CHECK_CUDA(cudaMallocAsync(&dstBuff, opts.maxLocalBytes * world, stream));
  CHECK_CUDA(cudaMallocAsync(&refBuff, opts.maxLocalBytes * world, stream));
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
  auto a2aK = [&](const auto& blocks, const Args& kArgs, const purlin::Context& kCtx, const bool isLR, const int& runs) {
    if (isLR) {
      for (int i = 0; i < runs; ++i) {
        all2all<PurlinAtomLR, purlin::CollectiveConfigLR>
        <<<blocks, PurlinAtomLR::THREADS, kSLR, stream>>>(kArgs, kCtx);
      }
    }
    else {
      if (kArgs.bytes <= CHUNK_SIZE) {
        if (world >= 4 && (kArgs.bytes >= t128Lower && kArgs.bytes <= t128Higher)) {
          for (int i = 0; i < runs; ++i) {
            all2all<PurlinAtomTR128, nonChunkedConfig>
            <<<blocks, PurlinAtomTR128::THREADS, kSTR128, stream>>>(kArgs, kCtx);
          }
        }
        else {
          for (int i = 0; i < runs; ++i) {
            all2all<PurlinAtomTR, nonChunkedConfig>
            <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, kCtx);
          }
        }
      }
      else {
        for (int i = 0; i < runs; ++i) {
          all2all<PurlinAtomTR, chunkedConfig>
          <<<blocks, PurlinAtomTR::THREADS, kSTR, stream>>>(kArgs, kCtx);
        }
      }
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const auto actualWorld = world - 1;
  const auto nNonChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
    cuda::round_down(NON_CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
  const auto nChunkedPB = cuda::std::bit_floor(static_cast<uint32_t>(
    cuda::round_down(CHUNKED_PUT_BLOCKS, actualWorld) / actualWorld));
  for (size_t localBytes = opts.minLocalBytes; localBytes <= opts.maxLocalBytes; localBytes *= 2) {
    const auto stagingBlocks = (localBytes <= CHUNK_SIZE ? nNonChunkedPB : nChunkedPB)* actualWorld;
    const auto putBlocks = stagingBlocks + LOCAL_PUT_BLOCKS;
    const auto superUpper = cuda::std::bit_floor(cuda::round_down(num_sms - putBlocks, actualWorld) / actualWorld);
    const auto maxSuperBlockSize = cuda::std::min(static_cast<uint>(opts.maxSuperBlockSize), superUpper);
    ctx.stagingBlocks = cuda::fast_mod_div<long int>{static_cast<long int>(stagingBlocks)};
    // fill buffer with random values
    const auto seed = rd();
    static_assert(alignment % sizeof(float) == 0);
    const auto elems = (localBytes * world) / sizeof(float);
    auto* tS = reinterpret_cast<float*>(srcBuff);
    randUniform<ARCH>(tS, elems, seed, -1.f, 1.f, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto isLR = localBytes <= all2allLatencyThreshold<nArch>(world);
    int blocks = 0;
    if (isLR) {
      blocks = cuda::std::min(cuda::ceil_div(localBytes, PurlinAtomLR::THREADS*sizeof(purlin::LRP::RT)),
        static_cast<size_t>(CTAsUpperLR));
    }
    else {
      auto blocksNeeded = static_cast<int>(min((localBytes / PurlinAtomTR::RED_PIPELINE_BYTES),
        static_cast<size_t>(maxSuperBlockSize)) * actualWorld);
      blocksNeeded = localBytes <= static_cast<size_t>((8 * 1024 * 1024) / world) ?
      cuda::round_down(cuda::std::min(blocksNeeded, 32), actualWorld) : blocksNeeded;
      blocks = putBlocks + blocksNeeded;
      if (blocksNeeded < actualWorld) {
        // non-pipelined path
        blocks = putBlocks + (cuda::std::min(cuda::ceil_div(localBytes,
          static_cast<size_t>(PurlinAtomTR::THREADS*PurlinAtomTR::BaseConfig::ALIGNMENT_BYTES)),
          static_cast<size_t>(maxSuperBlockSize)) * actualWorld);
      }
    }
    const Args kArgs{
      .src = srcBuff,
      .dst = dstBuff,
      .bytes = localBytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    // correctness run
    a2aK(blocks, kArgs, ctx, isLR, 1);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    all2allReference(srcBuff, refBuff, localBytes, rank, world, comm, stream);
    auto a2a_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(dstBuff), {1, static_cast<matx::index_t>(elems)});
    auto tRef = matx::make_tensor<float>(reinterpret_cast<float*>(refBuff), {1, static_cast<matx::index_t>(elems)});
    // bitwise check
    (a2a_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);

    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      a2aK(blocks, kArgs, ctx, isLR, opts.runs);
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
      // benchmark purlin without graphs
      a2aK(blocks, kArgs, ctx, isLR, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      a2aK(blocks, kArgs, ctx, isLR, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }

    times.ep = (1.0 - (static_cast<double>(a2a_matches()) / static_cast<double>(tR.TotalSize()))) * 100.0;
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const auto usedThreads = (world >= 4 && kArgs.bytes >= t128Lower && kArgs.bytes <= t128Higher) ?
    PurlinAtomTR128::THREADS : PurlinAtomTR::THREADS;
    if (rank == 0) {
      const auto gb = (world * static_cast<double>(localBytes)) / 1e9;
      const auto purlin_algBW = gb / (times.t_ms * 1e-3);
      printf("%d, %lu, %lu, %lf, %lf, %lf, %d, %s, %d, %s, %s, %s, %s, %d, %s, %s, %s, %d, %s, %d, %d, %d\n",
        world, localBytes, world * localBytes,times.t_ms, purlin_algBW, times.ep, nArch,
        prop.name,
        isLR ? PurlinAtomLR::THREADS : usedThreads,
        isLR ? "N/A" : std::to_string(pipeStages).c_str(),
        isLR ? "N/A" : std::to_string(elementsPerThread).c_str(),
        isLR ? "N/A" : std::to_string(unrollFactor).c_str(),
        isLR ? std::to_string(worldUnroll).c_str() : "N/A",
        num_sms,
        isLR ? "N/A" : std::to_string(stagingBlocks).c_str(),
        isLR ? "N/A" : std::to_string(LOCAL_PUT_BLOCKS).c_str(),
        isLR ? "N/A" : std::to_string(blocks - putBlocks).c_str(),
        blocks,
        isLR ? "N/A" : std::to_string(CHUNK_SIZE / (1024UL * 1024)).c_str(),
        opts.graph_launches > 0 ? opts.runs : opts.warmup, opts.runs,opts.graph_launches);
    }
  }
  CHECK_CUDA(cudaFreeAsync(srcBuff, stream));
  CHECK_CUDA(cudaFreeAsync(dstBuff, stream));
  CHECK_CUDA(cudaFreeAsync(refBuff, stream));
  purlin::finalize(ctx, stream);
  destroyWorkspace(workspace, rank, stream);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  nvshmem_finalize();
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}
// NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_BOOTSTRAP=MPI mpirun -n <world> ./a2a <minLocalBytes> <maxLocalBytes> <maxSuperBlockSize> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.maxSuperBlockSize = -1; // -1 -> autotuned
  opts.graph_launches = 8;
  opts.warmup = 128;
  opts.runs = 128;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.maxSuperBlockSize = std::stoi(argv[3]);
  if (argc > 4) opts.graph_launches = std::stoi(argv[4]);
  if (argc > 5) opts.runs = std::stoi(argv[5]);
  if (argc > 6) opts.warmup = std::stoi(argv[6]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0 || opts.maxLocalBytes % purlin::MAX_ACCESS_ALIGNMENT != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(purlin::MAX_ACCESS_ALIGNMENT) + " bytes");
  }
  a2aHost(opts);
}
