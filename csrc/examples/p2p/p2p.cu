//
// Created by Osayamen on 3/9/26.
//
#include <random>
#include <cuda/cmath>

#include <matx.h>
#include <mpi.h>
#include <nvshmem.h>

#include <purlin/core.cuh>

#include <util.cuh>

constexpr auto threads = 64;
constexpr auto unrollFactor = 2;
constexpr auto alignment = 16;

constexpr auto pipeStages = 2;
constexpr auto elementsPerThread = 16;
constexpr auto nArch = purlin::normalizeArch<ARCH>();
using PurlinConfig = purlin::Configuration<
    purlin::Regime::throughput,
    threads,
    alignment,
    pipeStages,
    elementsPerThread,
    unrollFactor
>;

struct Args {
  cuda::std::byte* const src;
  cuda::std::byte* const dst;
  const size_t bytes;
  const cuda::fast_mod_div<long int> blocks;
};

constexpr int P2P_TURNOVER_THRESHOLD = ARCH >= 900 ? (1024 * 1024) : (512 * 1024);
template<typename PurlinAtom>
__launch_bounds__(PurlinAtom::THREADS, 1)
__global__ void p2pK(const __grid_constant__ Args kArgs) {
  extern __shared__ __align__(PurlinAtom::Config::ALIGNMENT_BYTES) cuda::std::byte workspace[];
  purlin::superPut<PurlinAtom>(kArgs.dst, kArgs.src, kArgs.bytes, workspace, kArgs.blocks);
}

__host__
void p2pHost(RunOptions& opts) {
  cuda::std::byte* srcBuf = nullptr; // local
  cuda::std::byte* dstBuf = nullptr; // symmetric
  nvshmem_init();

  const auto world = nvshmem_n_pes();
  const auto rank = nvshmem_my_pe();
  const auto devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

  if (world != 2) {
    if (rank == 0) {
      printf("Two processes required\n");
    }
    return;
  }
  if (rank == 0) {
    printf("bytes,purlin(ms),purlin(GB/s),error(%%),nArch,GPUName,threads,pipeStages,stageExtent,unrollFactor,"
           "SMsOnGPU,blocks,warmup,runs,graph_launches\n");
    fflush(stdout);
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  constexpr auto maxActualSBSize = 64;
  opts.maxSuperBlockSize = min(opts.maxSuperBlockSize, maxActualSBSize);
  CHECK_CUDA(cudaMallocAsync(&srcBuf, opts.maxLocalBytes, stream));
  using PurlinAtom = purlin::Atom<nArch, PurlinConfig>;
  auto kernel = p2pK<PurlinAtom>;
  dstBuf = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxLocalBytes));
  constexpr auto kernelSharedSize = PurlinAtom::COPY_SMEM_SIZE;
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
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::random_device rd;
  auto pk = [&](const auto& blocks, const Args& kArgs, const int& runs = 1) {
    if (rank == 0) {
      for (int i = 0; i < runs; ++i) {
        p2pK<PurlinAtom><<<blocks, threads, kernelSharedSize, stream>>>(kArgs);
      }
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const auto peer = rank == 0 ? 1 : 0;
  CHECK_CUDA(cudaPeekAtLastError());
  auto* translatedBuf = static_cast<cuda::std::byte*>(nvshmem_ptr(dstBuf, peer));
  //auto* translatedBuf = dstBuf;
  for (size_t bytes = opts.minLocalBytes; bytes <= opts.maxLocalBytes; bytes *= 2) {
    uint seed;
    if (rank == 0) {
      seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // fill buffer with random values
    const auto mySeed = seed + rank;
    static_assert(alignment % sizeof(float) == 0);
    const auto elems = bytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(srcBuf);
    randUniform<ARCH>(tS, elems, mySeed, -1.f, 1.f, stream);
    auto blocks = static_cast<int>(min(cuda::ceil_div(bytes, static_cast<size_t>(PurlinAtom::THREADS * alignment)),
      static_cast<size_t>(opts.maxSuperBlockSize)));
    if (bytes >= P2P_TURNOVER_THRESHOLD) {
      blocks = cute::min(bytes / PurlinAtom::COPY_PIPELINE_BYTES, opts.maxSuperBlockSize);
    }
    const auto usedPipelining = (bytes / blocks) >= PurlinAtom::COPY_PIPELINE_BYTES;
    nvshmemx_sync_all_on_stream(stream); // ensures the buffer is available
    const Args kArgs{
      .src = srcBuf,
      .dst = translatedBuf,
      .bytes = bytes,
      .blocks = cuda::fast_mod_div<long int>{blocks}
    };
    pk(blocks, kArgs);
    CHECK_CUDA(cudaPeekAtLastError());
    nvshmemx_barrier_all_on_stream(stream); // ensures we have received the peer's payload
    // check correctness
    const auto expectedSeed = seed + peer;
    randUniform<ARCH>(tS, elems, expectedSeed, -1.f, 1.f, stream);
    auto p2p_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(dstBuf), {1, static_cast<matx::index_t>(elems)});
    auto tRef = matx::make_tensor<float>(tS, {1, static_cast<matx::index_t>(elems)});
    // bitwise check
    if (rank == 1) {
      (p2p_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);
    }
    nvshmemx_sync_all_on_stream(stream); // ensures we complete the correctness checks before subsequent transfers
    // benchmark purlin p2p
    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      pk(blocks, kArgs, opts.runs);
      CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

      CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      // warmup once
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
      pk(blocks, kArgs, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      pk(blocks, kArgs, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }
    times.ep = (1.0 - (static_cast<double>(p2p_matches()) / static_cast<double>(tR.TotalSize()))) * 100;
    times.t_ms = t_ms;
    // aggregate results across ranks
    MPI_Bcast(&times.ep, 1, MPI_DOUBLE, 1, MPI_COMM_WORLD);
    MPI_Bcast(&times.t_ms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = static_cast<double>(bytes) / 1e9;
      const auto purlin_algBW = gb / (times.t_ms * 1e-3);
      printf("%lu,%lf, %lf, %lf, %d, %s, %d, %s, %s, %d, %d, %d, %d, %d, %d\n",
        bytes,times.t_ms, purlin_algBW, times.ep, nArch, prop.name,
        threads,
        usedPipelining ? std::to_string(pipeStages).c_str() : "N/A",
        usedPipelining ? std::to_string(elementsPerThread).c_str() : "N/A",
        unrollFactor,
        num_sms,
        blocks,
        opts.graph_launches > 0 ? opts.runs : opts.warmup,
        opts.runs, opts.graph_launches);
    }
  }
  // 7) Synchronize / cleanup
  CHECK_CUDA(cudaFreeAsync(srcBuf, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvshmem_free(dstBuf);
  nvshmem_finalize();
  CHECK_CUDA(cudaStreamDestroy(stream));
}

// ./p2p <minBytes> <maxBytes> <maxSuperBlockSize> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.maxSuperBlockSize = ARCH >= 900 ? 16 : 8;
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
  if (opts.minLocalBytes % alignment != 0 || opts.maxLocalBytes % alignment != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(alignment) + " bytes");
  }
  p2pHost(opts);
}
