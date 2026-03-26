//
// Created by Osayamen on 3/9/26.
//
#include <random>

#include <matx.h>
#include <mpi.h>
#include <cuda/cmath>
#include <nvshmem.h>

#include "../common.cuh"
#include "../debug.cuh"

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
    printf("bytes,ce(ms),ce(GB/s),error(%%),GPUName,warmup,runs,graph_launches\n");
    fflush(stdout);
  }
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId)); // Get properties for current rank

  CHECK_CUDA(cudaMallocAsync(&srcBuf, opts.maxLocalBytes, stream));
  dstBuf = static_cast<cuda::std::byte*>(nvshmem_malloc(opts.maxLocalBytes));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::random_device rd;
  auto pk = [&stream](cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes, const int& runs = 1) {
    for (int i = 0; i < runs; ++i) {
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    }
  };
  matx::cudaExecutor exec{stream};
  Times times{};
  const auto peer = rank == 0 ? 1 : 0;
  CHECK_CUDA(cudaPeekAtLastError());
  auto* translatedBuf = static_cast<cuda::std::byte*>(nvshmem_ptr(dstBuf, peer));
  //auto* translatedBuf = dstBuf;
  for (size_t localBytes = opts.minLocalBytes; localBytes <= opts.maxLocalBytes; localBytes *= 2) {
    uint seed;
    if (rank == 0) {
      seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // fill buffer with random values
    const auto mySeed = seed + rank;
    const auto elems = localBytes / sizeof(float);
    auto* tS = reinterpret_cast<float*>(srcBuf);
    randUniform<ARCH>(tS, elems, mySeed, -1.f, 1.f, stream);
    nvshmemx_sync_all_on_stream(stream); // ensures the buffer is available
    pk(translatedBuf, srcBuf, localBytes, 1);
    nvshmemx_sync_all_on_stream(stream); // ensures we have received the peer's payload
    // check correctness
    const auto expectedSeed = seed + peer;
    randUniform<ARCH>(tS, elems, expectedSeed, -1.f, 1.f, stream);
    auto p2p_matches = matx::make_tensor<long int>({});
    auto tR = matx::make_tensor<float>(reinterpret_cast<float*>(dstBuf), {1, static_cast<matx::index_t>(elems)});
    auto tRef = matx::make_tensor<float>(tS, {1, static_cast<matx::index_t>(elems)});
    // bitwise check
    (p2p_matches = matx::sum(matx::isclose(tR, tRef, 0, 0))).run(exec);
    nvshmemx_sync_all_on_stream(stream); // ensures we complete the correctness checks before subsequent transfers
    // benchmark p2p
    float t_ms = 0.0f;
    if (opts.graph_launches > 0) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graphExec = nullptr;

      // capture kernel launches
      CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      pk(translatedBuf, srcBuf, localBytes, opts.runs);
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
      // benchmark without graphs
      pk(translatedBuf, srcBuf, localBytes, opts.warmup);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      cudaEventRecord(start, stream);
      pk(translatedBuf, srcBuf, localBytes, opts.runs);
      cudaEventRecord(stop, stream);
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&t_ms, start, stop));
      t_ms /= static_cast<float>(opts.runs);
    }
    times.ep = 1.0 - (static_cast<double>(p2p_matches()) / static_cast<double>(tR.TotalSize()));
    times.t_ms = t_ms;
    // get max results across ranks
    MPI_Allreduce(MPI_IN_PLACE, &times, sizeof(Times) / sizeof(double), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const auto gb = static_cast<double>(localBytes) / 1e9;
      const auto tack_algBW = gb / (times.t_ms * 1e-3);
      printf("%lu,%lf, %lf, %lf, %s, %d, %d, %d\n",
        localBytes,times.t_ms, tack_algBW, times.ep, prop.name, 
        opts.graph_launches > 0 ? opts.runs : opts.warmup, opts.runs, opts.graph_launches);
    }
  }
  // 7) Synchronize / cleanup
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
}

// ./ce_p2p <minBytes> <maxBytes> <graph_launches> <runs> <warmup>
int main(const int argc, char** argv) {
  RunOptions opts{};
  opts.graph_launches = 8;
  if (argc > 1) opts.minLocalBytes = parseSize(argv[1]);
  if (argc > 2) opts.maxLocalBytes = parseSize(argv[2]);
  if (argc > 3) opts.graph_launches = std::stoi(argv[3]);
  if (argc > 4) opts.runs = std::stoi(argv[4]);
  if (argc > 5) opts.warmup = std::stoi(argv[5]);
  if (!cuda::is_power_of_two(opts.minLocalBytes) || !cuda::is_power_of_two(opts.maxLocalBytes)) {
    throw std::invalid_argument("Sizes must be a power of two");
  }
  if (opts.minLocalBytes % sizeof(float) != 0 || opts.maxLocalBytes % sizeof(float) != 0) {
    throw std::invalid_argument("Size must be a multiple of " + std::to_string(sizeof(float)) + " bytes");
  }
  p2pHost(opts);
}