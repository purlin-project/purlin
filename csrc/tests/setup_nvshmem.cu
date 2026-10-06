#include <contrib/symm_mem.cuh>
#include <purlin/setup.cuh>

__global__ void markPeers(purlin::Context ctx) {
  for (int peer = 0; peer < static_cast<int>(ctx.world); ++peer) {
    ctx.signals[peer][ctx.rank] = ctx.rank + 1;
  }
}

int main() {
  nvshmem_init();
  const int rank = nvshmem_my_pe();
  const int world = nvshmem_n_pes();
  CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const size_t staging : {purlin::MIN_CHUNK_SIZE, purlin::STAGING_BUFFER_SIZE_}) {
    auto managed = purlin::initialize(rank, world, stream, purlin::NvshmemMemory{}, staging);
    auto& ctx = managed.context();
    std::vector<uint64_t*> peers(world);
    CHECK_CUDA(cudaMemcpy(peers.data(), ctx.signals, sizeof(uint64_t*) * world, cudaMemcpyDeviceToHost));
    std::vector<uint64_t> signals(world);
    for (int peer = 0; peer < world; ++peer) {
      CHECK_CUDA(cudaMemcpy(signals.data(), peers[peer], sizeof(uint64_t) * world, cudaMemcpyDeviceToHost));
      for (const auto value : signals) {
        if (value != 0) nvshmem_global_exit(EXIT_FAILURE);
      }
    }
    nvshmem_barrier_all();
    markPeers<<<1, 1, 0, stream>>>(ctx);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();
    CHECK_CUDA(cudaMemcpy(signals.data(), peers[rank], sizeof(uint64_t) * world, cudaMemcpyDeviceToHost));
    for (int peer = 0; peer < world; ++peer) {
      if (signals[peer] != static_cast<uint64_t>(peer + 1)) nvshmem_global_exit(EXIT_FAILURE);
    }
    purlin::finalize(managed, stream);
    purlin::finalize(managed, stream);
  }
  CHECK_CUDA(cudaStreamDestroy(stream));
  if (rank == 0) std::puts("NVSHMEM managed setup checks passed");
  nvshmem_finalize();
  return EXIT_SUCCESS;
}
