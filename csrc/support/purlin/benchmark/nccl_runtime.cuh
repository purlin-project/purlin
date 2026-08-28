#ifndef PURLIN_SUPPORT_BENCHMARK_NCCL_RUNTIME_CUH
#define PURLIN_SUPPORT_BENCHMARK_NCCL_RUNTIME_CUH

#include <stdexcept>
#include <string>

#include "checks.cuh"
#include "nccl_communicator.cuh"

namespace bench {

class NcclRuntime {
public:
  NcclRuntime(int& argc, char**& argv) {
    int initialized = 0;
    MPI_CHECK(MPI_Initialized(&initialized));
    if (!initialized) {
      MPI_CHECK(MPI_Init(&argc, &argv));
      ownsMpi_ = true;
    }

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
    if (world < 2) throw std::runtime_error("NCCL benchmarks require at least two MPI ranks");

    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
      rank, MPI_INFO_NULL, &localCommunicator_));
    MPI_CHECK(MPI_Comm_rank(localCommunicator_, &localRank));

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (localRank >= deviceCount) {
      throw std::runtime_error("Node-local MPI rank " + std::to_string(localRank) +
        " has no corresponding visible CUDA device");
    }

    device = localRank;
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProperties, device));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    ownedCommunicator_.initialize(rank, world);
    communicator = ownedCommunicator_.get();
  }

  NcclRuntime(const NcclRuntime&) = delete;
  NcclRuntime& operator=(const NcclRuntime&) = delete;

  ~NcclRuntime() {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    ownedCommunicator_.release();
    communicator = nullptr;
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    if (localCommunicator_ != MPI_COMM_NULL) {
      MPI_Comm_free(&localCommunicator_);
    }
    if (ownsMpi_) {
      int finalized = 0;
      MPI_Finalized(&finalized);
      if (!finalized) MPI_Finalize();
    }
  }

  void checkAsyncError() const {
    ownedCommunicator_.checkAsyncError();
  }

  int rank = 0;
  int world = 0;
  int localRank = 0;
  int device = 0;
  cudaDeviceProp deviceProperties{};
  cudaStream_t stream = nullptr;
  ncclComm_t communicator = nullptr;

private:
  MPI_Comm localCommunicator_ = MPI_COMM_NULL;
  NcclCommunicator ownedCommunicator_{};
  bool ownsMpi_ = false;
};

} // namespace bench

#endif
