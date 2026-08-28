#ifndef PURLIN_SUPPORT_BENCHMARK_NCCL_COMMUNICATOR_CUH
#define PURLIN_SUPPORT_BENCHMARK_NCCL_COMMUNICATOR_CUH

#include "checks.cuh"

namespace bench {

class NcclCommunicator {
public:
  NcclCommunicator() = default;
  NcclCommunicator(const NcclCommunicator&) = delete;
  NcclCommunicator& operator=(const NcclCommunicator&) = delete;
  ~NcclCommunicator() { release(); }

  void initialize(const int rank, const int world) {
    release();
    ncclUniqueId identifier{};
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&identifier));
    MPI_CHECK(MPI_Bcast(&identifier, sizeof(identifier), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommInitRank(&communicator_, world, identifier, rank));
  }

  void release() {
    if (communicator_ == nullptr) return;
    ncclCommFinalize(communicator_);
    ncclCommDestroy(communicator_);
    communicator_ = nullptr;
  }

  void checkAsyncError() const {
    ncclResult_t asynchronous = ncclSuccess;
    NCCL_CHECK(ncclCommGetAsyncError(communicator_, &asynchronous));
    if (asynchronous != ncclSuccess) {
      std::fprintf(stderr, "NCCL asynchronous error: %s\n", ncclGetErrorString(asynchronous));
      MPI_Abort(MPI_COMM_WORLD, static_cast<int>(asynchronous));
    }
  }

  ncclComm_t get() const { return communicator_; }
  explicit operator bool() const { return communicator_ != nullptr; }

private:
  ncclComm_t communicator_ = nullptr;
};

} // namespace bench

#endif
