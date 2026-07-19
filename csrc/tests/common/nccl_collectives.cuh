#ifndef PURLIN_TESTS_COMMON_NCCL_COLLECTIVES_CUH
#define PURLIN_TESTS_COMMON_NCCL_COLLECTIVES_CUH

#include <cstddef>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>

#include "checks.cuh"

namespace bench {

template<typename Element>
consteval ncclDataType_t ncclDataType() {
  if constexpr (cuda::std::is_same_v<Element, __half>) return ncclHalf;
  else if constexpr (cuda::std::is_same_v<Element, __nv_bfloat16>) return ncclBfloat16;
  else if constexpr (cuda::std::is_same_v<Element, float>) return ncclFloat;
  else if constexpr (cuda::std::is_same_v<Element, double>) return ncclDouble;
  else {
    static_assert(cuda::std::is_same_v<Element, void>, "Unsupported NCCL reduction data type");
    return ncclNumTypes;
  }
}

template<typename Byte>
inline void ncclAllToAll(const Byte* source, Byte* destination,
  const size_t peerBytes, const int rank, const int world,
  ncclComm_t communicator, cudaStream_t stream) {
  static_assert(sizeof(Byte) == 1);
  if (peerBytes > 0) {
    CHECK_CUDA(cudaMemcpyAsync(destination + rank * peerBytes,
      source + rank * peerBytes, peerBytes, cudaMemcpyDeviceToDevice, stream));
  }
  NCCL_CHECK(ncclGroupStart());
  for (int peer = 0; peer < world; ++peer) {
    if (peer == rank || peerBytes == 0) continue;
    NCCL_CHECK(ncclSend(source + peer * peerBytes, peerBytes, ncclUint8,
      peer, communicator, stream));
    NCCL_CHECK(ncclRecv(destination + peer * peerBytes, peerBytes, ncclUint8,
      peer, communicator, stream));
  }
  NCCL_CHECK(ncclGroupEnd());
}

template<typename Byte>
inline void ncclAllGatherV(const Byte* source, Byte* destination,
  const std::vector<size_t>& sizes, const std::vector<size_t>& offsets,
  const int rank, const int world, ncclComm_t communicator, cudaStream_t stream) {
  static_assert(sizeof(Byte) == 1);
  NCCL_CHECK(ncclGroupStart());
  for (int root = 0; root < world; ++root) {
    if (sizes[root] == 0) continue;
    const void* send = rank == root ? static_cast<const void*>(source)
                                    : static_cast<const void*>(destination + offsets[root]);
    void* receive = destination + offsets[root];
    NCCL_CHECK(ncclBroadcast(send, receive, sizes[root], ncclUint8,
      root, communicator, stream));
  }
  NCCL_CHECK(ncclGroupEnd());
}

template<typename Byte>
inline void ncclAllToAllV(const Byte* source, Byte* destination,
  const std::vector<size_t>& sendSizes, const std::vector<size_t>& receiveSizes,
  const std::vector<size_t>& sendOffsets, const std::vector<size_t>& receiveOffsets,
  const int rank, const int world, ncclComm_t communicator, cudaStream_t stream) {
  static_assert(sizeof(Byte) == 1);
  if (sendSizes[rank] > 0) {
    CHECK_CUDA(cudaMemcpyAsync(destination + receiveOffsets[rank],
      source + sendOffsets[rank], sendSizes[rank], cudaMemcpyDeviceToDevice, stream));
  }
  NCCL_CHECK(ncclGroupStart());
  for (int peer = 0; peer < world; ++peer) {
    if (peer == rank) continue;
    if (sendSizes[peer] > 0) {
      NCCL_CHECK(ncclSend(source + sendOffsets[peer], sendSizes[peer], ncclUint8,
        peer, communicator, stream));
    }
    if (receiveSizes[peer] > 0) {
      NCCL_CHECK(ncclRecv(destination + receiveOffsets[peer], receiveSizes[peer], ncclUint8,
        peer, communicator, stream));
    }
  }
  NCCL_CHECK(ncclGroupEnd());
}

template<typename Element>
inline void ncclReduceScatterV(const Element* source, Element* destination,
  const std::vector<size_t>& sizes, const std::vector<size_t>& offsets,
  const int rank, const int world, ncclComm_t communicator, cudaStream_t stream) {
  (void)rank;
  NCCL_CHECK(ncclGroupStart());
  for (int root = 0; root < world; ++root) {
    const size_t elements = sizes[root] / sizeof(Element);
    if (elements == 0) continue;
    const Element* send = reinterpret_cast<const Element*>(
      reinterpret_cast<const std::byte*>(source) + offsets[root]);
    // NCCL ignores recvbuff on non-root ranks.
    NCCL_CHECK(ncclReduce(send, destination, elements, ncclDataType<Element>(), ncclSum,
      root, communicator, stream));
  }
  NCCL_CHECK(ncclGroupEnd());
}

} // namespace bench

#endif
