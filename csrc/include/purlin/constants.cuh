//
// Created by Osayamen on 3/20/26.
//

#ifndef PURLIN_CONSTANTS_CUH
#define PURLIN_CONSTANTS_CUH
#include <cuda/cmath>
namespace purlin {
  constexpr int WARP_SIZE = 32;
#if (ARCH >= 1000) && ((__CUDACC_VER_MAJOR__ >= 13) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 9)))
  constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
  constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif
  constexpr auto MAX_RANKS_PER_DOMAIN = 16;
  static_assert(sizeof(cuda::std::byte*) == sizeof(size_t));
  constexpr auto COLLECTIVE_STATE_BYTES = cuda::round_up(
    MAX_RANKS_PER_DOMAIN * 5 * sizeof(size_t) + 256, 128);
  constexpr auto RED_LATENCY_BOUND_THRESHOLD = 512UL * 1024UL;
  // Latency packets carry an eight-byte flag for every eight bytes of payload.
  // Four MiB therefore supports latency-regime payloads up to two MiB per rank.
  constexpr auto PACKET_BUFFER_SIZE = 4UL * 1024UL * 1024UL;
  constexpr auto MIN_CHUNK_SIZE = 1 * 1024 * 1024UL;
  static constexpr size_t STAGING_BUFFER_SIZE_ = 256 * 1024UL * 1024;
  static constexpr size_t MAX_NUM_CTAS = 64;
  constexpr auto MAX_STAGING_SIZE = 256 * 1024UL * 1024;
  constexpr auto MAX_CHUNKS = MAX_STAGING_SIZE / MIN_CHUNK_SIZE;
}
#endif //PURLIN_CONSTANTS_CUH
