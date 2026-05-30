//
// Created by Osayamen on 3/20/26.
//

#ifndef SUTURE_CONSTANTS_CUH
#define SUTURE_CONSTANTS_CUH
namespace suture {
  constexpr int WARP_SIZE = 32;
#if (ARCH >= 1000) && ((__CUDACC_VER_MAJOR__ >= 13) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 9)))
  constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
  constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif
  constexpr auto MAX_RANKS_PER_DOMAIN = 8;
  constexpr auto COLLECTIVE_STATE_BYTES = cuda::std::bit_ceil(
    static_cast<uint32_t>(3 * MAX_RANKS_PER_DOMAIN * sizeof(cuda::std::byte*)));
  constexpr auto RED_LATENCY_BOUND_THRESHOLD = 512UL * 1024UL;
  // *2 to include flags
  constexpr auto PACKET_BUFFER_SIZE = 2 * RED_LATENCY_BOUND_THRESHOLD;
  constexpr auto MIN_CHUNK_SIZE = 1 * 1024 * 1024UL;
  static constexpr size_t STAGING_BUFFER_SIZE_ = 256 * 1024UL * 1024;
  static constexpr size_t MAX_NUM_CTAS = 128;
  constexpr auto MAX_CHUNKS = suture::STAGING_BUFFER_SIZE_ / (2 * MIN_CHUNK_SIZE);
}
#endif //SUTURE_CONSTANTS_CUH
