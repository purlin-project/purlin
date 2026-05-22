//
// Created by Osayamen on 3/20/26.
//

#ifndef SUTURE_CONSTANTS_CUH
#define SUTURE_CONSTANTS_CUH
namespace suture {
  constexpr int WARP_SIZE = 32;
#if (ARCH >= 1000) && (__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 9)
  constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
  constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif
  constexpr auto MAX_RANKS_PER_DOMAIN = 16;
  constexpr auto COLLECTIVE_STATE_BYTES = 3 * MAX_RANKS_PER_DOMAIN * sizeof(cuda::std::byte*);
  constexpr auto RED_LATENCY_BOUND_THRESHOLD = 512UL * 1024UL;
  constexpr auto AG_LATENCY_BOUND_THRESHOLD = 512UL * 1024UL;
  // *2 to include flags
  constexpr auto PACKET_BUFFER_SIZE = 2 * RED_LATENCY_BOUND_THRESHOLD;
  constexpr auto MIN_CHUNK_SIZE = 1 * 1024 * 1024UL;
  static constexpr size_t STAGING_BUFFER_SIZE_ = 512 * 1024UL * 1024;
  static constexpr size_t MAX_NUM_CTAS = 256;
  constexpr auto MAX_CHUNKS = suture::STAGING_BUFFER_SIZE_ / MIN_CHUNK_SIZE;
  constexpr auto CHUNKED_PUT_BLOCKS = 16;
  constexpr auto NON_CHUNKED_PUT_BLOCKS = 32;

  enum class UseMulticast {
    yes,
    no
  };
}
#endif //SUTURE_CONSTANTS_CUH
