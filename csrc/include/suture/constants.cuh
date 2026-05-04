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
  constexpr auto AG_SUPER_BLOCK_THRESHOLD = 2UL * 1024UL * 1024UL;
  constexpr auto P2P_SUPER_BLOCK_THRESHOLD = 1UL * 1024UL * 1024UL;
  constexpr auto AR_SUPER_BLOCK_THRESHOLD = 1UL * 1024UL * 1024UL;
  constexpr auto RED_LATENCY_BOUND_THRESHOLD = 256UL * 1024UL;
  // *2 to include flags
  constexpr auto PACKET_BUFFER_SIZE = 2 * RED_LATENCY_BOUND_THRESHOLD;
  constexpr auto RED_CHUNK_SIZE = 16 * 1024UL * 1024UL;
  constexpr auto RED_PUT_BLOCKS = 32;
}
#endif //SUTURE_CONSTANTS_CUH