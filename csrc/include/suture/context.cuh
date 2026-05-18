//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_CONTEXT_CUH
#define SUTURE_CONTEXT_CUH
#include <cuda/cmath>
namespace suture {
  // for All-Reduce
  static constexpr size_t STAGING_BUFFER_SIZE_ = 512 * 1024UL * 1024;
  static constexpr size_t MAX_SUPER_BLOCK_SIZE_ = 64UL;
  static constexpr size_t MAX_NUM_CTAS = 256;
  struct Context {
    uint64_t** signals = nullptr; // [world], symmetric
    uint64_t* epochs = nullptr; // [MAX_NUM_CTAS]
    uint32_t* putCounter = nullptr; // [world]
    uint32_t* groupSense = nullptr; // [world]
    cuda::std::byte** stagingLR = nullptr; // [2, world, LAT_THRESHOLD], symmetric,
    cuda::std::byte** staging = nullptr; // [2, STAGING_BUFFER_SIZE_], symmetric
    cuda::fast_mod_div<int, true> world{2}; // must be > 1
    cuda::fast_mod_div<long int> superBlockSize{1};
    int rank = 0;
    static_assert(cuda::std::is_trivially_copyable_v<cuda::fast_mod_div<int>>);
  };
}
#endif //SUTURE_CONTEXT_CUH