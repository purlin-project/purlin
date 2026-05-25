//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_CONTEXT_CUH
#define SUTURE_CONTEXT_CUH
#include <cuda/cmath>
namespace suture {
  struct Context {
    cuda::std::byte** stagingLR = nullptr; // [2, world, LAT_THRESHOLD], symmetric,
    cuda::std::byte** staging = nullptr; // [2, STAGING_BUFFER_SIZE_], symmetric
    uint64_t** signals = nullptr; // [world], symmetric
    uint64_t** gatherSignals = nullptr; // [world], symmetric
    uint64_t* epochs = nullptr; // [MAX_NUM_CTAS]
    uint32_t* putCounter = nullptr; // [world, maxChunks]
    uint32_t* redCounter = nullptr; // [world, maxChunks]
    cuda::fast_mod_div<int, true> world{2}; // must be > 1
    cuda::fast_mod_div<int> actualWorld{1};
    cuda::fast_mod_div<size_t, true> world_l{2}; // API compatibility
    cuda::fast_mod_div<long int> stagingBlocks{1};
    int rank = 0;
    static_assert(cuda::std::is_trivially_copyable_v<cuda::fast_mod_div<int>>);
  };
}
#endif //SUTURE_CONTEXT_CUH