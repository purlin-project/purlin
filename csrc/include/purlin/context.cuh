//
// Created by Osayamen on 4/16/26.
//

#ifndef PURLIN_CONTEXT_CUH
#define PURLIN_CONTEXT_CUH
#include <cuda/cmath>
namespace purlin {
  struct VState {
    size_t maxOutBytes = 0; // max size across all output splits
    size_t maxBytes = 0; // max size across all sizes
    size_t totalBytes = 0; // sum of all input sizes
    size_t totalOutBytes = 0;
    size_t offset = 0; // this rank's offset
    size_t bytes = 0; // this rank's size
  };
  struct Context {
    cuda::std::byte** stagingLR = nullptr; // [2, world, LAT_THRESHOLD], symmetric,
    cuda::std::byte** staging = nullptr; // [2, stagingTRSize], symmetric
    cuda::std::byte* mcStagingTR = nullptr; // multicast (NVLS) mapping of staging, or null
    cuda::std::byte* mcStagingLR = nullptr; // multicast (NVLS) mapping of stagingLR, or null
    uint64_t** signals = nullptr; // [world], symmetric
    uint64_t** gatherSignals = nullptr; // [world], symmetric
    // Ring-staging backpressure: entry r on rank p carries the last chunk flag
    // rank r has drained from p's staging.
    uint64_t** consumedSignals = nullptr; // [world], symmetric
    uint64_t* epochs = nullptr; // [MAX_NUM_CTAS]
    uint32_t* putCounter = nullptr; // [world, maxChunks]
    uint32_t* redCounter = nullptr; // [world, maxChunks]
    uint32_t* consumedCounter = nullptr; // [world, maxChunks]
    size_t stagingTRSize = 0;
    /*state for variable length collectives*/
    LRP** varLenSignals = nullptr; // [2, world]
    LRP** varOffsetSignals = nullptr; // [2, world]
    size_t* sizes = nullptr;
    VState vState;
    /**************************************/
    cuda::fast_mod_div<int, true> world{2}; // must be > 1
    cuda::fast_mod_div<int> actualWorld{1};
    cuda::fast_mod_div<size_t, true> world_l{2}; // API compatibility
    cuda::fast_mod_div<long int> stagingBlocks{1};
    // Chunk slots per ring window; set by the host for ring-staging launches.
    cuda::fast_mod_div<int> ringSlots{1};
    int rank = 0;
    static_assert(cuda::std::is_trivially_copyable_v<cuda::fast_mod_div<int>>);
  };

  struct WorkspaceMemory {
    cuda::std::byte** stagingLR; // [2, world, PACKET_BUFFER_SIZE]
    cuda::std::byte** stagingTR; // [2, stagingTRSize]
    uint64_t** signals; // [world]
    uint64_t** gatherSignals; // [world]
    uint64_t** consumedSignals; // [world]
    LRP** varLenSignals; // [2, world]
    LRP** varOffsetSignals; // [2, world]
    cuda::std::byte* mcStagingTR = nullptr; // multicast (NVLS) mapping of stagingTR, optional
    cuda::std::byte* mcStagingLR = nullptr; // multicast (NVLS) mapping of stagingLR, optional
  };
}
#endif //PURLIN_CONTEXT_CUH
