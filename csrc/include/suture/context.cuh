//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_CONTEXT_CUH
#define SUTURE_CONTEXT_CUH
#include <cuda/cmath>
namespace suture {
  // for All-Reduce
  static constexpr size_t MAX_ALL_REDUCE_SIZE_ = 1024 * 1024UL * 1024;
  static constexpr size_t MAX_SUPER_BLOCK_SIZE_ = 64UL;
  static constexpr size_t MAX_NUM_CTAS = 256;
  struct SutureContext {
    uint64_t* signals = nullptr; // [world]
    uint* sigCounter = nullptr; // [world]
    uint64_t* sync = nullptr; // [world, maxSuperBlockSize], symmetric
    uint64_t* epochs = nullptr; // [MAX_NUM_CTAS]
    cuda::std::byte* staging = nullptr; // [2, world, LAT_THRESHOLD], symmetric,
    cuda::std::byte* reduceBuffer = nullptr; // [world, MAX_ALL_RED]
    size_t maxSuperBlockSize = MAX_SUPER_BLOCK_SIZE_;
    size_t maxARSize = MAX_ALL_REDUCE_SIZE_;
    cuda::fast_mod_div<int, true> world{2}; // must be > 1
    cuda::fast_mod_div<long int> superBlockSize{1};
    int rank = 0;
    int maxPutBlocks = 1;
    static_assert(cuda::std::is_trivially_copyable_v<cuda::fast_mod_div<int>>);

    __host__ __forceinline__
    void setSuperBlockSize(const int& superBlockSize_) {
      superBlockSize = cuda::fast_mod_div<long int>{superBlockSize_};
      maxPutBlocks = (world - 1) * superBlockSize_;
    }
  };
}
#endif //SUTURE_CONTEXT_CUH