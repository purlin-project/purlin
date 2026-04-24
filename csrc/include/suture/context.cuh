//
// Created by Osayamen on 4/16/26.
//

#ifndef SUTURE_CONTEXT_CUH
#define SUTURE_CONTEXT_CUH
#include <cuda/cmath>
#include <cuda/atomic>
#include <cuda/barrier>
namespace suture {
  struct Barrier {
    uint8_t* localSense = nullptr; //[maxNumParticipants]
    uint8_t* globalSense = nullptr;
    uint32_t* counter = nullptr;
    void arrive_and_wait(const int& participant, const int& expectedCount) const {
      if (expectedCount <= 1) {
        return;
      }
      const auto nextParity = static_cast<uint8_t>(localSense[participant] == 0 ? 1 : 0);
      localSense[participant] = nextParity;
      const cuda::atomic_ref<uint32_t, cuda::thread_scope_device> c{*counter};
      const auto count = c.fetch_add(1, cuda::memory_order_acq_rel) + 1;
      const cuda::atomic_ref<uint8_t, cuda::thread_scope_device> gS{*globalSense};
      if (count == expectedCount) {
        // reset counter
        c.store(0, cuda::memory_order_relaxed);
        // set global sense
        gS.store(nextParity, cuda::memory_order_release);
        return;
      }
      auto isComplete = gS.load(cuda::memory_order_relaxed) == nextParity;
      while (!isComplete) {
        isComplete = gS.load(cuda::memory_order_relaxed) == nextParity;
      }
      cuda::std::ignore = gS.load(cuda::memory_order_acquire);
    }
  };
  // for All-Reduce
  static constexpr size_t MAX_ALL_REDUCE_SIZE_ = 1024 * 1024UL * 1024;
  static constexpr size_t MAX_SUPER_BLOCK_SIZE_ = 64UL;
  struct SutureContext {
    uint32_t* signals = nullptr; // [world]
    uint64_t* sync0 = nullptr; // [world, maxSuperBlockSize], symmetric
    uint64_t* sync1 = nullptr; // [world, maxSuperBlockSize], symmetric
    uint8_t* senseBitsTR = nullptr; // [world, maxSuperBlockSize], non-symmetric
    uint8_t* senseBitsLR = nullptr; // [world, maxSuperBlockSize], non-symmetric
    cuda::std::byte* staging = nullptr; // [2, world, LAT_THRESHOLD], symmetric,
    cuda::std::byte* reduceBuffer = nullptr; // [world, MAX_ALL_RED]
    uint8_t* flagPutSense = nullptr; // [2, world, LAT_THRESHOLD / (sizeof(LRP.data)]
    uint8_t* flagRedSense = nullptr; // same size as above
    uint* sigCounter = nullptr; // [1]
    size_t maxSuperBlockSize = MAX_SUPER_BLOCK_SIZE_;
    size_t maxARSize = MAX_ALL_REDUCE_SIZE_;
    cuda::fast_mod_div<int> world{1};
    cuda::fast_mod_div<int> actualWorld{1};
    cuda::fast_mod_div<int> superBlockSize{1};
    int rank = 0;
    static_assert(cuda::std::is_trivially_copyable_v<cuda::fast_mod_div<int>>);

    __host__ __forceinline__
    void setSuperBlockSize(const int& superBlockSize_) {
      superBlockSize = cuda::fast_mod_div<int>{superBlockSize_};
    }
  };
}
#endif //SUTURE_CONTEXT_CUH