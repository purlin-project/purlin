//
// Created by Osayamen on 4/26/26.
//

#ifndef SUTURE_CONFIGURATION_CUH
#define SUTURE_CONFIGURATION_CUH
#include "constants.cuh"
namespace suture {
  enum class Regime {
    latency,
    throughput
  };
  static constexpr int UNUSED = 1;
  static constexpr int AUTO = -1;

  template<
    Regime regime,
    int threads,
    int AlignmentBytes,
    int pipeStages,
    int stageExtent,
    int unrollFactor,
    int worldUnroll = AUTO,
    int gmemAccessAlignment = MAX_ACCESS_ALIGNMENT
  >
  struct Configuration {
    static constexpr int THREADS = threads;
    static constexpr Regime REGIME = regime;
    static constexpr int PIPE_STAGES = pipeStages;
    static constexpr int ELEMS_PER_THREAD = stageExtent;
    static constexpr int UNROLL_FACTOR = unrollFactor == AUTO ? 2 : unrollFactor;
    static constexpr int ALIGNMENT_BYTES = AlignmentBytes == AUTO ? 16 : AlignmentBytes;
    static constexpr int WORLD_UNROLL = worldUnroll == AUTO ? 2 : worldUnroll;
    static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = gmemAccessAlignment;

    // assertions
    static_assert(UNROLL_FACTOR > 0);
    static_assert(WORLD_UNROLL > 0);
    static_assert(PIPE_STAGES > 0);
    static_assert(ELEMS_PER_THREAD > 0);
    static_assert(THREADS > 0 && THREADS % WARP_SIZE == 0);
    static_assert(cuda::is_power_of_two(GMEM_ACCESS_ALIGNMENT_BYTES) && cuda::is_power_of_two(ALIGNMENT_BYTES));
    static_assert(GMEM_ACCESS_ALIGNMENT_BYTES >= ALIGNMENT_BYTES);
  };
}
#endif //SUTURE_CONFIGURATION_CUH
