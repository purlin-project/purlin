//
// Created by Osayamen on 4/26/26.
//

#ifndef PURLIN_CONFIGURATION_CUH
#define PURLIN_CONFIGURATION_CUH
#include <cuda/cmath>
#include "constants.cuh"
namespace purlin {
  enum class Regime {
    latency,
    throughput
  };
  enum class Datapath {
    unicast,
    multimem
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
    int gmemAccessAlignment = MAX_ACCESS_ALIGNMENT,
    Datapath datapath = Datapath::unicast,
    int mmDepth = AUTO
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
    static constexpr Datapath DATAPATH = datapath;
    // Register-pipeline depth of the multimem reduce: outstanding ld_reduce per
    // thread. AUTO matches the smem pipeline's steady state per thread.
    static constexpr int MM_DEPTH = mmDepth == AUTO ? PIPE_STAGES * ELEMS_PER_THREAD : mmDepth;

    // assertions
    static_assert(UNROLL_FACTOR > 0);
    static_assert(WORLD_UNROLL > 0);
    static_assert(PIPE_STAGES > 0);
    static_assert(ELEMS_PER_THREAD > 0);
    static_assert(MM_DEPTH > 0);
    static_assert(THREADS > 0 && THREADS % WARP_SIZE == 0);
    static_assert(cuda::is_power_of_two(GMEM_ACCESS_ALIGNMENT_BYTES) && cuda::is_power_of_two(ALIGNMENT_BYTES));
    static_assert(GMEM_ACCESS_ALIGNMENT_BYTES >= ALIGNMENT_BYTES);
  };

  // Rebind a configuration to the multimem datapath, preserving everything else;
  // an explicit mmDepth (e.g. a tuning-policy value) overrides the derived depth.
  template<typename C, int mmDepth = AUTO>
  using WithMultimem = Configuration<
    C::REGIME,
    C::THREADS,
    C::ALIGNMENT_BYTES,
    C::PIPE_STAGES,
    C::ELEMS_PER_THREAD,
    C::UNROLL_FACTOR,
    C::WORLD_UNROLL,
    C::GMEM_ACCESS_ALIGNMENT_BYTES,
    Datapath::multimem,
    mmDepth == AUTO ? C::MM_DEPTH : mmDepth
  >;
}
#endif //PURLIN_CONFIGURATION_CUH
