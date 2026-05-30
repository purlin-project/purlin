//
// Created by Osayamen on 4/26/26.
//

#ifndef SUTURE_CONFIGURATION_CUH
#define SUTURE_CONFIGURATION_CUH
#include "constants.cuh"
namespace suture {
  enum class Regime {
    latency,
    throughput,
    unused
  };
  static constexpr int UNUSED = 1;
  static constexpr int AUTO = -1;

  template<int nArch>
  struct GmemAlignmentHeuristic {
    static_assert(nArch < 1000);
    static constexpr int value =  16;
  };

  template<>
  struct GmemAlignmentHeuristic<1000> {
    static constexpr int value = MAX_ACCESS_ALIGNMENT;
  };

  template<
    int nArch,
    Regime regime,
    int threads,
    int AlignmentBytes,
    int pipeStages,
    int stageExtent,
    int unrollFactor,
    int worldUnroll = AUTO,
    int gmemAccessAlignment = AUTO
  >
  struct Configuration {
    static constexpr int Arch = nArch;
    static constexpr int THREADS = threads;
    static constexpr Regime REGIME = regime;
    static constexpr int PIPE_STAGES = pipeStages;
    static constexpr int ELEMS_PER_THREAD = stageExtent;
    static constexpr int UNROLL_FACTOR = unrollFactor == AUTO ? 2 : unrollFactor;
    static constexpr int ALIGNMENT_BYTES = AlignmentBytes == AUTO ? 16 : AlignmentBytes;
    static constexpr int WORLD_UNROLL = worldUnroll == AUTO ? 2 : worldUnroll;
    static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = gmemAccessAlignment == AUTO ?
    GmemAlignmentHeuristic<nArch>::value : gmemAccessAlignment;

    // assertions
    static_assert(UNROLL_FACTOR > 0);
    static_assert(WORLD_UNROLL > 0);
    static_assert(PIPE_STAGES > 0);
    static_assert(THREADS > 0 && THREADS % WARP_SIZE == 0);
    static_assert(cuda::is_power_of_two(GMEM_ACCESS_ALIGNMENT_BYTES) && cuda::is_power_of_two(ALIGNMENT_BYTES));
    static_assert(GMEM_ACCESS_ALIGNMENT_BYTES >= ALIGNMENT_BYTES);
  };
}
#endif //SUTURE_CONFIGURATION_CUH
