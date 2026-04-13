//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_ATOM_CUH
#define SUTURE_ATOM_CUH
#include "constants.cuh"
namespace suture {
  template<
    int threads,
    int AlignmentBytes = 16,
    int unrollFactor = 2,
    int pipeStages = 4, // tuned default
    int stageExtent = 4,
    int stageBytes = WARP_SIZE * AlignmentBytes * stageExtent
  >
  struct Configuration {
    static constexpr int THREADS = threads;
    static_assert(THREADS > 0 && THREADS % 32 == 0);
    static constexpr int PIPE_STAGES = pipeStages;
    static constexpr int ELEMS_PER_THREAD = stageExtent;
    static constexpr int UNROLL_FACTOR = unrollFactor;
    static_assert(UNROLL_FACTOR > 0);
    static constexpr int ALIGNMENT_BYTES = AlignmentBytes;
    static constexpr int STAGE_BYTES = stageBytes;
  };

  template<int nArch, typename Config_>
  struct Atom {
    static_assert(nArch == 700 || nArch == 800 || nArch == 900 || nArch == 1000);
  };
}
#endif //SUTURE_ATOM_CUH