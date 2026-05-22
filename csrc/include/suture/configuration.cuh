//
// Created by azureuser on 4/26/26.
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
  struct ThreadsHeuristic {
    static_assert(nArch >= 900);
    static constexpr int value = 288;
  };
  template<>
  struct ThreadsHeuristic<700> {
    static constexpr int value = 128;
  };
  template<>
  struct ThreadsHeuristic<800> {
    static constexpr int value = 128;
  };

  template<int nArch>
  struct PipeStagesHeuristic {
    static_assert(nArch >= 900);
    static constexpr int value = 1;
  };
  template<>
  struct PipeStagesHeuristic<800> {
    static constexpr int value = 8;
  };
  template<>
  struct PipeStagesHeuristic<700> {
    static constexpr int value = UNUSED;
  };

  // elements per thread
  template<int nArch>
  struct StageExtentHeuristic {
    static_assert(nArch >= 900);
    static constexpr int value = 8;
  };

  template<>
  struct StageExtentHeuristic<800> {
    static constexpr int value = 4;
  };

  template<>
  struct StageExtentHeuristic<700> {
    static constexpr int value = UNUSED;
  };

  template<int nArch>
  struct UnrollFactorHeuristic {
    static_assert(nArch >= 800);
    static constexpr int value = 2;
  };

  template<>
  struct UnrollFactorHeuristic<700> {
    static constexpr int value = 4; // TODO: tune this
  };

  template<int nArch>
  struct GmemAlignmentHeuristic {
    static_assert(nArch < 1000);
    static constexpr int value =  16;
  };

  template<>
  struct GmemAlignmentHeuristic<1000> {
    static constexpr int value = 32;
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
    UseMulticast um = UseMulticast::no,
    int gmemAccessAlignment = AUTO
  >
  struct Configuration {
    static constexpr int Arch = nArch;
    static constexpr int THREADS = threads == AUTO ? ThreadsHeuristic<nArch>::value : threads;
    static constexpr Regime REGIME = regime;
    static constexpr int PIPE_STAGES = pipeStages == AUTO ? PipeStagesHeuristic<nArch>::value : pipeStages;
    static constexpr int ELEMS_PER_THREAD = stageExtent == AUTO ? StageExtentHeuristic<nArch>::value : stageExtent;
    static constexpr int UNROLL_FACTOR = unrollFactor == AUTO ? UnrollFactorHeuristic<nArch>::value : unrollFactor;
    static constexpr int ALIGNMENT_BYTES = AlignmentBytes == AUTO ? 16 : AlignmentBytes;
    static constexpr int WORLD_UNROLL = worldUnroll == AUTO ? 4 : worldUnroll;
    static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = gmemAccessAlignment == AUTO ?
    GmemAlignmentHeuristic<nArch>::value : gmemAccessAlignment;
    static constexpr UseMulticast USE_MULTICAST = um;

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
