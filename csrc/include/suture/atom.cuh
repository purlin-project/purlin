//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_ATOM_CUH
#define SUTURE_ATOM_CUH
namespace suture {
  template<
    int threads,
    int AlignmentBytes = 16,
    int unrollFactor = 2,
    int pipeStages = 4, // tuned default
    int stageExtent = 4,
    typename DataType_ = void
  >
  struct Configuration {
    static constexpr int THREADS = threads;
    static_assert(THREADS > 0 && THREADS % 32 == 0);
    static constexpr int PIPE_STAGES = pipeStages;
    static constexpr int ELEMS_PER_THREAD = stageExtent;
    static constexpr int UNROLL_FACTOR = unrollFactor;
    static constexpr int ALIGNMENT_BYTES = AlignmentBytes;
    using DataType = DataType_;
  };

  template<int nArch, typename Config_>
  struct Atom {
    static_assert(nArch == 700 || nArch == 800 || nArch == 900 || nArch == 1000);
  };
}
#endif //SUTURE_ATOM_CUH