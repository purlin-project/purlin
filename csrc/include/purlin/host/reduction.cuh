#ifndef PURLIN_HOST_REDUCTION_CUH
#define PURLIN_HOST_REDUCTION_CUH
#include <stdexcept>

namespace purlin {
  // Deterministic uses rank-ordered unicast accumulation, even when multicast
  // is available. Non-deterministic permits multimem without an ordering guarantee.
  enum class ReductionMode {
    nonDeterministic = 0,
    deterministic = 1
  };

  // Bridge runtime frontends to the compile-time host dispatch policy.
  template<typename F>
  inline void dispatchReductionMode(const ReductionMode mode, F&& operation) {
    switch (mode) {
      case ReductionMode::deterministic:
        operation.template operator()<ReductionMode::deterministic>();
        return;
      case ReductionMode::nonDeterministic:
        operation.template operator()<ReductionMode::nonDeterministic>();
        return;
    }
    throw std::invalid_argument("Invalid reduction mode");
  }
}
#endif // PURLIN_HOST_REDUCTION_CUH
