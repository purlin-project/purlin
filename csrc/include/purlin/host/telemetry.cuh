//
// Created by osayamen on 5/28/26.
//

#ifndef PURLIN_TELEMETRY_CUH
#define PURLIN_TELEMETRY_CUH
#include <nvtx3/nvtx3.hpp>
namespace purlin {
  struct purlinDomain {
    static constexpr auto const* name{"Purlin"};
  };
  using PurlinRange = nvtx3::scoped_range_in<purlinDomain>;
}
#endif //PURLIN_TELEMETRY_CUH
