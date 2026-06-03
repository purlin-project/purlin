//
// Created by osayamen on 5/28/26.
//

#ifndef SUTURE_TELEMETRY_CUH
#define SUTURE_TELEMETRY_CUH
#include <nvtx3/nvtx3.hpp>
namespace suture {
  struct sutureDomain {
    static constexpr auto const* name{"Suture"};
  };
  using SutureRange = nvtx3::scoped_range_in<sutureDomain>;
}
#endif //SUTURE_TELEMETRY_CUH
