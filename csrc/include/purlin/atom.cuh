//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_ATOM_CUH
#define PURLIN_ATOM_CUH
#include "configuration.cuh"
namespace purlin {
  template<int nArch, typename Config_>
  struct Atom {
    static_assert(nArch == 700 || nArch == 800 || nArch == 900 || nArch == 1000);
  };
}
#endif //PURLIN_ATOM_CUH