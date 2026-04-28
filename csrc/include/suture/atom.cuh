//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_ATOM_CUH
#define SUTURE_ATOM_CUH
#include "configuration.cuh"
namespace suture {
  template<int nArch, typename Config_>
  struct Atom {
    static_assert(nArch == 700 || nArch == 800 || nArch == 900 || nArch == 1000);
  };
}
#endif //SUTURE_ATOM_CUH