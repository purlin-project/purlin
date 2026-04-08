//
// Created by Osayamen on 4/7/26.
//

#ifndef SUTURE_ATOM_CUH
#define SUTURE_ATOM_CUH
#include "base.cuh"
namespace suture {
  template<int nArch, StateSpace sourceSpace = StateSpace::GMEM>
  struct Atom {};
}
#endif //SUTURE_ATOM_CUH