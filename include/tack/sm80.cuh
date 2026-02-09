//
// Created by azureuser on 1/8/26.
//

#ifndef tack_SM80_CUH
#define tack_SM80_CUH
#include "descriptor.cuh"
namespace tack
{
  // GMEM -> GMEM
  template<
   int threads, // we could make this dynamic
   long int size,
   Regime regime
  >
  struct TransferDescriptor<80, StateSpace::GMEM, threads, size, regime> {

  };
}
#endif //tack_SM80_CUH