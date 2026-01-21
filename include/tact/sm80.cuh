//
// Created by azureuser on 1/8/26.
//

#ifndef TACT_SM80_CUH
#define TACT_SM80_CUH
#include "descriptor.cuh"

template<
  tact::DataType dataType,
  int BlockThreads,
  tact::Regime regime
>
struct tact::TransferDescriptor<80, dataType, BlockThreads, regime> {

};
#endif //TACT_SM80_CUH