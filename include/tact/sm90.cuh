//
// Created by azureuser on 1/8/26.
//

#ifndef TACT_SM90_CUH
#define TACT_SM90_CUH
#include "descriptor.cuh"

template<
  tact::DataType dataType,
  int BlockThreads,
  tact::Regime regime
>
struct tact::TransferDescriptor<90, dataType, BlockThreads, regime> {

};
#endif //TACT_SM90_CUH