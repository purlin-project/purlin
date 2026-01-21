//
// Created by azureuser on 1/8/26.
//

#ifndef TACT_SM100_CUH
#define TACT_SM100_CUH
#include "descriptor.cuh"

template<
  tact::DataType dataType,
  int BlockThreads,
  tact::Regime regime
>
struct tact::TransferDescriptor<100, dataType, BlockThreads, regime> {

};
#endif //TACT_SM100_CUH