//
// Created by azureuser on 1/8/26.
//

#ifndef TACT_TACT_CUH
#define TACT_TACT_CUH
// host-side APIs
#include "bootstrap.cuh" // initialize, finalize
#include "debug.cuh"
#include "memory.cuh" // malloc, calloc, free

// device-side structures
#include "descriptor.cuh"

// arch specializations
#include "sm70.cuh"
#include "sm80.cuh"
#include "sm90.cuh"
#include "sm100.cuh"
#endif //TACT_TACT_CUH