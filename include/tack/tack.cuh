//
// Created by azureuser on 1/8/26.
//

#ifndef tack_tack_CUH
#define tack_tack_CUH
// host-side APIs
#include "bootstrap.cuh" // initialize, finalize
#include "../../examples/debug.cuh"
#include "memory.cuh" // malloc, calloc, free

// device-side structures
#include "descriptor.cuh"

// arch specializations
#include "sm70.cuh"
#include "sm80.cuh"
#include "sm90.cuh"
#include "sm100.cuh"
#endif //tack_tack_CUH