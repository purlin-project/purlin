//
// Created by osayamen on 5/31/26.
//

#ifndef PURLIN_CORE_CUH
#define PURLIN_CORE_CUH
#include "atom.cuh"
#include "base.cuh"
#include "fascia.cuh" // 700 or default
#if defined(__CLION_IDE__) || ARCH >= 800
#include "tendon.cuh" // 800
#endif
#if defined(__CLION_IDE__) || ARCH >= 900
#include "ligament.cuh" // 900
#endif
#if defined(__CLION_IDE__) || ARCH >= 1000
#include "cortex.cuh" // 1000
#endif
#include "collective.cuh"
#include "setup.cuh"
#endif //PURLIN_CORE_CUH
