//
// Created by osayamen on 5/31/26.
//

#ifndef SUTURE_CORE_CUH
#define SUTURE_CORE_CUH
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
#endif //SUTURE_CORE_CUH
