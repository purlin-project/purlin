cmake_minimum_required(VERSION 3.27)
project(suture LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Language standards
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES "native")

# flags
add_library(suture_warnings INTERFACE)

# ------------------------------------------------------------------------------
# Global warning flags
# ------------------------------------------------------------------------------
target_compile_options(suture_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra
        -fno-strict-aliasing
        -Wno-unknown-pragmas
        -Wnull-dereference
        -Wnarrowing
        -Wno-switch
        -Wduplicated-branches
        -Wformat=2
        -Wno-unused-but-set-parameter
        -Wno-sign-compare
        >

        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall,-Wextra
        -Xcompiler=-fno-strict-aliasing
        -Xcompiler=-Wno-unknown-pragmas,-Wnull-dereference,-Wnarrowing
        -Xcompiler=-Wno-switch,-Wduplicated-branches,-Wformat=2
        -Xcompiler=-Wno-unused-but-set-parameter
        -Xcudafe --display_error_number
        -Xcompiler=-Wno-sign-compare
        >
)

include(CheckCompilerFlag)
find_package(CUDAToolkit REQUIRED)

set(ENV{CPM_SOURCE_CACHE} "${CMAKE_SOURCE_DIR}/cmake/cache")
include(cmake/CPM.cmake)

# ------------------------------------------------------------------------------
# Native arch parsing
# ------------------------------------------------------------------------------

if(DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE)
    string(REGEX MATCH "^[0-9]+" ARCH "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+")
    string(REGEX MATCH "^[0-9]+" ARCH "${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(FATAL_ERROR "Could not determine native CUDA architecture.")
endif()

math(EXPR GPU_ARCH "${ARCH} * 10" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU 0 Compute Capability: ${ARCH}")

set(ARCH_TAG "${ARCH}")
if(ARCH GREATER_EQUAL 90)
    string(APPEND ARCH_TAG "a")
endif()

set(ENV{CUTLASS_NVCC_ARCHS} "${ARCH_TAG}")

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        VERSION 3.3.0
)

find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)
find_package(mathdx REQUIRED COMPONENTS curanddx CONFIG)
find_package(MPI REQUIRED COMPONENTS CXX)

set(ENV{NVSHMEM_BUILD_LTOIR_LIBRARY} ON)
set(NVSHMEM_BUILD_LTOIR_LIBRARY ON)
find_package(NVSHMEM REQUIRED)

find_library(NCCL
        NAMES nccl libnccl.so
        REQUIRED
)

CPMAddPackage(
        NAME CUTLASS
        GITHUB_REPOSITORY nvidia/cutlass
        GIT_TAG v4.4.1
        DOWNLOAD_ONLY TRUE
        OPTIONS
        "CUTLASS_NVCC_ARCHS=${ARCH_TAG}"
)

set(MATX_NVTX_FLAGS ON)
CPMAddPackage(
        NAME MATX
        GITHUB_REPOSITORY NVIDIA/MatX
        GIT_TAG v1.0.0
)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Set -DCMAKE_BUILD_TYPE=Release (or Debug).")
endif()

function(suture_bare_target tgt)
    set_property(TARGET ${tgt} PROPERTY CUDA_ARCHITECTURES ${ARCH_TAG})
    target_include_directories(${tgt} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
    if(CCCL_ADDED)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(CUTLASS_ADDED)
        target_include_directories(${tgt} SYSTEM PRIVATE
                "${CUTLASS_SOURCE_DIR}/include"
        )
    endif()
    target_link_libraries(${tgt} PRIVATE
            CUDA::cudart
            MPI::MPI_CXX
            mathdx::curanddx
    )
    if(MATX_ADDED)
        target_link_libraries(${tgt} PRIVATE matx::matx)
    endif()

    target_link_libraries(${tgt} PRIVATE suture_warnings)
    target_compile_options(${tgt} PRIVATE
            # CUDA options
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-t0>
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>

            # Config-specific
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-Xptxas -v>
    )

    target_compile_definitions(${tgt} PRIVATE
            ARCH=${GPU_ARCH}
    )
endfunction()

function(suture_add_lto tgt)
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=lto_${ARCH_TAG}>
    )
endfunction()

function(suture_enable_rdc tgt)
    set_target_properties(${tgt} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# ------------------------------------------------------------------------------
# Workaround for build failure
# ------------------------------------------------------------------------------

set_target_properties(nvshmem::nvshmem_host PROPERTIES
        IMPORTED_LOCATION_RELEASE "$ENV{NVSHMEM_LIB_HOME}/libnvshmem_host.so.3"
)

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

set(P2P_THREADS "1056" CACHE STRING "Compile-time thread count for examples/p2p/p2p.cu")
set(P2P_UNROLL_FACTOR "1" CACHE STRING "Compile-time unroll factor for examples/p2p/p2p.cu")
set(P2P_PIPE_STAGES "1" CACHE STRING "Compile-time pipe stages for examples/p2p/p2p.cu")
set(P2P_ELEMENTS_PER_THREAD "4" CACHE STRING "Compile-time elements per thread for examples/p2p/p2p.cu")

# AG
add_executable(ag examples/ag/ag.cu)
suture_bare_target(ag)
suture_enable_rdc(ag)
target_link_libraries(ag PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host ${NCCL})
suture_add_lto(ag)

# Copy Engine AG
add_executable(ce_ag examples/ag/copy_engine_ag.cu)
suture_bare_target(ce_ag)
suture_enable_rdc(ce_ag)
target_link_libraries(ce_ag PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host ${NCCL})
suture_add_lto(ce_ag)

# P2P
add_executable(p2p examples/p2p/p2p.cu)
suture_bare_target(p2p)
suture_enable_rdc(p2p)
target_compile_definitions(p2p PRIVATE
        P2P_THREADS=${P2P_THREADS}
        P2P_UNROLL_FACTOR=${P2P_UNROLL_FACTOR}
        P2P_PIPE_STAGES=${P2P_PIPE_STAGES}
        P2P_ELEMENTS_PER_THREAD=${P2P_ELEMENTS_PER_THREAD}
)
target_link_libraries(p2p PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
suture_add_lto(p2p)

add_executable(ce_p2p examples/p2p/ce_p2p.cu)
suture_bare_target(ce_p2p)
suture_enable_rdc(ce_p2p)
target_link_libraries(ce_p2p PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
suture_add_lto(ce_p2p)

find_library(MSCCLPP
        NAMES mscclpp libmscclpp.so
        HINTS "$ENV{MSCCLPP_HOME}/lib"
        REQUIRED
)

add_executable(mscclpp_p2p examples/p2p/mscclpp_p2p.cu)
suture_bare_target(mscclpp_p2p)
target_link_libraries(mscclpp_p2p PRIVATE "${MSCCLPP}")
target_include_directories(mscclpp_p2p PRIVATE "$ENV{MSCCLPP_HOME}/include")

# AR
add_executable(ar examples/ar/ar.cu)
suture_bare_target(ar)
suture_enable_rdc(ar)
target_link_libraries(ar PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host ${NCCL})
suture_add_lto(ar)
