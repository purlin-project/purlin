cmake_minimum_required(VERSION 3.27)
project(purlin LANGUAGES CXX CUDA)

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
add_library(purlin_warnings INTERFACE)

# ------------------------------------------------------------------------------
# Global warning flags
# ------------------------------------------------------------------------------
target_compile_options(purlin_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra
        -fno-strict-aliasing
        -Wno-unknown-pragmas
        -Wnull-dereference
        -Wnarrowing
        -Wno-switch
        -Wformat=2
        -Wno-unused-but-set-parameter
        >

        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall,-Wextra
        -Xcompiler=-fno-strict-aliasing
        -Xcompiler=-Wno-unknown-pragmas,-Wnull-dereference,-Wnarrowing
        -Xcompiler=-Wno-switch,-Wformat=2
        -Xcompiler=-Wno-unused-but-set-parameter
        >
)

include(CheckCompilerFlag)
find_package(CUDAToolkit REQUIRED)

set(ENV{CPM_SOURCE_CACHE} "${CMAKE_SOURCE_DIR}/cmake/cache")
include(cmake/CPM.cmake)

set(PURLIN_TUNE_DEFINITIONS "" CACHE STRING
    "Semicolon-separated compile definitions used by the B300 host-API tuning harness")

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

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        VERSION 3.3.0
)
find_package(MPI REQUIRED COMPONENTS CXX)

set(ENV{NVSHMEM_BUILD_LTOIR_LIBRARY} ON)
set(NVSHMEM_BUILD_LTOIR_LIBRARY ON)
find_package(NVSHMEM REQUIRED)

set(MATX_NVTX_FLAGS ON)
CPMAddPackage(
        NAME MATX
        GITHUB_REPOSITORY NVIDIA/MatX
        GIT_TAG v1.0.0
)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Set -DCMAKE_BUILD_TYPE=Release (or Debug).")
endif()

add_library(purlin_benchmark_support INTERFACE)
target_include_directories(purlin_benchmark_support INTERFACE
        "${CMAKE_CURRENT_LIST_DIR}/support"
)
target_link_libraries(purlin_benchmark_support INTERFACE
        CUDA::cudart
        MPI::MPI_CXX
        matx::matx
        purlin_warnings
)

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.5.0-c-cpp
        GIT_SHALLOW TRUE
)

function(purlin_bare_target tgt)
    set_property(TARGET ${tgt} PROPERTY CUDA_ARCHITECTURES ${ARCH_TAG})
    target_include_directories(${tgt} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
    if(TARGET nvtx3-cpp)
        target_link_libraries(${tgt} INTERFACE nvtx3-cpp)
    endif ()
    if(CCCL_ADDED)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    target_link_libraries(${tgt} PRIVATE
            CUDA::cudart
            MPI::MPI_CXX
            purlin_benchmark_support
    )
    if(MATX_ADDED)
        target_link_libraries(${tgt} PRIVATE matx::matx)
    endif()

    target_link_libraries(${tgt} PRIVATE purlin_warnings atomic)
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
            PURLIN_NVTX=1
    )
endfunction()

function(purlin_add_lto tgt)
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=lto_${ARCH_TAG}>
    )
endfunction()

function(purlin_enable_rdc tgt)
    set_target_properties(${tgt} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

function(collective_benchmark_target tgt)
    set_property(TARGET ${tgt} PROPERTY CUDA_ARCHITECTURES ${ARCH_TAG})
    target_link_libraries(${tgt} PRIVATE
            purlin_benchmark_support
    )
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-Xptxas -v>
    )
endfunction()

function(purlin_benchmark_target tgt)
    collective_benchmark_target(${tgt})
    target_include_directories(${tgt} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
    target_link_libraries(${tgt} PRIVATE atomic)
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-t0>
    )
    target_compile_definitions(${tgt} PRIVATE
            ARCH=${GPU_ARCH}
            PURLIN_NVTX=1
    )
    if(PURLIN_TUNE_DEFINITIONS)
        target_compile_definitions(${tgt} PRIVATE ${PURLIN_TUNE_DEFINITIONS})
    endif()
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

# Host-only split-policy checks; running this target needs no GPUs or MPI launch.
add_executable(testVariableCounts tests/variable_counts.cpp)
target_include_directories(testVariableCounts PRIVATE "${CMAKE_CURRENT_LIST_DIR}/support")
target_link_libraries(testVariableCounts PRIVATE CUDA::cudart MPI::MPI_CXX)

# AG
add_executable(ag examples/ag/ag.cu)
purlin_bare_target(ag)
purlin_enable_rdc(ag)
target_link_libraries(ag PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(ag)

add_executable(testAG tests/ag.cu)
purlin_benchmark_target(testAG)
purlin_enable_rdc(testAG)
target_link_libraries(testAG PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testAG)

add_executable(testAGV tests/agv.cu)
purlin_benchmark_target(testAGV)
purlin_enable_rdc(testAGV)
target_link_libraries(testAGV PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testAGV)

add_executable(testA2A tests/a2a.cu)
purlin_benchmark_target(testA2A)
purlin_enable_rdc(testA2A)
target_link_libraries(testA2A PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testA2A)

add_executable(testA2AV tests/a2av.cu)
purlin_benchmark_target(testA2AV)
purlin_enable_rdc(testA2AV)
target_link_libraries(testA2AV PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testA2AV)

add_executable(testAGVSKEW tests/agvskew.cu)
purlin_benchmark_target(testAGVSKEW)
purlin_enable_rdc(testAGVSKEW)
target_link_libraries(testAGVSKEW PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testAGVSKEW)

add_executable(testRSVSKEW tests/rsvskew.cu)
purlin_benchmark_target(testRSVSKEW)
purlin_enable_rdc(testRSVSKEW)
target_link_libraries(testRSVSKEW PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testRSVSKEW)

add_executable(testA2AVSKEW tests/a2avskew.cu)
purlin_benchmark_target(testA2AVSKEW)
purlin_enable_rdc(testA2AVSKEW)
target_link_libraries(testA2AVSKEW PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testA2AVSKEW)

add_executable(a2a examples/a2a/a2a.cu)
purlin_bare_target(a2a)
purlin_enable_rdc(a2a)
target_link_libraries(a2a PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(a2a)

# Copy Engine AG
add_executable(ce_ag examples/ag/copy_engine_ag.cu)
purlin_bare_target(ce_ag)
purlin_enable_rdc(ce_ag)
target_link_libraries(ce_ag PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(ce_ag)

# P2P
add_executable(push examples/p2p/push.cu)
purlin_bare_target(push)
purlin_enable_rdc(push)
target_link_libraries(push PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(push)

add_executable(pull examples/p2p/pull.cu)
purlin_bare_target(pull)
purlin_enable_rdc(pull)
target_link_libraries(pull PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(pull)

add_executable(ce_p2p examples/p2p/ce_p2p.cu)
purlin_bare_target(ce_p2p)
purlin_enable_rdc(ce_p2p)
target_link_libraries(ce_p2p PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(ce_p2p)

# AR
add_executable(ar examples/ar/ar.cu)
purlin_bare_target(ar)
purlin_enable_rdc(ar)
target_link_libraries(ar PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(ar)

add_executable(testAR tests/ar.cu)
purlin_benchmark_target(testAR)
purlin_enable_rdc(testAR)
target_link_libraries(testAR PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testAR)

add_executable(rs examples/rs/rs.cu)
purlin_bare_target(rs)
purlin_enable_rdc(rs)
target_link_libraries(rs PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(rs)

add_executable(testRS tests/rs.cu)
purlin_benchmark_target(testRS)
purlin_enable_rdc(testRS)
target_link_libraries(testRS PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testRS)

add_executable(testRSV tests/rsv.cu)
purlin_benchmark_target(testRSV)
purlin_enable_rdc(testRSV)
target_link_libraries(testRSV PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testRSV)

add_executable(testVarlenEmpty tests/varlen_empty.cu)
purlin_benchmark_target(testVarlenEmpty)
purlin_enable_rdc(testVarlenEmpty)
target_link_libraries(testVarlenEmpty PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
purlin_add_lto(testVarlenEmpty)
