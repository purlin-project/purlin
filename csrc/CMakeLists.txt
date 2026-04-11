cmake_minimum_required(VERSION 3.27)
project(suture CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(LIB_FLAGS "-Wall -Wextra -Wno-unknown-pragmas -Wnull-dereference -Wnarrowing -Wno-unused-but-set-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIB_FLAGS}")

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number")
set(CMAKE_CUDA_FLAGS "-Xfatbin -compress-all -Xcudafe --display_error_number -Xcompiler \"${LIB_FLAGS}\"")

include(CheckCompilerFlag)
find_package(CUDAToolkit REQUIRED)
set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")
include(cmake/CPM.cmake)
string(REGEX MATCH "^[0-9]+" ARCH "${CMAKE_CUDA_ARCHITECTURES_NATIVE}") # xx-real -> xx
math(EXPR GPU_ARCH "${ARCH} * 10" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU 0 Compute Capability: ${ARCH}")

set(ARCH_TAG "${ARCH}")
if(ARCH GREATER_EQUAL 90)
    string(APPEND ARCH_TAG "a") # accelerated arch-specific instruction set
endif()
set(ENV{CUTLASS_NVCC_ARCHS} "${ARCH_TAG}")

CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        VERSION 3.1.2
)
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)
find_package(mathdx REQUIRED COMPONENTS curanddx CONFIG)
find_package(MPI REQUIRED COMPONENTS CXX)
set(ENV{NVSHMEM_BUILD_LTOIR_LIBRARY} ON)
set(NVSHMEM_BUILD_LTOIR_LIBRARY ON)
find_package(NVSHMEM REQUIRED)
find_library(NCCL
        NAMES libnccl.so
        REQUIRED
)

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.1-c-cpp
        GIT_SHALLOW TRUE
)

#CUTLASS
CPMAddPackage(
        NAME CUTLASS
        GITHUB_REPOSITORY nvidia/cutlass
        GIT_TAG main
        DOWNLOAD_ONLY TRUE
        OPTIONS
        "CUTLASS_NVCC_ARCHS=${ARCH_TAG}"
)

set(MATX_NVTX_FLAGS ON)
#set(MATX_ENABLE_FILEIO ON)
CPMAddPackage(
        NAME MATX
        GITHUB_REPOSITORY NVIDIA/MatX
        GIT_TAG v0.9.4
)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Set -DCMAKE_BUILD_TYPE=Release (or Debug).")
endif()

function(suture_bare_target tgt)
    set_property(TARGET ${tgt} PROPERTY CUDA_ARCHITECTURES ${ARCH_TAG})
    target_link_libraries(${tgt} PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
            atomic
            MPI::MPI_CXX
    )
    if(NTX3_ADDED)
        target_link_libraries(${tgt} PRIVATE nvtx3-cpp)
    endif ()
    if(MATX_ADDED)
        target_link_libraries(${tgt} PRIVATE matx::matx)
    endif ()
    if(CCCL_ADDED)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(CUTLASS_ADDED)
        target_include_directories(${tgt} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    endif ()
    target_link_libraries(${tgt} PRIVATE mathdx::curanddx)
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-t0; -lineinfo> #use sm_xxa for xx >= 90
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
    )
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
    )
    target_compile_definitions(${tgt} PRIVATE ARCH=${GPU_ARCH})
endfunction()

function(suture_add_lto tgt)
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=lto_${ARCH_TAG}>
    )
endfunction()
## WAR for build failure
set_target_properties(nvshmem::nvshmem_host PROPERTIES
        IMPORTED_LOCATION_RELEASE "$ENV{NVSHMEM_LIB_HOME}/libnvshmem_host.so.3"
)

#AG
add_executable(ag
        examples/ag/ag.cu
)
suture_bare_target(ag)
set_target_properties(ag PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(ag PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host ${NCCL})
suture_add_lto(ag)

#Copy Engine AG
add_executable(ce_ag
        examples/ag/copy_engine_ag.cu
)
suture_bare_target(ce_ag)
set_target_properties(ce_ag PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(ce_ag PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host ${NCCL})
suture_add_lto(ce_ag)

add_executable(p2p
        examples/p2p/p2p.cu
)
suture_bare_target(p2p)
set_target_properties(p2p PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(p2p PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
suture_add_lto(p2p)

add_executable(ce_p2p
        examples/p2p/ce_p2p.cu
)
suture_bare_target(ce_p2p)
set_target_properties(ce_p2p PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(ce_p2p PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host)
suture_add_lto(ce_p2p)

find_library(MSCCLPP
        NAMES libmscclpp.so
        HINTS "$ENV{MSCCLPP_HOME}/lib"
        REQUIRED
)
add_executable(mscclpp_p2p
        examples/p2p/mscclpp_p2p.cu
)
suture_bare_target(mscclpp_p2p)
target_link_libraries(mscclpp_p2p PRIVATE "${MSCCLPP}")
target_include_directories(mscclpp_p2p PRIVATE "$ENV{MSCCLPP_HOME}/include")

#AR
add_executable(ar
        examples/ar/atomic_ar.cu
)
suture_bare_target(ar)
set_target_properties(ar PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(ar PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host ${NCCL})
suture_add_lto(ar)

add_executable(ar2
        examples/ar/ar.cu
)
suture_bare_target(ar2)
set_target_properties(ar2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(ar2 PRIVATE nvshmem::nvshmem_device nvshmem::nvshmem_host ${NCCL})
suture_add_lto(ar2)
