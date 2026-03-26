cmake_minimum_required(VERSION 3.27)
project(tack_v010 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# flags
add_library(tack_warnings INTERFACE)

target_compile_options(tack_warnings INTERFACE
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

find_package(CUDAToolkit REQUIRED)

if (NOT DEFINED TACK_SOURCE_DIR)
  message(FATAL_ERROR "Need -DTACK_SOURCE_DIR")
endif()

if (NOT DEFINED GENERATED_SRC)
  message(FATAL_ERROR "Need -DGENERATED_SRC")
endif()

if (NOT DEFINED TARGET_MODULE_NAME)
  message(FATAL_ERROR "Need -DTARGET_MODULE_NAME")
endif()

if (NOT DEFINED ARCH)
  message(FATAL_ERROR "Need -DARCH")
endif()

if(${ARCH} LESS 70)
  message(FATAL_ERROR "Unsupported ARCH")
endif()

math(EXPR GPU_ARCH "${ARCH} * 10" OUTPUT_FORMAT DECIMAL)
set(ARCH_TAG "${ARCH}")
if(ARCH GREATER_EQUAL 90)
    string(APPEND ARCH_TAG "a") # accelerated arch-specific instruction set
endif()

# ---- CPM ----
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CPM.cmake)

#pybind11
set(PYBIND11_FINDPYTHON ON)
CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  VERSION 3.0.2
)
if(NOT COMMAND pybind11_add_module)
    message(FATAL_ERROR "pybind11 not found")
endif()
pybind11_add_module(${TARGET_MODULE_NAME} MODULE ${GENERATED_SRC})

target_link_libraries(${TARGET_MODULE_NAME} INTERFACE
  CUDA::cudart
  CUDA::cuda_driver
  tack_warnings
)

target_link_libraries(${TARGET_MODULE_NAME} INTERFACE atomic)

# -------------------------------------------------------
# CPM dependencies
# -------------------------------------------------------

# CCCL
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY nvidia/cccl
  VERSION 3.1.3
)
if(TARGET CCCL::CCCL)
  target_link_libraries(${TARGET_MODULE_NAME} INTERFACE CCCL::CCCL)
else()
  message(FATAL_ERROR "Unable to link CCCL")
endif()

# CUTLASS (header-only)
CPMAddPackage(
  NAME CUTLASS
  GITHUB_REPOSITORY nvidia/cutlass
  GIT_TAG v4.4.1
  DOWNLOAD_ONLY TRUE
)
if(DEFINED CUTLASS_SOURCE_DIR AND EXISTS "${CUTLASS_SOURCE_DIR}/include")
  target_include_directories(${TARGET_MODULE_NAME} INTERFACE
    ${CUTLASS_SOURCE_DIR}/include
  )
else()
  message(FATAL_ERROR "Unable to include CUTLASS")
endif()

#NVSHMEM
find_package(NVSHMEM REQUIRED)
# NVSHMEM imported target names differ across installs
if(TARGET nvshmem::nvshmem_host AND TARGET nvshmem::nvshmem_device)
    # WAR for build failure
  if (NOT DEFINED ENV{NVSHMEM_LIB_HOME})
    message(FATAL_ERROR "NVSHMEM_LIB_HOME not defined")
  endif ()
  set_target_properties(nvshmem::nvshmem_host PROPERTIES
    IMPORTED_LOCATION_RELEASE "$ENV{NVSHMEM_LIB_HOME}/libnvshmem_host.so"
  )
  target_link_libraries(${TARGET_MODULE_NAME} INTERFACE
    nvshmem::nvshmem_device
    nvshmem::nvshmem_host
  )
else()
  message(FATAL_ERROR "NVSHMEM found but no supported CMake targets detected")
endif()

target_include_directories(${TARGET_MODULE_NAME} PRIVATE "${TACK_SOURCE_DIR}/tack/include")

target_compile_options(${TARGET_MODULE_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr; -Xptxas -v; -t0>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=lto_${ARCH_TAG}>
)