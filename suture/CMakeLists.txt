cmake_minimum_required(VERSION 3.27)
project(suture_v010 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# flags
add_library(suture_warnings INTERFACE)

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

find_package(CUDAToolkit REQUIRED)

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
  VERSION 3.0.4
)
if(NOT COMMAND pybind11_add_module)
    message(FATAL_ERROR "pybind11 not found")
endif()
pybind11_add_module(${TARGET_MODULE_NAME} MODULE ${GENERATED_SRC})

# -------------------------------------------------------
# dependencies
# -------------------------------------------------------
CPMAddPackage(
    NAME suture
    GITHUB_REPOSITORY osayamenja/suture
    GIT_TAG v0.1.1
)
target_link_libraries(${TARGET_MODULE_NAME} PRIVATE suture::suture)

target_compile_options(${TARGET_MODULE_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr; -Xptxas -v; -t0>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
)
target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE
            ARCH=${GPU_ARCH}
            SUTURE_NVTX=1
)