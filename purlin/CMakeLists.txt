cmake_minimum_required(VERSION 3.27)
project(purlin_v010 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# flags
add_library(purlin_warnings INTERFACE)

target_compile_options(purlin_warnings INTERFACE
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
include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)

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
if (DEFINED PURLIN_SOURCE_DIR AND
    EXISTS "${PURLIN_SOURCE_DIR}/csrc/include" AND
    EXISTS "${PURLIN_SOURCE_DIR}/CMakeLists.txt")
    add_subdirectory("${PURLIN_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/purlin-source" EXCLUDE_FROM_ALL)
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE purlin::purlin)
else()
    CPMAddPackage(
        NAME purlin
        GITHUB_REPOSITORY osayamenja/purlin
        GIT_TAG v0.2.0
    )
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE purlin::purlin)
endif()

target_compile_options(${TARGET_MODULE_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr; -t0>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
)
target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE
            ARCH=${GPU_ARCH}
            PURLIN_NVTX=1
)
if (DEFINED JIT_WORLD)
target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE
            PURLIN_JIT_WORLD=${JIT_WORLD}
)
endif()
