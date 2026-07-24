cmake_minimum_required(VERSION 3.28)

project(purlin LANGUAGES CXX CUDA)

# -------------------------------------------------------
# Make CPM available
# -------------------------------------------------------
if(NOT COMMAND CPMAddPackage)
  include(${CMAKE_CURRENT_LIST_DIR}/csrc/cmake/CPM.cmake)
endif()

# -------------------------------------------------------
# Header-only purlin library
# -------------------------------------------------------
add_library(purlin INTERFACE)
add_library(purlin::purlin ALIAS purlin)

target_include_directories(purlin INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/csrc/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(purlin INTERFACE cxx_std_20)
find_package(CUDAToolkit REQUIRED)

# -------------------------------------------------------
# Required dependencies discoverable via find_package()
# -------------------------------------------------------
target_link_libraries(purlin INTERFACE
  CUDA::cudart
  CUDA::cuda_driver
  atomic
)

# CCCL
CPMAddPackage(
  NAME CCCL
  URL https://github.com/NVIDIA/cccl/archive/refs/tags/v3.3.0.tar.gz
)
if(TARGET CCCL::CCCL)
  target_link_libraries(purlin INTERFACE CCCL::CCCL)
else()
  message(FATAL_ERROR "Unable to link CCCL")
endif()

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.5.0-c-cpp
        GIT_SHALLOW TRUE
)
if(TARGET nvtx3-cpp)
  target_link_libraries(purlin INTERFACE nvtx3-cpp)
endif ()
