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
  GITHUB_REPOSITORY nvidia/cccl
  VERSION 3.3.0
)
if(TARGET CCCL::CCCL)
  target_link_libraries(purlin INTERFACE CCCL::CCCL)
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
  target_include_directories(purlin INTERFACE
    ${CUTLASS_SOURCE_DIR}/include
  )
else()
  message(FATAL_ERROR "Unable to include CUTLASS")
endif()