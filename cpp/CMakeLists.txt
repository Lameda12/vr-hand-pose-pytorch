cmake_minimum_required(VERSION 3.18)
project(turin_preprocess CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build optimizations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# pybind11
find_package(pybind11 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA-accelerated path (optional)
option(USE_CUDA "Enable CUDA-accelerated preprocessing" OFF)
if(USE_CUDA)
  find_package(OpenCV REQUIRED COMPONENTS cudawarping cudaimgproc)
  add_compile_definitions(USE_CUDA)
  message(STATUS "CUDA preprocessing enabled")
endif()

# Main extension module
pybind11_add_module(preprocess_cpp preprocess.cpp)

target_include_directories(preprocess_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(preprocess_cpp PRIVATE ${OpenCV_LIBS})

# AVX2 vectorization hint (optional, degrades gracefully)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
  target_compile_options(preprocess_cpp PRIVATE -mavx2 -mfma)
  message(STATUS "AVX2 vectorization enabled")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
