cmake_minimum_required(VERSION 3.14)
project(AnchorDETRTrt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# Compiler Options
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_CXX_FLAGS "-O2 -ffast-math")
endif()

set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS "--std=c++14
                     -gencode arch=compute_60,code=sm_60
                     -gencode arch=compute_61,code=sm_61
                     -gencode arch=compute_70,code=sm_70
                     -gencode arch=compute_75,code=sm_75
                     -gencode arch=compute_80,code=sm_80")

include_directories(
	/usr/local/cuda/include/
	/usr/local/TensorRT/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

link_directories(
	/usr/local/cuda/lib64
	/usr/local/TensorRT/lib
        /etc/alternatives/cuda/lib64
)

cuda_add_executable(AnchorDETRTrt TrtExec.cpp LayerNormPlugin.cu)
target_link_libraries(AnchorDETRTrt
        cudart
        nvinfer
        nvparsers
        nvinfer_plugin
        nvonnxparser)
cuda_add_library(TrtPlugin SHARED LayerNormPlugin.cu)
target_link_libraries(TrtPlugin
	cudart
	nvinfer
	nvinfer_plugin)
