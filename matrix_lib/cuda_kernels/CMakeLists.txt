cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)
project(cuda_kernels LANGUAGES CUDA CXX)

file(GLOB_RECURSE SRC
    "src/*.cu"
    "src/*.cuh"
)
add_library(cuda_kernels STATIC ${SRC})

# Post Build
add_custom_command(TARGET cuda_kernels POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cuda_kernels> ${CMAKE_SOURCE_DIR})