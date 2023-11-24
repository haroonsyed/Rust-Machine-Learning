cmake_minimum_required(VERSION 3.11)

project(cuda_kernels LANGUAGES CUDA CXX)

# Cuda Code
file(GLOB_RECURSE SRC
    "src/*.cu"
    "src/*.cuh"
)
add_library(cuda_kernels STATIC ${SRC})

# Configure
find_package(CUDA REQUIRED)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(cuda_kernels PROPERTIES CUDA_ARCHITECTURES "52;75;86;89;90")
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall --std=c++11 -O3 -fPIC" )
target_compile_options(cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Debug Symbols
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

# Link libraries
target_link_libraries(
  cuda_kernels
  ${CUDA_LIBRARIES}
  )

# Install
install(TARGETS cuda_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Post Build
add_custom_command(TARGET cuda_kernels POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cuda_kernels> ${CMAKE_SOURCE_DIR})
