#pragma once
#include <cublas_v2.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "./err_check_util.cu"

extern bool library_init;
extern cudaStream_t mem_stream;
extern cudaMemPool_t mempool;
extern std::vector<cudaStream_t> exec_streams;
extern bool parallel_stream_execution;

extern cublasHandle_t handle;
extern size_t mat_generated_count;
extern std::unordered_map<size_t, float*> mat_map;

extern "C" {
// Matrix Setup API
cudaStream_t get_stream();
size_t register_matrix(size_t rows, size_t cols);
size_t register_matrix_with_data(float* data, size_t rows, size_t cols);
void unregister_matrix(size_t mat_id);
void get_matrix_data(size_t mat_id, int rows, int cols, float* data_buffer);

// Execution Type
void enable_parallel_stream_execution();
void disable_parallel_stream_execution();
}
