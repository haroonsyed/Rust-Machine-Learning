#pragma once
#include <cublas_v2.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "./err_check_util.cu"

extern cublasHandle_t handle;

extern "C" {
// Memory Manager Functions
void* get_block_gpu_address(size_t block_id, size_t block_offset);
size_t memory_manager_allocate(size_t size);
void memory_manager_free(size_t block_id, size_t size);
void memory_manager_upload_to_allocation(void* address, void* data, size_t size);
void memory_manager_upload_from_pinned_buffer(void* pinned_data, void* device_address, size_t size);
void memory_manager_upload_async_from_pinned_buffer(void* pinned_data, void* device_address, size_t size);
void* memory_get_pinned_allocation(size_t size);
std::vector<void*> get_device_kernel_args_pointers(size_t num_buffers);

// Matrix Setup API
cudaStream_t get_stream();
float* get_matrix_gpu_address(size_t mat_id);
size_t register_matrix(size_t rows, size_t cols);
void register_matrix_group(size_t rows, size_t columns, size_t count, size_t* mat_ids);
void upload_matrix_data(size_t mat_id, float* data);
size_t register_matrix_with_data(float* data, size_t rows, size_t cols);
void unregister_matrix(size_t mat_id);
void increase_matrix_ref_count(size_t mat_id);
void decrease_matrix_ref_count(size_t mat_id);
void get_matrix_data(size_t mat_id, float* data_buffer);

// Matrix info API
size_t get_matrix_rows(size_t mat_id);
size_t get_matrix_columns(size_t mat_id);
size_t get_matrix_length(size_t mat_id);
void reshape_matrix(size_t mat_id, size_t rows, size_t columns);
}
