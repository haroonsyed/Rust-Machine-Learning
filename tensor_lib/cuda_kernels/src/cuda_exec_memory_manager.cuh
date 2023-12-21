#pragma once
#include <cublas_v2.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "./err_check_util.cu"
#include "./types.cuh"

extern cublasHandle_t handle;

extern "C" {

// Memory Manager Funsize_tctions
void* get_block_gpu_address(size_t block_id);
size_t memory_manager_device_allocate(size_t block_id);
void memory_manager_free(size_t size_t);
void memory_manager_upload_to_allocation(void* address, void* data, size_t size);
void memory_manager_upload_from_pinned_buffer(void* device_address, void* pinned_data, size_t size);
void memory_manager_upload_async_from_pinned_buffer(void* device_address, void* pinned_data, size_t size);
void* memory_manager_get_pinned_allocation(size_t size);
std::vector<Matrix*> get_device_kernel_args_pointers(size_t num_buffers);

// Matrix Setup API
cudaStream_t get_stream();
Matrix register_matrix(size_t rows, size_t columns);
void register_matrix_group(size_t rows, size_t columns, size_t count, Matrix* matrices);
void upload_matrix_data(Matrix* matrix, float* data);
void upload_matrix_data_async(Matrix* matrix, float* data);
Matrix register_matrix_with_data(float* data, size_t rows, size_t columns);
void unregister_matrix(Matrix* matrix);
void increase_matrix_ref_count(Matrix* matrix);
void decrease_matrix_ref_count(Matrix* matrix);
void get_matrix_data(Matrix* matrix, float* data_buffer);

// Matrix info API
size_t get_matrix_rows(Matrix* matrix);
size_t get_matrix_columns(Matrix* matrix);
size_t get_matrix_length(Matrix* matrix);
void reshape_matrix(Matrix* matrix, size_t rows, size_t columns);
}
