#pragma once
#include "./cuda_exec_memory_manager.cuh"
#include "./types.cuh"

extern "C" {

// Test Functions
void test();
void test_array_fill(float* buffer, size_t length);

// Execution Type
void enable_parallel_stream_execution();
void disable_parallel_stream_execution();

// Misc
void cuda_synchronize();

// Matrix operation API
size_t cuda_element_add(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
void cuda_element_add_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool inplace);
size_t cuda_element_subtract(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
void cuda_element_subtract_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool inplace);
size_t cuda_element_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
void cuda_element_multiply_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool inplace);
size_t cuda_element_divide(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
void cuda_element_divide_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool inplace);
size_t cuda_scalar_multiply(size_t mat_id, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
void cuda_scalar_multiply_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
size_t cuda_scalar_divide(size_t mat_id, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
void cuda_scalar_divide_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
size_t cuda_scalar_add(size_t mat_id, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
void cuda_scalar_add_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
size_t cuda_scalar_subtract(size_t mat_id, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
void cuda_scalar_subtract_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
size_t cuda_scalar_multiply_matrix(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t scalar_mat_id, bool inplace);
void cuda_scalar_multiply_matrix_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t scalar_mat_id, bool inplace);
size_t cuda_scalar_divide_matrix(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t scalar_mat_id, bool inplace);
void cuda_scalar_divide_matrix_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t scalar_mat_id, bool inplace);
size_t cuda_scalar_add_matrix(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t scalar_mat_id, bool inplace);
void cuda_scalar_add_matrix_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t scalar_mat_id, bool inplace);
size_t cuda_scalar_subtract_matrix(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t scalar_mat_id, bool inplace);
void cuda_scalar_subtract_matrix_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t scalar_mat_id, bool inplace);
size_t cuda_matrix_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols);
size_t cuda_add_vector(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_divide_by_vector(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_element_sqrt(size_t mat_id, size_t mat_rows, size_t mat_cols, bool inplace);
void cuda_element_sqrt_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool inplace);
size_t cuda_element_exp(size_t mat_id, size_t mat_rows, size_t mat_cols, bool inplace);
void cuda_element_exp_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool inplace);
size_t cuda_element_ReLU(size_t mat_id, size_t mat_rows, size_t mat_col, bool inplace);
void cuda_element_ReLU_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool inplace);
size_t cuda_element_ReLU_prime(size_t mat_id, size_t mat_rows, size_t mat_cols, bool inplace);
void cuda_element_ReLU_prime_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool inplace);
size_t cuda_sum_rows(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_sum_columns(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_transpose(size_t mat_id, size_t mat_rows, size_t mat_cols);
Tuple cuda_max_pool(size_t mat_id, size_t mat_rows, size_t mat_cols);
void cuda_max_pool_packed(size_t* mat_ids, Tuple* out_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_nearest_neighbor_2x_upsample(size_t mat_id, size_t mat_rows, size_t mat_cols, bool odd_upsample);
void cuda_nearest_neighbor_2x_upsample_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool odd_upsample);
size_t cuda_rotate_180(size_t mat_id, size_t mat_rows, size_t mat_cols);
void cuda_rotate_180_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_convolution(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols, ConvolutionType conv_type);
void cuda_convolution_packed(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_ids, ConvolutionType conv_type);
size_t cuda_img2col(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t kernel_rows, size_t kernel_cols, ConvolutionType conv_type);  // Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
size_t cuda_flatten_array(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);                                                               // Take n same_dimension matrices and flatten them into an array
void cuda_unflatten_array(size_t array_id, size_t arr_size, size_t mat_rows, size_t mat_cols, size_t* mat_ids);                                                  // Take an array and unflatten it into n same_dimension matrices
void cuda_unflatten_array_strided(size_t array_id, size_t arr_size, size_t mat_rows, size_t mat_cols, size_t* mat_ids);                                          // Take an array and unflatten it into n same_dimension matrices. Each array's first n elements are the first elements in memory. [arr1_elem1, arr2_elem1, arr3_elem1, arr1_elem2, arr2_elem2, arr3_elem2, ...]
size_t cuda_center_pad(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t pad_rows, size_t pad_cols);
size_t cuda_softmax(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_crop(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t crop_offset_rows, size_t crop_offset_cols, size_t crop_rows, size_t crop_cols);
size_t cuda_copy(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_sum_all_matrix_elements(size_t mat_id, size_t mat_rows, size_t mat_cols);
}