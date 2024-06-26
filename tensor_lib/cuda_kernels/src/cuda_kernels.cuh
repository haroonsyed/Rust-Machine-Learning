#pragma once
#include "./cuda_exec_memory_manager.cuh"

extern "C" {

// Test Functions
void test();
void test_array_fill(float* buffer, size_t length);

// Misc
void cuda_synchronize();

// Matrix operation API
Matrix cuda_element_add(Matrix* matrix_1, Matrix* matrix_2);
void cuda_element_add_inplace(Matrix* matrix_1, Matrix* matrix_2);
void cuda_element_add_packed(Matrix* matrix_1s, Matrix* matrix_2s, Matrix* out_matrices, size_t num_matrices);
void cuda_element_add_packed_inplace(Matrix* matrix_1s, Matrix* matrix_2s, size_t num_matrices);
Matrix cuda_element_subtract(Matrix* matrix_1, Matrix* matrix_2);
void cuda_element_subtract_inplace(Matrix* matrix_1, Matrix* matrix_2);
void cuda_element_subtract_packed(Matrix* matrix_1s, Matrix* matrix_2s, Matrix* out_matrices, size_t num_matrices);
void cuda_element_subtract_packed_inplace(Matrix* matrix_1s, Matrix* matrix_2s, size_t num_matrices);
Matrix cuda_element_multiply(Matrix* matrix_1, Matrix* matrix_2);
void cuda_element_multiply_inplace(Matrix* matrix_1, Matrix* matrix_2);
void cuda_element_multiply_packed(Matrix* matrix_1s, Matrix* matrix_2s, Matrix* out_matrices, size_t num_matrices);
void cuda_element_multiply_packed_inplace(Matrix* matrix_1s, Matrix* matrix_2s, size_t num_matrices);
Matrix cuda_element_divide(Matrix* matrix_1, Matrix* matrix_2);
void cuda_element_divide_inplace(Matrix* matrix_1, Matrix* matrix_2);
void cuda_element_divide_packed(Matrix* matrix_1s, Matrix* matrix_2s, Matrix* out_matrices, size_t num_matrices);
void cuda_element_divide_packed_inplace(Matrix* matrix_1s, Matrix* matrix_2s, size_t num_matrices);
Matrix cuda_scalar_add(Matrix* matrix, float scalar);
void cuda_scalar_add_inplace(Matrix* matrix, float scalar);
void cuda_scalar_add_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices, float scalar);
void cuda_scalar_add_packed_inplace(Matrix* matrices, size_t num_matrices, float scalar);  // No longer safe to repeat matrix in matrices (i.e double multiply a matrix) (removed atomic operation)
Matrix cuda_scalar_subtract(Matrix* matrix, float scalar);
void cuda_scalar_subtract_inplace(Matrix* matrix, float scalar);
void cuda_scalar_subtract_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices, float scalar);
void cuda_scalar_subtract_packed_inplace(Matrix* matrices, size_t num_matrices, float scalar);  // No longer safe to repeat matrix in matrices (i.e double multiply a matrix) (removed atomic operation)
Matrix cuda_scalar_multiply(Matrix* matrix, float scalar);
void cuda_scalar_multiply_inplace(Matrix* matrix, float scalar);
void cuda_scalar_multiply_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices, float scalar);
void cuda_scalar_multiply_packed_inplace(Matrix* matrices, size_t num_matrices, float scalar);  // No longer safe to repeat matrix in matrices (i.e double multiply a matrix) (removed atomic operation)
Matrix cuda_scalar_divide(Matrix* matrix, float scalar);
void cuda_scalar_divide_inplace(Matrix* matrix, float scalar);
void cuda_scalar_divide_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices, float scalar);
void cuda_scalar_divide_packed_inplace(Matrix* matrices, size_t num_matrices, float scalar);  // No longer safe to repeat matrix in matrices (i.e double multiply a matrix) (removed atomic operation)
Matrix cuda_matrix_multiply(Matrix* matrix_1, Matrix* matrix_2);
Matrix cuda_add_vector(Matrix* matrix_1, Matrix* matrix_2);
void cuda_add_vector_inplace(Matrix* matrix_1, Matrix* matrix_2);
Matrix cuda_divide_by_vector(Matrix* matrix_1, Matrix* matrix_2);
void cuda_divide_by_vector_inplace(Matrix* matrix_1, Matrix* matrix_2);
Matrix cuda_element_sqrt(Matrix* matrix);
void cuda_element_sqrt_inplace(Matrix* matrix);
void cuda_element_sqrt_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices);
void cuda_element_sqrt_packed_inplace(Matrix* matrices, size_t num_matrices);
Matrix cuda_element_exp(Matrix* matrix);
void cuda_element_exp_inplace(Matrix* matrix);
void cuda_element_exp_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices);
void cuda_element_exp_packed_inplace(Matrix* matrices, size_t num_matrices);
Matrix cuda_element_ReLU(Matrix* matrix);
void cuda_element_ReLU_inplace(Matrix* matrix);
void cuda_element_ReLU_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices);
void cuda_element_ReLU_packed_inplace(Matrix* matrices, size_t num_matrices);
Matrix cuda_element_ReLU_prime(Matrix* matrix);
void cuda_element_ReLU_prime_inplace(Matrix* matrix);
void cuda_element_ReLU_prime_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices);
void cuda_element_ReLU_prime_packed_inplace(Matrix* matrices, size_t num_matrices);
Matrix cuda_element_ln(Matrix* matrix);
void cuda_element_ln_inplace(Matrix* matrix);
Matrix cuda_sum_rows(Matrix* matrix);
Matrix cuda_sum_columns(Matrix* matrix);
Matrix cuda_transpose(Matrix* matrix);
void cuda_max_pool(Matrix* matrix, Matrix* out_pooled, Matrix* out_bitmask);
void cuda_max_pool_packed(Matrix* matrices, Matrix* out_pooled, Matrix* out_bitmasks, size_t num_matrices);
Matrix cuda_nearest_neighbor_2x_upsample(Matrix* matrix, bool odd_upsample);
void cuda_nearest_neighbor_2x_upsample_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices, bool odd_upsample);
Matrix cuda_rotate_180(Matrix* matrix);
void cuda_rotate_180_packed(Matrix* matrices, Matrix* out_matrices, size_t num_matrices);
Matrix cuda_correlate(Matrix* matrix, Matrix* kernel, PaddingType padding_type);
void cuda_correlate_packed(Matrix* matrices, size_t num_matrices, Matrix* kernels, Matrix* out_matrices, PaddingType padding_type);
Matrix cuda_convolve(Matrix* matrix, Matrix* kernel, PaddingType padding_type);
void cuda_convolve_packed(Matrix* matrices, size_t num_matrices, Matrix* kernels, Matrix* out_matrices, PaddingType padding_type);
Matrix cuda_img2col(Matrix* matrices, size_t num_matrices, size_t kernel_rows, size_t kernel_cols, PaddingType padding_type);  // Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
Matrix cuda_flatten_array(Matrix* matrices, size_t num_matrices);                                                              // Take n same_dimension matrices and flatten them into an array
void cuda_unflatten_array(Matrix* array, size_t out_rows, size_t out_cols, Matrix* out_matrices);                              // Take an array and unflatten it into n same_dimension matrices
void cuda_unflatten_array_strided(Matrix* array, size_t out_rows, size_t out_cols, Matrix* out_matrices);                      // Take an array and unflatten it into n same_dimension matrices. Each array's first n elements are the first elements in memory. [arr1_elem1, arr2_elem1, arr3_elem1, arr1_elem2, arr2_elem2, arr3_elem2, ...]
Matrix cuda_center_pad(Matrix* matrix, size_t pad_rows, size_t pad_cols);
Matrix cuda_softmax(Matrix* matrix);
Matrix cuda_crop(Matrix* matrix, size_t crop_offset_rows, size_t crop_offset_cols, size_t crop_rows, size_t crop_cols);
Matrix cuda_copy(Matrix* matrix);
Matrix cuda_sum_all_matrix_elements(Matrix* matrix);
Matrix cuda_max_by_column(Matrix* matrix);
Matrix cuda_max_by_row(Matrix* matrix);
Matrix cuda_argmax_by_column(Matrix* matrix);
Matrix cuda_argmax_by_row(Matrix* matrix);
Matrix cuda_one_hot_encode(float* data, size_t data_size, size_t num_classes);
Matrix cuda_one_hot_encode_vector(Matrix* matrix, size_t num_classes);

// Neural Network Specific Functions
void cuda_cnn_feed_forward(Matrix* channels, Matrix* filters, Matrix* biases, size_t channel_count_per_sample, size_t sample_count, size_t filter_count, Matrix* results);
void cuda_cnn_back_propogate(Matrix* sample_output_errors, Matrix* prev_inputs, Matrix* filters, size_t sample_count, size_t filter_count, size_t input_depth, Matrix* delta_bias, Matrix* delta_filter, Matrix* delta_input);
void cuda_adam_optimizer_packed(Matrix* d_v, Matrix* d_s, Matrix* curr_gradients, Matrix* results, size_t num_matrices, float d_v_beta, float d_s_beta, float learning_rate);
}