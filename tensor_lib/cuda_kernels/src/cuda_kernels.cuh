#include <cublas_v2.h>
#include <cuda.h>
#include <stdio.h>

// Experiment to make access to map thread safe.
// For now only use one thread with this library.

// std::shared_mutex mat_map_mutex;
// Function to perform thread-safe write operation on mat_map
// void writeToMatMap(size_t key, float* value) {
//     std::unique_lock<std::shared_mutex> lock(mat_map_mutex);
//     mat_map[key] = value;
// }
// // Function to perform thread-safe read operation on mat_map
// float* readFromMatMap(size_t key) {
//     std::shared_lock<std::shared_mutex> lock(mat_map_mutex);
//     return mat_map[key];
// }
// Function to perform thread-safe remove operation on mat_map
// void removeFromMatMap(size_t key) {
//     std::unique_lock<std::shared_mutex> lock(mat_map_mutex);
//     mat_map.erase(key);
// }

// Make enum for convolution types
enum ConvolutionType {
    VALID,
    SAME,
    FULL
};

// Make sure bindings are not mangled for rust
extern "C" {
void test();
void test_array_fill(float* buffer, size_t length);

// Misc
void cuda_synchronize();
struct Tuple {  // Used to return tuple with interop to rust
    size_t a;
    size_t b;
};

// Matrix Setup API (reduces overhead of keeping matrices in ram)
size_t register_matrix(float* data, size_t rows, size_t cols);
void unregister_matrix(size_t mat_id);
void get_matrix_data(size_t mat_id, int rows, int cols, float* data_buffer);

// Execution Type
void enable_parallel_stream_execution();
void disable_parallel_stream_execution();

// Matrix operation API, Returns id of new matrix. Consumer should not release
size_t cuda_element_add(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_element_subtract(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_element_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_scalar_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, float scalar, bool inplace);
size_t cuda_matrix_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols);
size_t cuda_add_vector(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_divide_by_vector(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_element_exp(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace);
size_t cuda_element_ReLU(size_t mat1_id, size_t mat1_rows, size_t mat1_col, bool inplace);
size_t cuda_element_ReLU_prime(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace);
size_t cuda_sum_rows(size_t mat1_id, size_t mat1_rows, size_t mat1_cols);
size_t cuda_sum_columns(size_t mat1_id, size_t mat1_rows, size_t mat1_cols);
size_t cuda_transpose(size_t mat1_id, size_t mat1_rows, size_t mat1_cols);
Tuple cuda_max_pool(size_t mat1_id, size_t mat1_rows, size_t mat1_cols);
size_t cuda_nearest_neighbor_2x_upsample(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool odd_upsample);
size_t cuda_rotate_180(size_t mat1_id, size_t mat1_rows, size_t mat1_cols);
size_t cuda_convolution(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols, ConvolutionType conv_type);
void cuda_convolution_packed(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_ids, ConvolutionType conv_type);
size_t cuda_img2col(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t kernel_rows, size_t kernel_cols, ConvolutionType conv_type);  // Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
size_t cuda_flatten_array(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);                                                               // Take n same_dimension matrices and flatten them into an array
void cuda_unflatten_array(size_t array_id, size_t arr_size, size_t mat_rows, size_t mat_cols, size_t* mat_ids);                                                  // Take an array and unflatten it into n same_dimension matrices
void cuda_unflatten_array_strided(size_t array_id, size_t arr_size, size_t mat_rows, size_t mat_cols, size_t* mat_ids);                                          // Take an array and unflatten it into n same_dimension matrices. Each array's first n elements are the first elements in memory. [arr1_elem1, arr2_elem1, arr3_elem1, arr1_elem2, arr2_elem2, arr3_elem2, ...]
}
