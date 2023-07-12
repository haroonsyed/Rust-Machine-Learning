#include <cuda.h>

#include <iostream>
#include <unordered_map>

size_t mat_generated_count = 0;
std::unordered_map<size_t, double*> mat_map;

// Make sure bindings are not mangled for rust
extern "C" {
void test();
void test_array_fill(double* buffer, size_t length);

// Matrix Setup API (reduces overhead of keeping matrices in ram)
size_t register_matrix(double* data, size_t rows, size_t cols);
void unregister_matrix(size_t mat_id);
void get_matrix_data(size_t mat_id, int rows, int cols, double* data_buffer);

// Matrix operation API
size_t cuda_element_add(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols);            // Returns id of new matrix
size_t cuda_element_subtract(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols);       // Returns id of new matrix
size_t cuda_element_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols);       // Returns id of new matrix
size_t cuda_scalar_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, double scalar);                                             // Returns id of new matrix
size_t cuda_matrix_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols);        // Returns id of new matrix
size_t cuda_add_vector_to_columns(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols);  // Returns id of new matrix
                                                                                                                                            // mat2_cols should be 1
size_t cuda_sum_rows(size_t mat1_id, size_t mat1_rows, size_t mat1_cols);                                                                   // Returns id of new matrix
size_t cuda_sum_columns(size_t mat1_id, size_t mat1_rows, size_t mat1_cols);                                                                // Returns id of new matrix
size_t cuda_transpose(size_t mat1_id, size_t mat1_rows, size_t mat1_cols);                                                                  // Returns id of new matrix
}
