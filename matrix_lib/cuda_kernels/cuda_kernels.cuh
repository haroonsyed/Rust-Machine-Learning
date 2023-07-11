#include <cuda.h>

#include <iostream>

// Make sure bindings are not mangled for rust
extern "C" {
void test();
void test_array_fill(double* buffer, size_t length);

// Actual matrix lib
void cuda_element_add(double* mat1_buffer, size_t mat1_rows, size_t mat1_cols, double* mat2_buffer, size_t mat2_rows, size_t mat2_cols, double* out_buffer, size_t out_rows, size_t out_cols);
}
