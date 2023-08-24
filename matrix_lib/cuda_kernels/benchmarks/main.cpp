#include <stddef.h>

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono;

extern "C" {
void test();
void test_array_fill(float* buffer, size_t length);

// Misc
void cuda_synchronize();

// Matrix Setup API (reduces overhead of keeping matrices in ram)
size_t register_matrix(float* data, size_t rows, size_t cols);
void unregister_matrix(size_t mat_id);
void get_matrix_data(size_t mat_id, int rows, int cols, float* data_buffer);

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
}

int main() {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    int dim = 4096;
    std::vector<float> data;
    for (int i = 0; i < dim * dim; i++) {
        data.push_back(23.47);
    }

    // Register
    int mat1 = register_matrix(&data[0], dim, dim);
    int mat2 = register_matrix(&data[0], dim, dim);

    auto start_host = high_resolution_clock::now();

    int num_iter = 100;
    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        int result_id = cuda_matrix_multiply(mat1, dim, dim, mat2, dim, dim);
        cuda_synchronize();
        unregister_matrix(result_id);
    }

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;

    // Okay something is wrong with the overhead on rust benchmark. Something taking 184.3 ms here is taking 1.3 seconds there.
    // Same functions on ffi being called...
}