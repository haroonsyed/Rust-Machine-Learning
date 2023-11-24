#include <stddef.h>

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono;

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
size_t register_matrix_with_data(float* data, size_t rows, size_t cols);
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

void warmup() {
    std::cout << "Warming Up..." << std::endl;

    int mat_dim = 4096;
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    // WARMUP
    for (int i = 0; i < 50; i++) {
        // Perform multiplication
        int result_id = cuda_matrix_multiply(mat1, mat_dim, mat_dim, mat2, mat_dim, mat_dim);
        unregister_matrix(result_id);
    }
    cuda_synchronize();

    std::cout << "Finished warming up, starting benchmark..." << std::endl;
}

void bench_matrix_transpose(int mat_dim, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        int result_id = cuda_transpose(mat1, mat_dim, mat_dim);
        cuda_synchronize();
        unregister_matrix(result_id);
    }

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_matrix_transpose was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void bench_max_pool(int mat_dim, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        Tuple result = cuda_max_pool(mat1, mat_dim, mat_dim);
        unregister_matrix(result.a);
        unregister_matrix(result.b);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_max_pool was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void bench_rotate_180(int mat_dim, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        int result_id = cuda_rotate_180(mat1, mat_dim, mat_dim);
        unregister_matrix(result_id);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_rotate_180 was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void bench_convolution(int mat_dim, int num_iter, int kernel_size) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(i);
    }

    std::vector<float> kernel_data;
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel_data.push_back(i);
    }

    // Register
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        int result_id = cuda_convolution(mat1, mat_dim, mat_dim, kernel, kernel_size, kernel_size, VALID);
        unregister_matrix(result_id);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_convolution was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void bench_multiple_convolutions(int mat_dim, int num_iter, int num_matrices, int kernel_size) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(i);
    }

    std::vector<float> kernel_data;
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel_data.push_back(i);
    }

    // Register
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        for (int i = 0; i < num_matrices; i++) {
            int result_id = cuda_convolution(mat1, mat_dim, mat_dim, kernel, kernel_size, kernel_size, VALID);
            unregister_matrix(result_id);
        }
    }
    cuda_synchronize();

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_multiple_convolutions was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_packed_convolution(int mat_dim, int num_iter, int num_matrices, int kernel_size) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(i);
    }

    std::vector<float> kernel_data;
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel_data.push_back(i);
    }

    // Register
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);

    // Repeat the mat ids and kernel ids to simulate multiple images and kernels
    std::vector<size_t> mat_ids;
    std::vector<size_t> kernel_ids;
    std::vector<size_t> result_ids;
    for (int i = 0; i < num_matrices; i++) {
        mat_ids.push_back(mat1);
        kernel_ids.push_back(kernel);
        result_ids.push_back(0);
    }

    cuda_synchronize();

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        cuda_convolution_packed(&mat_ids[0], num_matrices, mat_dim, mat_dim, &kernel_ids[0], kernel_size, kernel_size, &result_ids[0], VALID);

        // Unregister matrices
        for (auto id : result_ids) {
            unregister_matrix(id);
        }
    }

    cuda_synchronize();

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_packed_convolution was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_img2col(int mat_dim, int num_iter, int kernel_size) {
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(i);
    }

    size_t mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    int img2col_rows = kernel_size * kernel_size;
    int img2col_cols = (mat_dim - kernel_size + 1) * (mat_dim - kernel_size + 1);

    std::vector<size_t> matrices = {mat1};

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // img2col of input
        int mat1_img2col = cuda_img2col(&matrices[0], 1, mat_dim, mat_dim, kernel_size, kernel_size, VALID);

        unregister_matrix(mat1_img2col);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_img2col was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void bench_convolution_2(int mat_dim, int num_iter, int kernel_size) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(i);
    }

    std::vector<float> kernel_data;
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel_data.push_back(i);
    }

    // Register
    size_t mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    size_t kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);

    int img2col_rows = kernel_size * kernel_size;
    int img2col_cols = (mat_dim - kernel_size + 1) * (mat_dim - kernel_size + 1);

    auto start_host = high_resolution_clock::now();

    std::vector<size_t> kernels = {kernel};
    std::vector<size_t> matrices = {mat1};

    for (int i = 0; i < num_iter; i++) {
        // Flatten the kernel
        int kernel_flat = cuda_flatten_array(&kernels[0], 1, kernel_size, kernel_size);

        // img2col of input
        int mat1_img2col = cuda_img2col(&matrices[0], 1, mat_dim, mat_dim, kernel_size, kernel_size, VALID);

        // Perform matrix multiplication
        int result_id = cuda_matrix_multiply(kernel_flat, 1, kernel_size * kernel_size, mat1_img2col, img2col_rows, img2col_cols);
        unregister_matrix(kernel_flat);
        unregister_matrix(mat1_img2col);
        unregister_matrix(result_id);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_convolution_2 was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void bench_matrix_multiply(int mat_dim, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        int result_id = cuda_matrix_multiply(mat1, mat_dim, mat_dim, mat2, mat_dim, mat_dim);
        unregister_matrix(result_id);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_matrix_multiply was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void bench_flatten_array(int mat_dim, int mat_count, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    std::vector<size_t> mat_ids;
    for (int i = 0; i < mat_count; i++) {
        mat_ids.push_back(register_matrix_with_data(&data[0], mat_dim, mat_dim));
    }

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        int result_id = cuda_flatten_array(&mat_ids[0], mat_ids.size(), mat_dim, mat_dim);
        unregister_matrix(result_id);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_flatten_array was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void bench_unflatten_array(int mat_dim, int mat_count, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_count * mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    size_t mat = register_matrix_with_data(&data[0], 1, mat_count * mat_dim * mat_dim);
    std::vector<size_t> unflattened;
    for (int i = 0; i < mat_count; i++) {
        unflattened.push_back(0);
    }

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        cuda_unflatten_array(mat, mat_count * mat_dim * mat_dim, mat_dim, mat_dim, &unflattened[0]);
        for (auto id : unflattened) {
            unregister_matrix(id);
        }
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_unflatten_array was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
}

void stress_test_mix_short_lived_long_lived_blocks() {
    std::vector<float> data;
    for (int i = 0; i < 2 * 3; i++) {
        data.push_back(i + 1);
    }

    std::vector<size_t> results;
    for (int i = 0; i < 10000; i++) {
        // Register
        int mat1 = register_matrix_with_data(&data[0], 2, 3);

        // Perform multiplication
        int result_id = cuda_sum_rows(mat1, 2, 3);
        unregister_matrix(mat1);
    }

    for (auto id : results) {
        unregister_matrix(id);
    }
}

int main() {
    // Get the temps and frequencies up
    // warmup();

    // Used with ncu to profile kernels. Will expand to have all kernels, but for now just has the most time consuming ones

    const int mat_dim = 1024;
    const int kernel_dim = 3;
    const int num_iter = 1000;
    // bench_flatten_array(mat_dim, 256, num_iter);
    // bench_unflatten_array(mat_dim, 256, num_iter);
    // bench_img2col(mat_dim, num_iter, kernel_dim);
    // bench_convolution(mat_dim, num_iter, kernel_dim);
    bench_multiple_convolutions(32, num_iter, 1024, kernel_dim);
    bench_packed_convolution(32, num_iter, 1024, kernel_dim);  // Seems to be faster with matrices smaller than or eqal to 32x32, scaling at about 2x to 3x. Shows overhead of launches.
    // bench_convolution_2(mat_dim, num_iter, kernel_dim);
    // bench_rotate_180(mat_dim, num_iter);
    // bench_max_pool(mat_dim, num_iter);
    // bench_matrix_transpose(mat_dim, num_iter);
    // bench_matrix_multiply(mat_dim, num_iter);

    // Stress tests
    // TODO: Fix adjacent short and long lived blocks.
    // stress_test_mix_short_lived_long_lived_blocks();
}