#include <stddef.h>

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono;

enum PaddingType {
    VALID,
    SAME,
    FULL
};

struct Matrix {
    float* address;
    size_t block_id;
};

extern "C" {
// Memory Manager Functions
void* get_block_gpu_address(size_t block_id);
size_t memory_manager_device_allocate(size_t block_id);
void memory_manager_free(size_t size_t);
void memory_manager_upload_to_allocation(void* address, void* data, size_t size);
void memory_manager_upload_from_pinned_buffer(void* device_address, void* pinned_data, size_t size);
void memory_manager_upload_async_from_pinned_buffer(void* device_address, void* pinned_data, size_t size);
void* memory_manager_get_pinned_allocation(size_t size);

// Matrix Setup API
Matrix register_matrix(size_t rows, size_t columns);
void register_matrix_group(size_t rows, size_t columns, size_t count, Matrix* matrices);
void register_matrix_group_with_value(size_t rows, size_t columns, size_t count, Matrix* matrices, float value);
void unregister_matrix_group(Matrix* matrix_group);
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

void warmup() {
    std::cout << "Warming Up..." << std::endl;

    int mat_dim = 4096;
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    // WARMUP
    for (int i = 0; i < 50; i++) {
        // Perform multiplication
        Matrix result = cuda_matrix_multiply(&mat1, &mat2);
        decrease_matrix_ref_count(&result);
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
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        Matrix result = cuda_transpose(&mat1);
        decrease_matrix_ref_count(&result);
    }

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_matrix_transpose was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_max_pool(int mat_dim, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix pooled;
    Matrix bitmask;

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        cuda_max_pool(&mat1, &pooled, &bitmask);
        decrease_matrix_ref_count(&pooled);
        decrease_matrix_ref_count(&bitmask);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_max_pool was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_rotate_180(int mat_dim, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        Matrix result = cuda_rotate_180(&mat1);
        decrease_matrix_ref_count(&result);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_rotate_180 was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
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
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        Matrix result = cuda_correlate(&mat1, &kernel, VALID);
        decrease_matrix_ref_count(&result);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_convolution was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_multiple_correlations(int mat_dim, int num_iter, int num_matrices, int kernel_size) {
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
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        for (int i = 0; i < num_matrices; i++) {
            Matrix result = cuda_correlate(&mat1, &kernel, VALID);
            decrease_matrix_ref_count(&result);
        }
    }
    cuda_synchronize();

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_multiple_convolutions was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_packed_correlation(int mat_dim, int num_iter, int num_matrices, int kernel_size) {
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
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);
    Matrix garbage;

    // Repeat the mat ids and kernel ids to simulate multiple images and kernels
    std::vector<Matrix> mats;
    std::vector<Matrix> kernels;
    std::vector<Matrix> results;
    for (int i = 0; i < num_matrices; i++) {
        mats.push_back(mat1);
        kernels.push_back(kernel);
        results.push_back(garbage);
    }

    cuda_synchronize();

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        cuda_correlate_packed(&mats[0], num_matrices, &kernels[0], &results[0], VALID);

        // Unregister matrices
        for (auto result : results) {
            decrease_matrix_ref_count(&result);
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

    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    int img2col_rows = kernel_size * kernel_size;
    int img2col_cols = (mat_dim - kernel_size + 1) * (mat_dim - kernel_size + 1);

    std::vector<Matrix> matrices = {mat1};

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // img2col of input
        Matrix mat1_img2col = cuda_img2col(&matrices[0], 1, kernel_size, kernel_size, VALID);

        decrease_matrix_ref_count(&mat1_img2col);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_img2col was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
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
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);

    int img2col_rows = kernel_size * kernel_size;
    int img2col_cols = (mat_dim - kernel_size + 1) * (mat_dim - kernel_size + 1);

    auto start_host = high_resolution_clock::now();

    std::vector<Matrix> kernels = {kernel};
    std::vector<Matrix> matrices = {mat1};

    for (int i = 0; i < num_iter; i++) {
        // Flatten the kernel
        Matrix kernel_flat = cuda_flatten_array(&kernels[0], 1);

        // img2col of input
        Matrix mat1_img2col = cuda_img2col(&matrices[0], 1, kernel_size, kernel_size, VALID);

        // Perform matrix multiplication
        Matrix result = cuda_matrix_multiply(&kernel_flat, &mat1_img2col);
        decrease_matrix_ref_count(&kernel_flat);
        decrease_matrix_ref_count(&mat1_img2col);
        decrease_matrix_ref_count(&result);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_convolution_2 was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_matrix_multiply(int mat_dim, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    Matrix mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    Matrix mat2 = register_matrix_with_data(&data[0], mat_dim, mat_dim);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        Matrix result = cuda_matrix_multiply(&mat1, &mat2);
        decrease_matrix_ref_count(&result);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_matrix_multiply was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_non_square_matrix_multiply(int M, int K, int N, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> a_data;
    for (int i = 0; i < M * K; i++) {
        a_data.push_back(23.47);
    }

    std::vector<float> b_data;
    for (int i = 0; i < K * N; i++) {
        b_data.push_back(35.346);
    }

    // Register
    Matrix mat1 = register_matrix_with_data(&a_data[0], M, K);
    Matrix mat2 = register_matrix_with_data(&b_data[0], K, N);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        Matrix result = cuda_matrix_multiply(&mat1, &mat2);
        decrease_matrix_ref_count(&result);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_non_square_matrix_multiply was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_flatten_array(int mat_dim, int mat_count, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    std::vector<Matrix> mat_ids;
    for (int i = 0; i < mat_count; i++) {
        mat_ids.push_back(register_matrix_with_data(&data[0], mat_dim, mat_dim));
    }

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        Matrix result = cuda_flatten_array(&mat_ids[0], mat_ids.size());
        decrease_matrix_ref_count(&result);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_flatten_array was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_unflatten_array(int mat_dim, int mat_count, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data;
    for (int i = 0; i < mat_count * mat_dim * mat_dim; i++) {
        data.push_back(23.47);
    }

    // Register
    Matrix mat = register_matrix_with_data(&data[0], 1, mat_count * mat_dim * mat_dim);
    std::vector<Matrix> unflattened;
    for (int i = 0; i < mat_count; i++) {
        Matrix garbage;
        unflattened.push_back(garbage);
    }

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        cuda_unflatten_array(&mat, mat_dim, mat_dim, &unflattened[0]);
        for (auto mat : unflattened) {
            decrease_matrix_ref_count(&mat);
        }
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_unflatten_array was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void bench_one_hot_encode(int num_labels, int num_iter) {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    std::vector<float> data(num_labels);
    for (int i = 0; i < num_labels; i++) {
        data[i] = i;
    }

    // Register
    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        Matrix result = cuda_one_hot_encode(&data[0], num_labels, num_labels);
        decrease_matrix_ref_count(&result);
    }
    cuda_synchronize();

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_host - start_host);

    std::cout << "Including overhead bench_one_hot_encode was: " << (float)cpu_time.count() / num_iter << " us" << std::endl;
}

void stress_test_mix_short_lived_long_lived_blocks() {
    std::vector<float> data;
    for (int i = 0; i < 2 * 3; i++) {
        data.push_back(i + 1);
    }

    std::vector<Matrix> results;
    for (int i = 0; i < 10000; i++) {
        // Register
        Matrix mat1 = register_matrix_with_data(&data[0], 2, 3);

        // Perform multiplication
        Matrix result = cuda_sum_rows(&mat1);
        results.push_back(result);
        decrease_matrix_ref_count(&mat1);
    }

    for (auto result : results) {
        decrease_matrix_ref_count(&result);
    }
}

void simulate_cnn_steps() {
    // Feed forward
}

int main() {
    // Get the temps and frequencies up
    warmup();

    // Used with ncu to profile kernels. Will expand to have all kernels, but for now just has the most time consuming ones

    const int mat_dim = 1024;
    const int kernel_dim = 28;
    const int num_iter = 2048;
    // bench_flatten_array(mat_dim, 256, num_iter);
    // bench_unflatten_array(mat_dim, 256, num_iter);
    // bench_img2col(mat_dim, num_iter, kernel_dim);
    // bench_convolution(mat_dim, num_iter, kernel_dim);
    // bench_multiple_correlations(32, num_iter, 1024, kernel_dim);
    bench_packed_correlation(30, num_iter, 65536, kernel_dim);  // Seems to be faster with matrices smaller than or eqal to 32x32, scaling at about 2x to 3x. Shows overhead of launches.
    // bench_convolution_2(mat_dim, num_iter, kernel_dim);
    // bench_rotate_180(mat_dim, num_iter);
    // bench_max_pool(mat_dim, num_iter);
    // bench_matrix_transpose(mat_dim, num_iter);
    // bench_matrix_multiply(mat_dim, num_iter);
    // bench_one_hot_encode(8, num_iter);
    // bench_non_square_matrix_multiply(10, 12544, 32, 1024);

    // Stress tests
    // stress_test_mix_short_lived_long_lived_blocks();
}