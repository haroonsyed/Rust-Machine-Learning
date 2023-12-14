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

// Used to return tuple with interop to rust
struct Tuple {
    size_t a;
    size_t b;
};

extern "C" {
// Memory Manager Functions
void* get_block_gpu_address(size_t block_id, size_t block_offset);
size_t memory_manager_device_allocate(size_t size);
void memory_manager_free(size_t block_id, size_t size);
void memory_manager_upload_to_allocation(void* address, void* data, size_t size);
void memory_manager_upload_from_pinned_buffer(void* device_address, void* pinned_data, size_t size);
void memory_manager_upload_async_from_pinned_buffer(void* device_address, void* pinned_data, size_t size);
void* memory_manager_get_pinned_allocation(size_t size);
std::vector<void*> get_device_kernel_args_pointers(size_t num_buffers);

// Matrix Setup API
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

// Test Functions
void test();
void test_array_fill(float* buffer, size_t length);

// Misc
void cuda_synchronize();

// Matrix operation API
size_t cuda_element_add(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
void cuda_element_add_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
void cuda_element_add_packed_inplace(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_element_subtract(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
void cuda_element_subtract_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
void cuda_element_subtract_packed_inplace(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_element_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
void cuda_element_multiply_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
void cuda_element_multiply_packed_inplace(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_element_divide(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
void cuda_element_divide_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
void cuda_element_divide_packed_inplace(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_scalar_multiply(size_t mat_id, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
void cuda_scalar_multiply_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar);
void cuda_scalar_multiply_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar);
size_t cuda_scalar_divide(size_t mat_id, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
void cuda_scalar_divide_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar);
void cuda_scalar_divide_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar);
size_t cuda_scalar_add(size_t mat_id, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
void cuda_scalar_add_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar);
void cuda_scalar_add_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar);
size_t cuda_scalar_subtract(size_t mat_id, size_t mat_rows, size_t mat_cols, float scalar, bool inplace);
void cuda_scalar_subtract_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar);
void cuda_scalar_subtract_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar);
size_t cuda_matrix_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols);
size_t cuda_add_vector(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_divide_by_vector(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace);
size_t cuda_element_sqrt(size_t mat_id, size_t mat_rows, size_t mat_cols, bool inplace);
void cuda_element_sqrt_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
void cuda_element_sqrt_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_element_exp(size_t mat_id, size_t mat_rows, size_t mat_cols, bool inplace);
void cuda_element_exp_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
void cuda_element_exp_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_element_ReLU(size_t mat_id, size_t mat_rows, size_t mat_col, bool inplace);
void cuda_element_ReLU_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
void cuda_element_ReLU_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_element_ReLU_prime(size_t mat_id, size_t mat_rows, size_t mat_cols, bool inplace);
void cuda_element_ReLU_prime_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
void cuda_element_ReLU_prime_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_sum_rows(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_sum_columns(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_transpose(size_t mat_id, size_t mat_rows, size_t mat_cols);
Tuple cuda_max_pool(size_t mat_id, size_t mat_rows, size_t mat_cols);
void cuda_max_pool_packed(size_t* mat_ids, Tuple* out_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_nearest_neighbor_2x_upsample(size_t mat_id, size_t mat_rows, size_t mat_cols, bool odd_upsample);
void cuda_nearest_neighbor_2x_upsample_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool odd_upsample);
size_t cuda_rotate_180(size_t mat_id, size_t mat_rows, size_t mat_cols);
void cuda_rotate_180_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);
size_t cuda_correlate(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols, PaddingType padding_type);
void cuda_correlate_packed(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_ids, PaddingType padding_type);
size_t cuda_convolve(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols, PaddingType padding_type);
void cuda_convolve_packed(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_ids, PaddingType padding_type);
size_t cuda_img2col(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t kernel_rows, size_t kernel_cols, PaddingType padding_type);  // Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
size_t cuda_flatten_array(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols);                                                              // Take n same_dimension matrices and flatten them into an array
void cuda_unflatten_array(size_t array_id, size_t arr_size, size_t mat_rows, size_t mat_cols, size_t* mat_ids);                                                 // Take an array and unflatten it into n same_dimension matrices
void cuda_unflatten_array_strided(size_t array_id, size_t arr_size, size_t mat_rows, size_t mat_cols, size_t* mat_ids);                                         // Take an array and unflatten it into n same_dimension matrices. Each array's first n elements are the first elements in memory. [arr1_elem1, arr2_elem1, arr3_elem1, arr1_elem2, arr2_elem2, arr3_elem2, ...]
size_t cuda_center_pad(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t pad_rows, size_t pad_cols);
size_t cuda_softmax(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_crop(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t crop_offset_rows, size_t crop_offset_cols, size_t crop_rows, size_t crop_cols);
size_t cuda_copy(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_sum_all_matrix_elements(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_max_by_column(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_max_by_row(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_argmax_by_column(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_argmax_by_row(size_t mat_id, size_t mat_rows, size_t mat_cols);
size_t cuda_one_hot_encode(float* data, size_t data_size, size_t num_classes);
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
        decrease_matrix_ref_count(result_id);
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
        decrease_matrix_ref_count(result_id);
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
        decrease_matrix_ref_count(result.a);
        decrease_matrix_ref_count(result.b);
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
        decrease_matrix_ref_count(result_id);
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
        int result_id = cuda_correlate(mat1, mat_dim, mat_dim, kernel, kernel_size, kernel_size, VALID);
        decrease_matrix_ref_count(result_id);
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_convolution was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
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
    int mat1 = register_matrix_with_data(&data[0], mat_dim, mat_dim);
    int kernel = register_matrix_with_data(&kernel_data[0], kernel_size, kernel_size);

    auto start_host = high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
        for (int i = 0; i < num_matrices; i++) {
            int result_id = cuda_correlate(mat1, mat_dim, mat_dim, kernel, kernel_size, kernel_size, VALID);
            decrease_matrix_ref_count(result_id);
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
        cuda_correlate_packed(&mat_ids[0], num_matrices, mat_dim, mat_dim, &kernel_ids[0], kernel_size, kernel_size, &result_ids[0], VALID);

        // Unregister matrices
        for (auto id : result_ids) {
            decrease_matrix_ref_count(id);
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

        decrease_matrix_ref_count(mat1_img2col);
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
        decrease_matrix_ref_count(kernel_flat);
        decrease_matrix_ref_count(mat1_img2col);
        decrease_matrix_ref_count(result_id);
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
        decrease_matrix_ref_count(result_id);
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
        decrease_matrix_ref_count(result_id);
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
        cuda_unflatten_array(mat, mat_count * mat_dim * mat_dim, mat_dim, mat_dim, &unflattened[0]);
        for (auto id : unflattened) {
            decrease_matrix_ref_count(id);
        }
    }
    cuda_synchronize();

    float gpu_time = 0;

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Including overhead bench_unflatten_array was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;
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
        size_t result = cuda_one_hot_encode(&data[0], num_labels, num_labels);
        decrease_matrix_ref_count(result);
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

    std::vector<size_t> results;
    for (int i = 0; i < 10000; i++) {
        // Register
        int mat1 = register_matrix_with_data(&data[0], 2, 3);

        // Perform multiplication
        int result_id = cuda_sum_rows(mat1, 2, 3);
        decrease_matrix_ref_count(mat1);
    }

    for (auto id : results) {
        decrease_matrix_ref_count(id);
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
    const int kernel_dim = 30;
    const int num_iter = 1024;
    // bench_flatten_array(mat_dim, 256, num_iter);
    // bench_unflatten_array(mat_dim, 256, num_iter);
    // bench_img2col(mat_dim, num_iter, kernel_dim);
    // bench_convolution(mat_dim, num_iter, kernel_dim);
    // bench_multiple_correlations(32, num_iter, 1024, kernel_dim);
    bench_packed_correlation(32, num_iter, 65536, kernel_dim);  // Seems to be faster with matrices smaller than or eqal to 32x32, scaling at about 2x to 3x. Shows overhead of launches.
    // bench_convolution_2(mat_dim, num_iter, kernel_dim);
    // bench_rotate_180(mat_dim, num_iter);
    // bench_max_pool(mat_dim, num_iter);
    // bench_matrix_transpose(mat_dim, num_iter);
    // bench_matrix_multiply(mat_dim, num_iter);
    // bench_one_hot_encode(8, num_iter);

    // Stress tests
    // stress_test_mix_short_lived_long_lived_blocks();
}