#include "./cuda_kernels.cuh"

__global__ void test_kernel() {
    printf("Hello from the kernel!\n");
}

__global__ void test_kernel_2(int* result) {
    *result = 8;
}

void test() {
    int result;
    int* d_result;
    test_kernel<<<1, 1>>>();
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(cudaGetLastError()));
    }

    cudaError_t err = cudaMalloc((void**)&d_result, sizeof(int));
    if (err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }

    test_kernel_2<<<1, 1>>>(d_result);
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(cudaGetLastError()));
    }

    err = cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaFree(d_result);

    printf("Result: %d\n", result);
    std::cout << "Finished Running Kernels." << std::endl;
}

void test_array_fill(double* buffer, size_t length) {
    for (int i = 0; i < length; i++) {
        buffer[i] = i;
    }
}

__global__ void matrix_multiply_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* mat2_buffer, int mat2_rows, int mat2_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][:] weighted sum mat2[:][j]
        // Where common dimension : is mat1col/mat2row

        double weighted_sum = 0.0;
        for (int common = 0; common < mat1_cols; common++) {
            // mat1[i][common]
            int mat1_index = mat1_cols * tidX + common;
            // mat1[common][j]
            int mat2_index = mat2_cols * common + tidY;

            weighted_sum += mat1_buffer[mat1_index] * mat2_buffer[mat2_index];
        }

        int output_index = tidX * out_cols + tidY;
        out_buffer[output_index] = weighted_sum;
    }
}
void cuda_matrix_multiply(double* mat1_buffer, size_t mat1_rows, size_t mat1_cols, double* mat2_buffer, size_t mat2_rows, size_t mat2_cols, double* out_buffer, size_t out_rows, size_t out_cols) {
    // Setup the cuda buffers
    double* gpu_mat1_buffer;
    double* gpu_mat2_buffer;
    double* gpu_out_buffer;
    cudaMalloc(&gpu_mat1_buffer, sizeof(double) * mat1_rows * mat1_cols);
    cudaMalloc(&gpu_mat2_buffer, sizeof(double) * mat2_rows * mat2_cols);
    cudaMalloc(&gpu_out_buffer, sizeof(double) * out_rows * out_cols);

    // Upload input data
    cudaMemcpy(gpu_mat1_buffer, mat1_buffer, sizeof(double) * mat1_rows * mat1_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mat2_buffer, mat2_buffer, sizeof(double) * mat2_rows * mat2_cols, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    matrix_multiply_kernel<<<block_dim, grid_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);

    // Download results to output
    cudaMemcpy(out_buffer, gpu_out_buffer, sizeof(double) * out_rows * out_cols, cudaMemcpyDeviceToHost);

    // Free resources
    cudaFree(gpu_mat1_buffer);
    cudaFree(gpu_mat2_buffer);
    cudaFree(gpu_out_buffer);
}