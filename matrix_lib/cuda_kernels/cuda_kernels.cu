#include "./cuda_kernels.cuh"

// Error checking macro: https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/////////////////////
/// TEST FUNCTIONS
/////////////////////
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
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
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

/////////////////////
/// Matrix Setup API
/////////////////////
void init_min_pool_size() {
    int device;
    cudaGetDevice(&device);
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, device);
    size_t threshold = sizeof(double) * 2048 * 2048;  // Around 68 Mb reserved
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    init_pool = false;
}
size_t register_matrix_buffer(double* gpu_buffer) {
    if (init_pool) {
        init_min_pool_size();
    }

    // Register with the map for retrieval later
    mat_map[mat_generated_count] = gpu_buffer;
    return mat_generated_count++;  // Fine if this overflows
}

size_t register_matrix(size_t rows, size_t cols) {
    // Upload the data
    double* gpu_buffer;
    gpuErrchk(cudaMallocAsync(&gpu_buffer, sizeof(double) * rows * cols, 0));

    return register_matrix_buffer(gpu_buffer);
}

size_t register_matrix(double* data, size_t rows, size_t cols) {
    // Upload the data
    double* gpu_buffer;
    gpuErrchk(cudaMallocAsync(&gpu_buffer, sizeof(double) * rows * cols, 0));
    gpuErrchk(cudaMemcpy(gpu_buffer, data, sizeof(double) * rows * cols, cudaMemcpyHostToDevice));
    // Potentially nasty bug by acting like you copied data when you havent finished if using cudaMemCpyAsync...
    return register_matrix_buffer(gpu_buffer);
}

void unregister_matrix(size_t mat_id) {
    gpuErrchk(cudaFreeAsync(mat_map[mat_id], 0));
    mat_map.erase(mat_id);
}

void get_matrix_data(size_t mat_id, int rows, int cols, double* data_buffer) {
    double* gpu_buffer = mat_map[mat_id];
    gpuErrchk(cudaMemcpy(data_buffer, gpu_buffer, sizeof(double) * rows * cols, cudaMemcpyDeviceToHost));
}

//////////////////////////
/// Matrix Operations API
//////////////////////////
/// For now everything is naive implementations to pass tests
/// TODO: Optimize memory accesses for coalition (tidX is problematic since it accesses down rows. Try to have block handle data sequentially). Possibly grid stride too.
__global__ void element_add_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* mat2_buffer, int mat2_rows, int mat2_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][j] + mat2[i][j]

        int index = tidX * out_cols + tidY;
        out_buffer[index] = mat1_buffer[index] + mat2_buffer[index];
    }
}

size_t cuda_element_add(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_mat2_buffer = mat_map[mat2_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    element_add_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_subtract_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* mat2_buffer, int mat2_rows, int mat2_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][j] - mat2[i][j]

        int index = tidX * out_cols + tidY;
        out_buffer[index] = mat1_buffer[index] - mat2_buffer[index];
    }
}

size_t cuda_element_subtract(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_mat2_buffer = mat_map[mat2_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    element_subtract_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_multiply_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* mat2_buffer, int mat2_rows, int mat2_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][j] * mat2[i][j]

        int index = tidX * out_cols + tidY;
        out_buffer[index] = mat1_buffer[index] * mat2_buffer[index];
    }
}

size_t cuda_element_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_mat2_buffer = mat_map[mat2_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    element_multiply_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void scalar_multiply_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double scalar, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][j] * scalar

        int index = tidX * out_cols + tidY;
        out_buffer[index] = mat1_buffer[index] * scalar;
    }
}

size_t cuda_scalar_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, double scalar, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    scalar_multiply_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, scalar, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
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

size_t cuda_matrix_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat2_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_mat2_buffer = mat_map[mat2_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    matrix_multiply_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void add_vector_to_columns_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* mat2_buffer, int mat2_rows, int mat2_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][j] + mat2[i][0]

        int mat1_index = tidX * mat1_cols + tidY;
        int mat2_index = tidX;

        int output_index = mat1_index;
        out_buffer[output_index] = mat1_buffer[mat1_index] + mat2_buffer[mat2_index];
    }
}

__global__ void add_vector_to_rows_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* mat2_buffer, int mat2_rows, int mat2_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][j] + mat2[0][j]

        int mat1_index = tidX * mat1_cols + tidY;
        int mat2_index = tidY;

        int output_index = mat1_index;
        out_buffer[output_index] = mat1_buffer[mat1_index] + mat2_buffer[mat2_index];
    }
}

size_t cuda_add_vector(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Determine orientation
    bool is_column_vector = (mat2_cols == 1 && mat2_rows == mat1_rows);

    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_mat2_buffer = mat_map[mat2_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    if (is_column_vector) {
        add_vector_to_columns_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    } else {
        add_vector_to_rows_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    }
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void divide_by_column_vector_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* mat2_buffer, int mat2_rows, int mat2_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][j] / mat2[i][0]

        int mat1_index = tidX * mat1_cols + tidY;
        int mat2_index = tidX;

        int output_index = mat1_index;
        out_buffer[output_index] = mat1_buffer[mat1_index] / mat2_buffer[mat2_index];
    }
}

__global__ void divide_by_row_vector_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* mat2_buffer, int mat2_rows, int mat2_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][j] / mat2[0][j]

        int mat1_index = tidX * mat1_cols + tidY;
        int mat2_index = tidY;

        int output_index = mat1_index;
        out_buffer[output_index] = mat1_buffer[mat1_index] / mat2_buffer[mat2_index];
    }
}

size_t cuda_divide_by_vector(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Determine orientation
    bool is_column_vector = (mat2_cols == 1 && mat2_rows == mat1_rows);

    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_mat2_buffer = mat_map[mat2_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    if (is_column_vector) {
        divide_by_column_vector_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    } else {
        divide_by_row_vector_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    }
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_exp_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = exp(mat1[i][j])

        int index = tidX * out_cols + tidY;
        out_buffer[index] = exp(mat1_buffer[index]);  // Also available __exp for fast
    }
}

size_t cuda_element_exp(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    element_exp_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_ReLU_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = x if x>0 else 0

        int index = tidX * out_cols + tidY;
        out_buffer[index] = mat1_buffer[index] > 0 ? mat1_buffer[index] : 0.0;
    }
}

size_t cuda_element_ReLU(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    element_ReLU_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_ReLU_prime_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = x if x>0 else 1

        int index = tidX * out_cols + tidY;
        out_buffer[index] = mat1_buffer[index] == 0.0 ? 0.0 : 1.0;
    }
}

size_t cuda_element_ReLU_prime(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    element_ReLU_prime_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void sum_rows_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][0] = sum (mat1[i][:])

        double row_sum = 0.0;
        int mat1_row_start_index = tidX * mat1_cols;
        for (int i = 0; i < mat1_cols; i++) {
            int mat1_index = mat1_row_start_index + i;
            row_sum += mat1_buffer[mat1_index];
        }

        int output_index = tidX * out_cols + tidY;
        out_buffer[output_index] = row_sum;
    }
}

size_t cuda_sum_rows(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = 1;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    sum_rows_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void sum_columns_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[0][j] = sum (mat1[:][j])

        double col_sum = 0.0;
        for (int i = 0; i < mat1_rows; i++) {
            int mat1_index = tidY + i * mat1_cols;
            col_sum += mat1_buffer[mat1_index];
        }

        int output_index = tidX * out_cols + tidY;
        out_buffer[output_index] = col_sum;
    }
}

size_t cuda_sum_columns(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = 1;
    int out_cols = mat1_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    sum_columns_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void transpose_kernel(double* mat1_buffer, int mat1_rows, int mat1_cols, double* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[j][i]

        int mat1_index = tidY * mat1_cols + tidX;

        int output_index = tidX * out_cols + tidY;
        out_buffer[output_index] = mat1_buffer[mat1_index];
    }
}

size_t cuda_transpose(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = mat1_cols;
    int out_cols = mat1_rows;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    double* gpu_mat1_buffer = mat_map[mat1_id];
    double* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_rows / block_dim.x) + 1, (out_cols / block_dim.y) + 1, 1);

    // Run the kernels
    transpose_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}