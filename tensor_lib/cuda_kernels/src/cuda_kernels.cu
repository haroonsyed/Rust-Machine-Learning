#include <unordered_map>
#include <vector>

#include "./cuda_kernels.cuh"

bool init_cublas = false;
bool init_pool = false;
cublasHandle_t handle;
size_t mat_generated_count(0);
std::unordered_map<size_t, float*> mat_map;

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
    printf("Finished Running Kernels.");
}

void test_array_fill(float* buffer, size_t length) {
    for (size_t i = 0; i < length; i++) {
        buffer[i] = i;
    }
}

void cuda_synchronize() {
    cudaDeviceSynchronize();
}

/////////////////////
/// Matrix Setup API
/////////////////////
void init_cublas_handle() {
    cublasCreate(&handle);
    cublasSetStream(handle, 0);
    init_cublas = true;
}
void init_min_pool_size() {
    int device;
    cudaGetDevice(&device);
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, device);
    size_t threshold = sizeof(float) * 2048 * 2048;  // Around 68 Mb reserved
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    init_pool = false;
}
size_t register_matrix_buffer(float* gpu_buffer) {
    if (init_pool) {
        init_min_pool_size();
    }

    // Register with the map for retrieval later
    mat_map[mat_generated_count] = gpu_buffer;
    return mat_generated_count++;  // Fine if this overflows
}

size_t register_matrix(size_t rows, size_t cols) {
    // Upload the data
    float* gpu_buffer;
    gpuErrchk(cudaMallocAsync(&gpu_buffer, sizeof(float) * rows * cols, 0));

    return register_matrix_buffer(gpu_buffer);
}

size_t register_matrix(float* data, size_t rows, size_t cols) {
    // Upload the data
    float* gpu_buffer;
    gpuErrchk(cudaMallocAsync(&gpu_buffer, sizeof(float) * rows * cols, 0));
    gpuErrchk(cudaMemcpy(gpu_buffer, data, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
    // Potentially nasty bug by acting like you copied data when you havent finished if using cudaMemCpyAsync...
    return register_matrix_buffer(gpu_buffer);
}

void unregister_matrix(size_t mat_id) {
    gpuErrchk(cudaFreeAsync(mat_map[mat_id], 0));
    mat_map.erase(mat_id);
}

void get_matrix_data(size_t mat_id, int rows, int cols, float* data_buffer) {
    float* gpu_buffer = mat_map[mat_id];
    gpuErrchk(cudaMemcpy(data_buffer, gpu_buffer, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
}

//////////////////////////
/// Matrix Operations API
//////////////////////////
/// TODO: Possibly grid stride. Optimize transpose. Optimize matmult.
__global__ void element_add_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;
    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] + mat2[i][j]
        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] + mat2_buffer[index];
    }
}
size_t cuda_element_add(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Create output buffer`
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);
    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_mat2_buffer = mat_map[mat2_id];
    float* gpu_out_buffer = mat_map[out_mat_id];
    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);
    // Run the kernels
    element_add_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
    // Return result matrix id
    return out_mat_id;
}

__global__ void element_subtract_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;
    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] - mat2[i][j]

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] - mat2_buffer[index];
    }
}

size_t cuda_element_subtract(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_mat2_buffer = mat_map[mat2_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    element_subtract_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_multiply_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] * mat2[i][j]

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] * mat2_buffer[index];
    }
}

size_t cuda_element_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_mat2_buffer = mat_map[mat2_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    element_multiply_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void scalar_multiply_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float scalar, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] * scalar

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] * scalar;
    }
}

size_t cuda_scalar_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, float scalar, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    scalar_multiply_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, scalar, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void matrix_multiply_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_rows && tidY < out_cols) {
        // O[i][j] = mat1[i][:] weighted sum mat2[:][j]
        // Where common dimension : is mat1col/mat2row

        float weighted_sum = 0.0;
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

__global__ void matrix_multiply_kernel_2(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    // Go by col row instead of row col. Enabled memory coalescing
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= out_rows || col >= out_cols) {
        return;
    }

    // O[i][j] = mat1[i][:] weighted sum mat2[:][j]
    // Where common dimension : is mat1col/mat2row

    float weighted_sum = 0.0;
    for (int common = 0; common < mat1_cols; common++) {
        // mat1[i][common]
        int mat1_index = mat1_cols * row + common;
        // mat1[common][j]
        int mat2_index = mat2_cols * common + col;

        weighted_sum += mat1_buffer[mat1_index] * mat2_buffer[mat2_index];
    }

    const int output_index = row * out_cols + col;
    out_buffer[output_index] = weighted_sum;
}

__global__ void matrix_multiply_kernel_3(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    const int block_dim = 32;
    const int block_area = block_dim * block_dim;

    // Block tiling with shared memory
    __shared__ float s_mat1[block_area];
    __shared__ float s_mat2[block_area];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    int mat1_block_pos = block_row * block_dim * mat1_cols;
    int mat2_block_pos = block_col * block_dim;
    int out_block_pos = block_row * block_dim * out_cols + block_col * block_dim;

    // So within our block we are gonna figure out this thread's position
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int out_row = block_row * block_dim + thread_row;
    int out_col = block_col * block_dim + thread_col;
    if (out_row >= out_rows || out_col >= out_cols) {
        return;
    }

    float weighted_sum = 0.0;
    int common_partial_block = mat1_cols % block_dim;
    int common_in_block = mat1_cols - common_partial_block;
    for (int k = 0; k < common_in_block; k += block_dim) {
        s_mat1[thread_row * block_dim + thread_col] = mat1_buffer[mat1_block_pos + thread_row * mat1_cols + thread_col];
        s_mat2[thread_row * block_dim + thread_col] = mat2_buffer[mat2_block_pos + thread_row * mat2_cols + thread_col];
        __syncthreads();

        mat1_block_pos += block_dim;
        mat2_block_pos += block_dim * mat2_cols;
        for (int i = 0; i < block_dim; i++) {
            weighted_sum += s_mat1[thread_row * block_dim + i] * s_mat2[i * block_dim + thread_col];
        }
        __syncthreads();
    }

    // Handle partial block case
    s_mat1[thread_row * block_dim + thread_col] = mat1_buffer[mat1_block_pos + thread_row * mat1_cols + thread_col];
    s_mat2[thread_row * block_dim + thread_col] = mat2_buffer[mat2_block_pos + thread_row * mat2_cols + thread_col];
    __syncthreads();

    mat1_block_pos += block_dim;
    mat2_block_pos += block_dim * mat2_cols;
    for (int i = 0; i < common_partial_block; i++) {
        weighted_sum += s_mat1[thread_row * block_dim + i] * s_mat2[i * block_dim + thread_col];
    }

    out_buffer[out_block_pos + (thread_row * out_cols) + thread_col] = weighted_sum;
}

// block_M is rows in mat1 shared block
// block_N is cols in mat2 shared block
// block_k is shared dimensions for shared block. Also the # of results each thread will compute in C
// For this to work we want the shared dimension block_K to be smaller than block_M and block_N
// This way, multiple threads reuse sections from mat1 and mat2 ,with more output work
// Example: bK is 8 while bM and bN are 64. Output is a 64x64 area.
//          So you can spin up 512 threads per block. They load vram->shared
//          Then each thread can work on 8 pieces of the output 64x64 area (64*64/8 = 512)
template <const int block_M, const int block_N, const int block_K>
__global__ void matrix_multiply_kernel_4(int M, int N, int K, float* mat1_buffer, float* mat2_buffer, float* out_buffer) {
    // Block tiling with shared memory
    // Each one of these threads will handle #block_K output result columns
    __shared__ float s_mat1[block_M * block_K];
    __shared__ float s_mat2[block_K * block_N];

    float thread_results[block_K] = {0.0};

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Get starting positions of each block
    int mat1_block_pos = block_row * block_M * K;
    int mat2_block_pos = block_col * block_N;
    int out_block_pos = block_row * block_M * N + block_col * block_N;

    // Used to track if out of bounds
    const int mat1_load_index_row = block_row * block_M + threadIdx.x;
    const int mat2_load_index_col = block_col * block_N + threadIdx.x;
    int mat_common_index = threadIdx.y;
    const bool exceeded_mat1_row = mat1_load_index_row >= M;
    const bool exceeded_mat2_col = mat2_load_index_col >= N;

    // outer loop over block tiles
    for (unsigned int common_block = 0; common_block < K; common_block += block_K) {
        const int within_mat1 = (int)!(exceeded_mat1_row || mat_common_index >= K);
        const int within_mat2 = (int)!(mat_common_index >= K || exceeded_mat2_col);
        int mat1_load_index = mat1_block_pos + threadIdx.x * K + threadIdx.y;
        int mat2_load_index = mat2_block_pos + threadIdx.y * N + threadIdx.x;

        // Prevent loading OOB
        mat1_load_index *= within_mat1;
        mat2_load_index *= within_mat2;

        // Load block data into shared memory. Load 0 is OOB.
        s_mat1[threadIdx.x * block_K + threadIdx.y] = mat1_buffer[mat1_load_index] * within_mat1;
        s_mat2[threadIdx.y * block_N + threadIdx.x] = mat2_buffer[mat2_load_index] * within_mat2;
        __syncthreads();

        // Advance block
        mat1_block_pos += block_K;
        mat2_block_pos += block_K * N;
        mat_common_index += block_K;

        // Go through common dimensions of block (across row of mat1 and down col of mat2)
        for (unsigned int block_common_index = 0; block_common_index < block_K; ++block_common_index) {
            const float shared_mat2_val = s_mat2[block_common_index * block_N + threadIdx.x];

            // Now this thread will accumulate the result for each t_row in the t_col of C
            for (unsigned int result_index = 0; result_index < block_K; ++result_index) {
                thread_results[result_index] +=
                    s_mat1[(threadIdx.y * block_K + result_index) * block_K + block_common_index] * shared_mat2_val;
            }
        }
        __syncthreads();
    }

    // Write results with bounds checking
    const int out_index_row = block_row * block_M + threadIdx.y * block_K;
    const int out_index_col = block_col * block_N + threadIdx.x;

    for (int i = 0; i < block_K; i++) {
        if (out_index_row + i < M && out_index_col < N) {
            out_buffer[out_block_pos + (threadIdx.y * block_K + i) * N + threadIdx.x] = thread_results[i];
        }
    }
}

// block_M is rows in mat1 shared block
// block_N is cols in mat2 shared block
// block_k is shared dimensions for shared block.
// The thread will calculate block_k * block_k results (So now a 2d version of v3)
// For this to work we want the shared dimension block_K to be extremely smaller than block_M and block_N
// This way, multiple threads reuse sections from mat1 and mat2 ,with more output work
// Example: bK is 8 while bM and bN are 128. Output is a 128x128 area.
//          So you can spin up 256 threads per block. They load vram->shared
//          Then each thread can work on 8x8 pieces of the output 128x128 area (128x128/64 = 256)
//          You might be wondering why not 512 threads like previously?
//          Well that increases the mem requirements per block, reducing occupancy.
template <const int block_M, const int block_N, const int block_K>
__global__ void matrix_multiply_kernel_5(int M, int N, int K, float* __restrict__ mat1_buffer, float* __restrict__ mat2_buffer, float* __restrict__ out_buffer) {
    // 2D Block tiling with shared memory
    __shared__ float s_mat1[block_M * block_K];
    __shared__ float s_mat2[block_K * block_N];

    float thread_results[block_K * block_K] = {0.0};

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Output within block details
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int out_block_row = tid / (block_M / block_K);
    const int out_block_col = tid % (block_N / block_K);

    const int num_threads_per_block = blockDim.x * blockDim.y;
    const int num_elements_to_load = (block_M * block_K) / num_threads_per_block;

    const int stride_mat1 = num_threads_per_block / block_K;
    const int stride_mat2 = num_threads_per_block / block_N;

    int mat1_pos = block_row * block_M * K;
    int mat2_pos = block_col * block_N;

// outer loop over block tiles
#pragma unroll
    for (int common_block = 0; common_block < K; common_block += block_K) {
#pragma unroll 4
        for (int i = 0; i < num_elements_to_load; i++) {
            const int mat1_row_within_block = (threadIdx.x + stride_mat1 * i);
            const int mat1_col_within_block = threadIdx.y;
            const int mat2_row_within_block = (threadIdx.y / num_elements_to_load) + i * stride_mat2;
            const int mat2_col_within_block = (threadIdx.y % num_elements_to_load) * blockDim.x + threadIdx.x;

            const int mat1_load_index_row = block_row * block_M + mat1_row_within_block;
            const int mat1_load_index_col = common_block + mat1_col_within_block;
            const int mat2_load_index_row = common_block + mat2_row_within_block;
            const int mat2_load_index_col = block_col * block_N + mat2_col_within_block;

            const bool exceeded_mat1_row = mat1_load_index_row >= M;
            const bool exceeded_mat1_col = mat1_load_index_col >= K;
            const bool exceeded_mat2_row = mat2_load_index_row >= K;
            const bool exceeded_mat2_col = mat2_load_index_col >= N;

            const int within_mat1 = (int)!(exceeded_mat1_row || exceeded_mat1_col);
            const int within_mat2 = (int)!(exceeded_mat2_row || exceeded_mat2_col);
            int mat1_load_index = mat1_pos + mat1_row_within_block * K + mat1_col_within_block;
            int mat2_load_index = mat2_pos + mat2_row_within_block * N + mat2_col_within_block;

            mat1_load_index *= within_mat1;
            mat2_load_index *= within_mat2;

            s_mat1[mat1_row_within_block * block_K + mat1_col_within_block] =
                mat1_buffer[mat1_load_index] * within_mat1;
            s_mat2[mat2_row_within_block * block_N + mat2_col_within_block] =
                mat2_buffer[mat2_load_index] * within_mat2;
        }

        mat1_pos += block_K;
        mat2_pos += block_K * N;

        __syncthreads();

        // Go through common dimensions of block (across row of mat1 and down col of mat2)
#pragma unroll 8
        for (int block_common_index = 0; block_common_index < block_K; block_common_index++) {
            // Now this thread will accumulate the block_K x block_K results from shared memory
#pragma unroll 8
            for (int result_index_row = 0; result_index_row < block_K; result_index_row++) {
#pragma unroll 8
                for (int result_index_col = 0; result_index_col < block_K; result_index_col++) {
                    thread_results[result_index_row * block_K + result_index_col] +=
                        s_mat1[(out_block_row * block_K + result_index_row) * block_K + block_common_index] *
                        s_mat2[(block_common_index * block_N) + (out_block_col * block_K + result_index_col)];
                }
            }
        }
        __syncthreads();
    }

    // Write results with bounds checking
    const int out_index_row = block_row * block_M + out_block_row * block_K;
    const int out_index_col = block_col * block_N + out_block_col * block_K;

#pragma unroll 8
    for (int i = 0; i < block_K; i++) {
#pragma unroll 8
        for (int j = 0; j < block_K; j++) {
            if (out_index_row + i < M && out_index_col + j < N) {
                out_buffer[(out_index_row + i) * N + out_index_col + j] = thread_results[i * block_K + j];
            }
        }
    }
}

size_t cuda_matrix_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat2_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_mat2_buffer = mat_map[mat2_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    // const int THREADS_PER_BLOCK_X = 32;
    // const int THREADS_PER_BLOCK_Y = 32;

    // dim3 block_dim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    // dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // // Run the kernels
    // matrix_multiply_kernel_3<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);

    // V4 launch
    const int M = mat1_rows;
    const int N = mat2_cols;
    const int K = mat1_cols;

    const int THREADS_PER_BLOCK_X = 32;
    const int THREADS_PER_BLOCK_Y = 8;

    dim3 block_dim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid_dim((N / 128) + 1, (M / 128) + 1, 1);
    matrix_multiply_kernel_5<128, 128, 8><<<grid_dim, block_dim>>>(M, N, K, gpu_mat1_buffer, gpu_mat2_buffer, gpu_out_buffer);

    // CUBLAS version (for comparison to mine)
    // if (!init_cublas) {
    //     init_cublas_handle();
    // }
    // float alpha = 1.0;
    // float beta = 0.0;
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mat2_cols, mat1_rows, mat1_cols, &alpha, gpu_mat2_buffer, mat2_cols, gpu_mat1_buffer, mat1_cols, &beta, gpu_out_buffer, mat2_cols);

    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void add_vector_to_columns_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] + mat2[i][0]

        int mat1_index = tidY * mat1_cols + tidX;
        int mat2_index = tidY;

        int output_index = mat1_index;
        out_buffer[output_index] = mat1_buffer[mat1_index] + mat2_buffer[mat2_index];
    }
}

__global__ void add_vector_to_rows_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] + mat2[0][j]

        int mat1_index = tidY * mat1_cols + tidX;
        int mat2_index = tidX;

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
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_mat2_buffer = mat_map[mat2_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

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

__global__ void divide_by_column_vector_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] / mat2[i][0]

        int mat1_index = tidY * mat1_cols + tidX;
        int mat2_index = tidY;

        int output_index = mat1_index;
        out_buffer[output_index] = mat1_buffer[mat1_index] / mat2_buffer[mat2_index];
    }
}

__global__ void divide_by_row_vector_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] / mat2[0][j]

        int mat1_index = tidY * mat1_cols + tidX;
        int mat2_index = tidX;

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
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_mat2_buffer = mat_map[mat2_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

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

__global__ void element_exp_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = exp(mat1[i][j])

        int index = tidY * out_cols + tidX;
        out_buffer[index] = exp(mat1_buffer[index]);  // Also available __exp for fast
    }
}

size_t cuda_element_exp(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    element_exp_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_ReLU_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = x if x>0 else 0

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] > 0 ? mat1_buffer[index] : 0.0;
    }
}

size_t cuda_element_ReLU(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    element_ReLU_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_ReLU_prime_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = x if x>0 else 1

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] == 0.0 ? 0.0 : 1.0;
    }
}

size_t cuda_element_ReLU_prime(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    element_ReLU_prime_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void sum_rows_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][0] = sum (mat1[i][:])

        float row_sum = 0.0;
        int mat1_row_start_index = tidY * mat1_cols;
        for (int i = 0; i < mat1_cols; i++) {
            int mat1_index = mat1_row_start_index + i;
            row_sum += mat1_buffer[mat1_index];
        }

        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = row_sum;
    }
}

size_t cuda_sum_rows(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = 1;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    sum_rows_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void sum_columns_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[0][j] = sum (mat1[:][j])

        float col_sum = 0.0;
        for (int i = 0; i < mat1_rows; i++) {
            int mat1_index = tidX + i * mat1_cols;
            col_sum += mat1_buffer[mat1_index];
        }

        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = col_sum;
    }
}

size_t cuda_sum_columns(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = 1;
    int out_cols = mat1_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    sum_columns_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void transpose_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[j][i]

        int mat1_index = tidX * mat1_cols + tidY;

        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = mat1_buffer[mat1_index];
    }
}

size_t cuda_transpose(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = mat1_cols;
    int out_cols = mat1_rows;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    transpose_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void cuda_max_pool_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols, float* gpu_max_bitmask) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // For each 2x2 area pick the maximum value
        // We will mem coalesce by getting first two in row 1
        // Then next 2 in row2

        // Grab data w/t bounds check
        // TODO: Bounds check

        int block_start_row = tidY * 2;
        int block_start_col = tidX * 2;
        int block_start = block_start_row * mat1_cols + block_start_col;

        // bool block_00_oob = false;
        bool block_01_oob = (block_start_col + 1) >= mat1_cols;
        bool block_10_oob = (block_start_row + 1) >= mat1_rows;
        bool block_11_oob = block_01_oob || block_10_oob;

        float small_float = -1e30;  // Should probably use FLT_MIN but language server no like it

        // TODO: Use bit operations instead of ternary (it's faster idk why the compiler can't figure it out)
        float block_00 = mat1_buffer[block_start];
        float block_01 = block_01_oob ? small_float : mat1_buffer[block_start + 1];
        block_start += mat1_cols;
        float block_10 = block_10_oob ? small_float : mat1_buffer[block_start];
        float block_11 = block_11_oob ? small_float : mat1_buffer[block_start + 1];

        float result = max(max(block_00, block_01), max(block_10, block_11));

        if (result == block_00) {
            gpu_max_bitmask[block_start_row * mat1_cols + block_start_col] = 1.0;
        } else if (result == block_01) {
            gpu_max_bitmask[block_start_row * mat1_cols + block_start_col + 1] = 1.0;
        } else if (result == block_10) {
            gpu_max_bitmask[(block_start_row + 1) * mat1_cols + block_start_col] = 1.0;
        } else if (result == block_11) {
            gpu_max_bitmask[(block_start_row + 1) * mat1_cols + block_start_col + 1] = 1.0;
        }

        // Write maxpool result
        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = result;
    }
}

// 2x2 since other reduction sizes are not really used
Tuple cuda_max_pool(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = mat1_rows / 2 + mat1_rows % 2;
    int out_cols = mat1_cols / 2 + mat1_cols % 2;
    size_t out_mat_id = register_matrix(out_rows, out_cols);
    size_t max_bitmask = register_matrix(mat1_rows, mat1_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];
    float* gpu_max_bitmask = mat_map[max_bitmask];

    // Zero out bitmask
    cudaMemset(gpu_max_bitmask, 0.0, mat1_rows * mat1_cols * sizeof(float));

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    cuda_max_pool_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols, gpu_max_bitmask);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return Tuple{out_mat_id, max_bitmask};
}

__global__ void cuda_nearest_neighbor_2x_upsample_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    // Upsample by nearest neighbor
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i/2][j/2]
        int mat1_index = (tidY / 2) * mat1_cols + (tidX / 2);

        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = mat1_buffer[mat1_index];
    }
}

// Odd upsample will leave out one row and one column from the upsampled matrix
size_t cuda_nearest_neighbor_2x_upsample(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool odd_upsample) {
    // Create output buffer
    int out_rows = mat1_rows * 2 - (int)odd_upsample;
    int out_cols = mat1_cols * 2 - (int)odd_upsample;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK_X = 16;
    const int THREADS_PER_BLOCK_Y = 16;

    dim3 block_dim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);
    cuda_nearest_neighbor_2x_upsample_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void cuda_rotate_180_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // Rotating an array 180 means
        // x_output = length - x_current
        // y_output = height - y_current
        int x_out = mat1_cols - tidX - 1;
        int y_out = mat1_rows - tidY - 1;
        float input = mat1_buffer[tidY * mat1_cols + tidX];

        int output_index = y_out * out_cols + x_out;
        out_buffer[output_index] = input;
    }
}

size_t cuda_rotate_180(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    cuda_rotate_180_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Naive implementation
__global__ void cuda_convolution_kernel_valid_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;
    int threadIdWithinBlock = threadIdx.y * blockDim.x + threadIdx.x;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = weighted sum of kernel with input, where kernel is kept within bounds of input
        float result = 0.0;
        const int kernel_top_left_row = tidY;
        const int kernel_top_left_col = tidX;

#pragma unroll 3
        for (int m = 0; m < kernel_rows; m++) {
#pragma unroll 3
            for (int n = 0; n < kernel_cols; n++) {
                const float mat1_val = mat1_buffer[(kernel_top_left_row + m) * mat1_cols + (kernel_top_left_col + n)];
                const float kernel_val = kernel_buffer[m * kernel_cols + n];
                result += mat1_val * kernel_val;
            }
        }

        const int out_index = tidY * out_cols + tidX;
        out_buffer[out_index] = result;
    }
}

// Be careful, this needs to be optimized or your CNN will suffer
size_t cuda_convolution_valid(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    // Dimension of output is input - kernel + 1
    int out_rows = mat1_rows - kernel_rows + 1;
    int out_cols = mat1_cols - kernel_cols + 1;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_kernel_buffer = mat_map[kernel_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    cuda_convolution_kernel_valid_1<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Naive implementation
__global__ void cuda_convolution_kernel_same_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = weighted sum of kernel with input, where kernel is centered at i,j

        float result = 0.0;
        const int apothem = kernel_rows / 2;
#pragma unroll 3
        for (int m = 0; m < kernel_rows; m++) {
#pragma unroll 3
            for (int n = 0; n < kernel_cols; n++) {
                int input_row = m - apothem + tidY;
                int input_col = n - apothem + tidX;
                bool input_row_in_bounds = input_row >= 0 && input_row < mat1_rows;
                bool input_col_in_bounds = input_col >= 0 && input_col < mat1_cols;

                if (input_row_in_bounds && input_col_in_bounds) {
                    const int curr_mat1_index = input_row * mat1_cols + input_col;
                    const int curr_kernel_index = m * kernel_cols + n;
                    result += mat1_buffer[curr_mat1_index] * kernel_buffer[curr_kernel_index];
                }
            }
        }

        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = result;
    }
}

// Convolution is zero-padded (Output is the same size as input)
// Expects odd size, square kernels ONLY
// Be careful, this needs to be optimized or your CNN will suffer
size_t cuda_convolution_same(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_kernel_buffer = mat_map[kernel_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    cuda_convolution_kernel_same_1<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Naive implementation
__global__ void cuda_convolution_kernel_full_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = weighted sum of kernel with input, where kernel is centered at i,j
        float result = 0.0;
        const int input_start_row = (-kernel_rows + 1) + tidY;
        const int input_start_col = (-kernel_cols + 1) + tidX;
        for (int m = 0; m < kernel_rows; m++) {
            for (int n = 0; n < kernel_cols; n++) {
                int input_row = input_start_row + m;
                int input_col = input_start_col + n;
                bool input_row_in_bounds = input_row >= 0 && input_row < mat1_rows;
                bool input_col_in_bounds = input_col >= 0 && input_col < mat1_cols;

                if (input_row_in_bounds && input_col_in_bounds) {
                    const int curr_mat1_index = input_row * mat1_cols + input_col;
                    const int curr_kernel_index = m * kernel_cols + n;
                    result += mat1_buffer[curr_mat1_index] * kernel_buffer[curr_kernel_index];
                }
            }
        }

        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = result;
    }
}

// Be careful, this needs to be optimized or your CNN will suffer
size_t cuda_convolution_full(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    // Dimension of output is input + kernel - 1
    int out_rows = mat1_rows + kernel_rows - 1;
    int out_cols = mat1_cols + kernel_cols - 1;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_kernel_buffer = mat_map[kernel_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    cuda_convolution_kernel_full_1<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

size_t cuda_convolution(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols, ConvolutionType conv_type) {
    if (conv_type == ConvolutionType::VALID) {
        return cuda_convolution_valid(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    } else if (conv_type == ConvolutionType::SAME) {
        return cuda_convolution_same(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    } else if (conv_type == ConvolutionType::FULL) {
        return cuda_convolution_full(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    }
}

__global__ void cuda_img2col_valid(float** mat_buffers, int input_depth, int input_rows, int input_cols, int filter_depth, int filter_rows, int filter_cols, float* out_buffer, int out_rows, int out_cols) {
    const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    // This thread will handle one patch of the image, through all the kernels
    // This means each thread handle one column of the output
    const int number_of_patches = out_cols;
    const int current_patch = tidX;

    if (current_patch < number_of_patches) {
// Go through each of the kernels
#pragma unroll
        for (int curr_channel = 0; curr_channel < input_depth; curr_channel++) {
            const float* current_buffer = mat_buffers[curr_channel];

            // Now translate current_patch into the patch's top left corner
            const int kernel_top_left_row = current_patch / (input_cols - filter_cols + 1);
            const int kernel_top_left_col = current_patch % (input_cols - filter_cols + 1);
            const int base_output_row = curr_channel * filter_rows * filter_cols;
            const int output_col = current_patch;

// Now construct the patch
#pragma unroll 3
            for (int m = 0; m < filter_rows; m++) {
#pragma unroll 3
                for (int n = 0; n < filter_cols; n++) {
                    const float mat_val = current_buffer[(kernel_top_left_row + m) * input_cols + (kernel_top_left_col + n)];
                    const int output_index = base_output_row + m * filter_cols + n;
                    out_buffer[output_index * out_cols + output_col] = mat_val;
                }
            }
        }
    }
}

size_t cuda_img2col_valid(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    const int kernel_count = num_matrices;  // num_matrices is the input depth
    const int out_rows = kernel_count * kernel_rows * kernel_cols;
    const int out_cols = (mat_rows - kernel_rows + 1) * (mat_cols - kernel_cols + 1);
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    std::vector<float*> mat_buffers;
    for (size_t i = 0; i < num_matrices; i++) {
        mat_buffers.push_back(mat_map[mat_ids[i]]);
    }

    float** gpu_mat_buffers;
    cudaMallocAsync(&gpu_mat_buffers, sizeof(float*) * num_matrices, 0);
    cudaMemcpy(gpu_mat_buffers, &mat_buffers[0], sizeof(float*) * num_matrices, cudaMemcpyHostToDevice);
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    // Let's have each threads handle its own patch between all the kernels
    // So we will calculate the number of patches == number of columns
    // Data access should be coalesced this way
    const int THREADS_PER_BLOCK = 1024;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, 1, 1);

    // Run the kernels
    cuda_img2col_valid<<<grid_dim, block_dim>>>(gpu_mat_buffers, num_matrices, mat_rows, mat_cols, kernel_count, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Cleanup
    cudaFreeAsync((void*)gpu_mat_buffers, 0);

    // Return result matrix id
    return out_mat_id;
}

// Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
size_t cuda_img2col(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t kernel_rows, size_t kernel_cols, ConvolutionType conv_type) {
    if (conv_type == ConvolutionType::VALID) {
        return cuda_img2col_valid(mat_ids, num_matrices, mat_rows, mat_cols, kernel_rows, kernel_cols);
    } else if (conv_type == ConvolutionType::SAME) {
        return 0;
    } else if (conv_type == ConvolutionType::FULL) {
        return 0;
    }
}

__global__ void cuda_flatten_array_kernel(float** mat_buffers, int mat_rows, int mat_cols, float* out_buffer, int out_rows, int out_cols) {
    const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidY = blockDim.y * blockIdx.y + threadIdx.y;
    const int output_index = tidY * out_cols + tidX;
    const int output_img_size = out_rows * out_cols;
    const int each_input_img_size = mat_rows * mat_cols;

    if (output_index < output_img_size) {
        // Grab the gpu buffer we are reffering to
        const int current_buffer_index = output_index / each_input_img_size;
        const float* current_buffer = mat_buffers[current_buffer_index];

        // Determine the pixel to copy
        const int current_buffer_pixel = output_index % each_input_img_size;

        // Write result
        out_buffer[output_index] = current_buffer[current_buffer_pixel];
    }
}

// Take n same_dimension matrices and flatten them into an array
size_t cuda_flatten_array(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = 1;
    int out_cols = num_matrices * mat_rows * mat_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    std::vector<float*> mat_buffers;
    for (size_t i = 0; i < num_matrices; i++) {
        mat_buffers.push_back(mat_map[mat_ids[i]]);
    }

    float** gpu_mat_buffers;
    cudaMallocAsync(&gpu_mat_buffers, sizeof(float*) * num_matrices, 0);
    cudaMemcpy(gpu_mat_buffers, &mat_buffers[0], sizeof(float*) * num_matrices, cudaMemcpyHostToDevice);
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    cuda_flatten_array_kernel<<<grid_dim, block_dim>>>(gpu_mat_buffers, mat_rows, mat_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Cleanup
    cudaFreeAsync((void*)gpu_mat_buffers, 0);

    // Return result matrix id
    return out_mat_id;
}

__global__ void cuda_unflatten_array_kernel(float* array_buffer, int arr_size, int mat_rows, int mat_cols, float** mat_buffers) {
    const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    const int array_buffer_index = tidX;

    if (array_buffer_index < arr_size) {
        // Check which mat_buffer to write to
        const int mat_size = mat_rows * mat_cols;
        const int mat_buffer_index = array_buffer_index / mat_size;
        const int mat_buffer_pixel = array_buffer_index % mat_size;

        // Write result
        mat_buffers[mat_buffer_index][mat_buffer_pixel] = array_buffer[array_buffer_index];
    }
}

// Take an array and unflatten it into n same_dimension matrices.
void cuda_unflatten_array(size_t array_id, size_t arr_size, size_t mat_rows, size_t mat_cols, size_t* mat_ids) {
    int mat_size = mat_rows * mat_cols;
    int num_matrices = arr_size / mat_size;

    // Create the buffers for the matrices
    std::vector<float*> gpu_mat_buffers;
    for (int i = 0; i < num_matrices; i++) {
        size_t mat_id = register_matrix(mat_rows, mat_cols);
        gpu_mat_buffers.push_back(mat_map[mat_id]);

        // Write back to rust vector
        mat_ids[i] = mat_id;
    }

    // Upload the gpu_mat_buffers to the gpu
    float** gpu_mat_buffers_ptr;
    cudaMallocAsync(&gpu_mat_buffers_ptr, sizeof(float*) * num_matrices, 0);
    cudaMemcpy(gpu_mat_buffers_ptr, &gpu_mat_buffers[0], sizeof(float*) * num_matrices, cudaMemcpyHostToDevice);

    // Get the flattened array
    float* gpu_array_buffer = mat_map[array_id];

    // Kernel launch parameters
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((arr_size / block_dim.x) + 1, 1, 1);

    // Run the kernels
    cuda_unflatten_array_kernel<<<grid_dim, block_dim>>>(gpu_array_buffer, arr_size, mat_rows, mat_cols, gpu_mat_buffers_ptr);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void cuda_unflatten_array_strided_kernel(float* array_buffer, int arr_size, int num_matrices, int mat_rows, int mat_cols, float** mat_buffers) {
    const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    const int array_buffer_index = tidX;

    if (array_buffer_index < arr_size) {
        // Check which mat_buffer to write to
        const int mat_buffer_index = array_buffer_index % num_matrices;
        const int mat_buffer_pixel = array_buffer_index / num_matrices;

        // Write result
        mat_buffers[mat_buffer_index][mat_buffer_pixel] = array_buffer[array_buffer_index];
    }
}

// Take an array and unflatten it into n same_dimension matrices. Each array's first n elements are the first elements in memory. [arr1_elem1, arr2_elem1, arr3_elem1, arr1_elem2, arr2_elem2, arr3_elem2, ...]
void cuda_unflatten_array_strided(size_t array_id, size_t arr_size, size_t mat_rows, size_t mat_cols, size_t* mat_ids) {
    int mat_size = mat_rows * mat_cols;
    int num_matrices = arr_size / mat_size;

    // Create the buffers for the matrices
    std::vector<float*> gpu_mat_buffers;
    for (int i = 0; i < num_matrices; i++) {
        size_t mat_id = register_matrix(mat_rows, mat_cols);
        gpu_mat_buffers.push_back(mat_map[mat_id]);

        // Write back to rust vector
        mat_ids[i] = mat_id;
    }

    // Upload the gpu_mat_buffers to the gpu
    float** gpu_mat_buffers_ptr;
    cudaMallocAsync(&gpu_mat_buffers_ptr, sizeof(float*) * num_matrices, 0);
    cudaMemcpy(gpu_mat_buffers_ptr, &gpu_mat_buffers[0], sizeof(float*) * num_matrices, cudaMemcpyHostToDevice);

    // Get the flattened array
    float* gpu_array_buffer = mat_map[array_id];

    // Kernel launch parameters
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((arr_size / block_dim.x) + 1, 1, 1);

    // Run the kernels
    cuda_unflatten_array_strided_kernel<<<grid_dim, block_dim>>>(gpu_array_buffer, arr_size, num_matrices, mat_rows, mat_cols, gpu_mat_buffers_ptr);
    gpuErrchk(cudaPeekAtLastError());
}