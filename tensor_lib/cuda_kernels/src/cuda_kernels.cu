#include <chrono>

#include "cuda_kernels.cuh"

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

//////////////////////////
/// Device Functions
//////////////////////////
__device__ float atomicMultiply(float* address, float val) {
    // We will need to use atomicCAS, since there is not a built in
    float expected_old = *address;
    float actual_old = __int_as_float(atomicCAS((int*)address, __float_as_int(expected_old), __float_as_int(expected_old * val)));
    while (expected_old != actual_old && !__isnanf(expected_old)) {
        expected_old = actual_old;
        actual_old = __int_as_float(atomicCAS((int*)address, __float_as_int(expected_old), __float_as_int(expected_old * val)));
    }
}

__device__ float atomicDivide(float* address, float val) {
    // We will need to use atomicCAS, since there is not a built in
    float expected_old = *address;
    float actual_old = __int_as_float(atomicCAS((int*)address, __float_as_int(expected_old), __float_as_int(expected_old / val)));
    while (expected_old != actual_old && !__isnanf(expected_old)) {
        expected_old = actual_old;
        actual_old = __int_as_float(atomicCAS((int*)address, __float_as_int(expected_old), __float_as_int(expected_old / val)));
    }
}

//////////////////////////
/// Matrix Operations API
//////////////////////////
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_mat2_buffer = get_matrix_gpu_address(mat2_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);
    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);
    // Run the kernels
    element_add_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
    // Return result matrix id
    return out_mat_id;
}
// Each block handles one matrix
__global__ void cuda_element_add_packed_kernel(float** mat1_buffers, float** mat2_buffers, float** out_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat1_buffer = mat1_buffers[current_matrix];
    const float* mat2_buffer = mat2_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat1_buffer[index] + mat2_buffer[index];
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
// Each block handles one matrix
__global__ void cuda_element_add_packed_inplace_kernel(float** mat1_buffers, float** mat2_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    float* mat1_buffer = mat1_buffers[current_matrix];
    const float* mat2_buffer = mat2_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;

            // Atomic because mat1 may be used multiple times
            atomicAdd(&mat1_buffer[index], mat2_buffer[index]);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
void cuda_element_add_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_mat2_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat1_ids[i]);
        pinned_mat2_buffers_ptr[i] = get_matrix_gpu_address(mat2_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_mat2_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_mat2_buffers_dp, pinned_mat2_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_add_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_mat2_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_element_add_packed_inplace(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Get the gpu buffers to operate on
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_mat2_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat1_ids[i]);
        pinned_mat2_buffers_ptr[i] = get_matrix_gpu_address(mat2_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_mat2_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_mat2_buffers_dp, pinned_mat2_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_add_packed_inplace_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_mat2_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_mat2_buffer = get_matrix_gpu_address(mat2_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    element_subtract_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_element_subtract_packed_kernel(float** mat1_buffers, float** mat2_buffers, float** out_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat1_buffer = mat1_buffers[current_matrix];
    const float* mat2_buffer = mat2_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat1_buffer[index] - mat2_buffer[index];
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
// Each block handles one matrix
__global__ void cuda_element_subtract_packed_inplace_kernel(float** mat1_buffers, float** mat2_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    float* mat1_buffer = mat1_buffers[current_matrix];
    const float* mat2_buffer = mat2_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;

            // Atomic because mat1 may be used multiple times
            atomicAdd(&mat1_buffer[index], -mat2_buffer[index]);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
void cuda_element_subtract_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_mat2_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat1_ids[i]);
        pinned_mat2_buffers_ptr[i] = get_matrix_gpu_address(mat2_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_mat2_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_mat2_buffers_dp, pinned_mat2_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_subtract_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_mat2_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_element_subtract_packed_inplace(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Get the gpu buffers to operate on
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_mat2_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat1_ids[i]);
        pinned_mat2_buffers_ptr[i] = get_matrix_gpu_address(mat2_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_mat2_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_mat2_buffers_dp, pinned_mat2_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_subtract_packed_inplace_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_mat2_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_mat2_buffer = get_matrix_gpu_address(mat2_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    element_multiply_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_element_multiply_packed_kernel(float** mat1_buffers, float** mat2_buffers, float** out_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat1_buffer = mat1_buffers[current_matrix];
    const float* mat2_buffer = mat2_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat1_buffer[index] * mat2_buffer[index];
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
// Each block handles one matrix
__global__ void cuda_element_multiply_packed_inplace_kernel(float** mat1_buffers, float** mat2_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    float* mat1_buffer = mat1_buffers[current_matrix];
    const float* mat2_buffer = mat2_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;

            // Atomic because mat1 may be used multiple times
            atomicMultiply(&mat1_buffer[index], mat2_buffer[index]);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
void cuda_element_multiply_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);
    // Get the gpu buffers to operate on
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_mat2_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat1_ids[i]);
        pinned_mat2_buffers_ptr[i] = get_matrix_gpu_address(mat2_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_mat2_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_mat2_buffers_dp, pinned_mat2_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_multiply_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_mat2_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_element_multiply_packed_inplace(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Get the gpu buffers to operate on
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_mat2_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat1_ids[i]);
        pinned_mat2_buffers_ptr[i] = get_matrix_gpu_address(mat2_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_mat2_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_mat2_buffers_dp, pinned_mat2_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_multiply_packed_inplace_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_mat2_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void element_divide_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] * mat2[i][j]

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] / mat2_buffer[index];
    }
}

size_t cuda_element_divide(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_mat2_buffer = get_matrix_gpu_address(mat2_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    element_divide_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_element_divide_packed_kernel(float** mat1_buffers, float** mat2_buffers, float** out_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat1_buffer = mat1_buffers[current_matrix];
    const float* mat2_buffer = mat2_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat1_buffer[index] / mat2_buffer[index];
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
// Each block handles one matrix
__global__ void cuda_element_divide_packed_inplace_kernel(float** mat1_buffers, float** mat2_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    float* mat1_buffer = mat1_buffers[current_matrix];
    const float* mat2_buffer = mat2_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;

            // Atomic because mat1 may be used multiple times
            atomicDivide(&mat1_buffer[index], mat2_buffer[index]);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
void cuda_element_divide_packed(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_mat2_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat1_ids[i]);
        pinned_mat2_buffers_ptr[i] = get_matrix_gpu_address(mat2_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_mat2_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_mat2_buffers_dp, pinned_mat2_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_divide_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_mat2_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_element_divide_packed_inplace(size_t* mat1_ids, size_t* mat2_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Get the gpu buffers to operate on
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_mat2_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat1_ids[i]);
        pinned_mat2_buffers_ptr[i] = get_matrix_gpu_address(mat2_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_mat2_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_mat2_buffers_dp, pinned_mat2_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_divide_packed_inplace_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_mat2_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    scalar_multiply_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, scalar, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_scalar_multiply_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols, float scalar) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat_buffer[index] * scalar;
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
// Each block handles one matrix
__global__ void cuda_scalar_multiply_packed_inplace_kernel(float** mat_buffers, int mat_rows, int mat_cols, float scalar) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    float* mat_buffer = mat_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;

            // Atomic because mat1 may be used multiple times
            atomicMultiply(&mat_buffer[index], scalar);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
void cuda_scalar_multiply_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_scalar_multiply_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols, scalar);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_scalar_multiply_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_scalar_multiply_packed_inplace_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, mat_rows, mat_cols, scalar);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void scalar_divide_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float scalar, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] * scalar

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] / scalar;
    }
}

size_t cuda_scalar_divide(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, float scalar, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    scalar_divide_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, scalar, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_scalar_divide_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols, float scalar) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat_buffer[index] / scalar;
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
// Each block handles one matrix
__global__ void cuda_scalar_divide_packed_inplace_kernel(float** mat_buffers, int mat_rows, int mat_cols, float scalar) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    float* mat_buffer = mat_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;

            // Atomic because mat1 may be used multiple times
            atomicDivide(&mat_buffer[index], scalar);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
void cuda_scalar_divide_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_scalar_divide_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols, scalar);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_scalar_divide_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_scalar_divide_packed_inplace_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, mat_rows, mat_cols, scalar);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void scalar_add_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float scalar, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] * scalar

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] + scalar;
    }
}

size_t cuda_scalar_add(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, float scalar, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    scalar_add_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, scalar, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_scalar_add_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols, float scalar) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat_buffer[index] + scalar;
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
// Each block handles one matrix
__global__ void cuda_scalar_add_packed_inplace_kernel(float** mat_buffers, int mat_rows, int mat_cols, float scalar) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    float* mat_buffer = mat_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;

            // Atomic because mat1 may be used multiple times
            atomicAdd(&mat_buffer[index], scalar);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
void cuda_scalar_add_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_scalar_add_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols, scalar);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_scalar_add_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_scalar_add_packed_inplace_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, mat_rows, mat_cols, scalar);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void scalar_subtract_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float scalar, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = mat1[i][j] * scalar

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] - scalar;
    }
}

size_t cuda_scalar_subtract(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, float scalar, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    scalar_subtract_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, scalar, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_scalar_subtract_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols, float scalar) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat_buffer[index] - scalar;
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
// Each block handles one matrix
__global__ void cuda_scalar_subtract_packed_inplace_kernel(float** mat_buffers, int mat_rows, int mat_cols, float scalar) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    float* mat_buffer = mat_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;

            // Atomic because mat1 may be used multiple times
            atomicAdd(&mat_buffer[index], -scalar);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}
void cuda_scalar_subtract_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_scalar_subtract_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols, scalar);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_scalar_subtract_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, float scalar) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat1_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat1_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat1_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat1_buffers_dp, pinned_mat1_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_scalar_subtract_packed_inplace_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffers_dp, mat_rows, mat_cols, scalar);
    gpuErrchk(cudaPeekAtLastError());
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_mat2_buffer = get_matrix_gpu_address(mat2_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    // const int THREADS_PER_BLOCK_X = 32;
    // const int THREADS_PER_BLOCK_Y = 32;

    // dim3 block_dim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    // dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // // Run the kernels
    // matrix_multiply_kernel_3<<<grid_dim, block_di>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);

    // V4 launch
    const int M = mat1_rows;
    const int N = mat2_cols;
    const int K = mat1_cols;

    const int THREADS_PER_BLOCK_X = 32;
    const int THREADS_PER_BLOCK_Y = 8;

    dim3 block_dim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid_dim((N + 128 - 1) / 128, (M + 128 - 1) / 128, 1);
    matrix_multiply_kernel_5<128, 128, 8><<<grid_dim, block_dim, 0, get_stream()>>>(M, N, K, gpu_mat1_buffer, gpu_mat2_buffer, gpu_out_buffer);

    // CUBLAS version (for comparison to mine)
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_mat2_buffer = get_matrix_gpu_address(mat2_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    if (is_column_vector) {
        add_vector_to_columns_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    } else {
        add_vector_to_rows_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_mat2_buffer = get_matrix_gpu_address(mat2_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    if (is_column_vector) {
        divide_by_column_vector_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    } else {
        divide_by_row_vector_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);
    }
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void element_sqrt_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = sqrt(mat1[i][j])

        int index = tidY * out_cols + tidX;
        out_buffer[index] = sqrt(mat1_buffer[index]);
    }
}

size_t cuda_element_sqrt(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    element_sqrt_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_element_sqrt_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = sqrt(mat_buffer[index]);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

void cuda_element_sqrt_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_sqrt_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_element_sqrt_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_sqrt_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_mat_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void element_exp_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = exp(mat1[i][j])

        int index = tidY * out_cols + tidX;
        out_buffer[index] = exp(mat1_buffer[index]);
    }
}

size_t cuda_element_exp(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    element_exp_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_element_exp_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = exp(mat_buffer[index]);
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

void cuda_element_exp_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_exp_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_element_exp_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_exp_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_mat_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    element_ReLU_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_element_ReLU_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat_buffer[index] > 0 ? mat_buffer[index] : 0.0;
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

void cuda_element_ReLU_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_ReLU_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_element_ReLU_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_ReLU_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_mat_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void element_ReLU_prime_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = x if x>0 else 1

        int index = tidY * out_cols + tidX;
        out_buffer[index] = mat1_buffer[index] > 0.0 ? 1.0 : 0.0;
    }
}

size_t cuda_element_ReLU_prime(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, bool inplace) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = inplace ? mat1_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    element_ReLU_prime_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_element_ReLU_prime_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < mat_rows) {
        while (tidX < mat_cols) {
            const int index = tidY * mat_cols + tidX;
            out_buffer[index] = mat_buffer[index] > 0.0 ? 1.0 : 0.0;
            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

void cuda_element_ReLU_prime_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_ReLU_prime_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
}

void cuda_element_ReLU_prime_packed_inplace(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Grab device pointers and put in pinned memory to upload as device array
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_element_ReLU_prime_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_mat_buffers_dp, mat_rows, mat_cols);
    gpuErrchk(cudaPeekAtLastError());
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    sum_rows_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    sum_columns_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    transpose_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void cuda_max_pool_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols, float* max_bitmask) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // For each 2x2 area pick the maximum value
        // We will mem coalesce by getting first two in row 1
        // Then next 2 in row2

        int block_start_row = tidY * 2;
        int block_start_col = tidX * 2;
        int block_start = block_start_row * mat1_cols + block_start_col;

        // bool block_00_oob = false;
        bool block_01_oob = (block_start_col + 1) >= mat1_cols;
        bool block_10_oob = (block_start_row + 1) >= mat1_rows;
        bool block_11_oob = block_01_oob || block_10_oob;

        // Unique small values to ensure bitmask is written once
        const float small_float_1 = -1e30;  // Should probably use FLT_MIN but language server no like it
        const float small_float_2 = -1e31;
        const float small_float_3 = -1e32;

        // TODO: Use bit operations instead of ternary (it's faster idk why the compiler can't figure it out)
        float block_00 = mat1_buffer[block_start];
        float block_01 = block_01_oob ? small_float_1 : mat1_buffer[block_start + 1];
        block_start += mat1_cols;
        float block_10 = block_10_oob ? small_float_2 : mat1_buffer[block_start];
        float block_11 = block_11_oob ? small_float_3 : mat1_buffer[block_start + 1];

        float result = max(max(block_00, block_01), max(block_10, block_11));

        // Set bitmask
        max_bitmask[block_start_row * mat1_cols + block_start_col] = (float)(result == block_00);
        if (!block_01_oob) {
            max_bitmask[block_start_row * mat1_cols + block_start_col + 1] = (float)(result == block_01);
        }
        if (!block_10_oob) {
            max_bitmask[(block_start_row + 1) * mat1_cols + block_start_col] = (float)(result == block_10);
        }
        if (!block_11_oob) {
            max_bitmask[(block_start_row + 1) * mat1_cols + block_start_col + 1] = (float)(result == block_11);
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);
    float* gpu_max_bitmask = get_matrix_gpu_address(max_bitmask);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_max_pool_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols, gpu_max_bitmask);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return Tuple{out_mat_id, max_bitmask};
}

// Each block handles one matrix
__global__ void cuda_max_pool_packed_kernel(float** mat_buffers, float** out_buffers, float** max_bitmasks, int mat_rows, int mat_cols, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];
    float* max_bitmask = max_bitmasks[current_matrix];

    // The work will be split among threads in the block
    while (tidY < out_rows) {
        while (tidX < out_cols) {
            // For each 2x2 area pick the maximum value
            // We will mem coalesce by getting first two in row 1
            // Then next 2 in row2

            int block_start_row = tidY * 2;
            int block_start_col = tidX * 2;
            int block_start = block_start_row * mat_cols + block_start_col;

            // bool block_00_oob = false;
            bool block_01_oob = (block_start_col + 1) >= mat_cols;
            bool block_10_oob = (block_start_row + 1) >= mat_rows;
            bool block_11_oob = block_01_oob || block_10_oob;

            // Unique small values to ensure bitmask is written once
            const float small_float_1 = -1e30;  // Should probably use FLT_MIN but language server no like it
            const float small_float_2 = -1e31;
            const float small_float_3 = -1e32;

            // TODO: Use bit operations instead of ternary (it's faster idk why the compiler can't figure it out)
            float block_00 = mat_buffer[block_start];
            float block_01 = block_01_oob ? small_float_1 : mat_buffer[block_start + 1];
            block_start += mat_cols;
            float block_10 = block_10_oob ? small_float_2 : mat_buffer[block_start];
            float block_11 = block_11_oob ? small_float_3 : mat_buffer[block_start + 1];

            float result = max(max(block_00, block_01), max(block_10, block_11));

            // Set bitmask
            max_bitmask[block_start_row * mat_cols + block_start_col] = (float)(result == block_00);
            if (!block_01_oob) {
                max_bitmask[block_start_row * mat_cols + block_start_col + 1] = (float)(result == block_01);
            }
            if (!block_10_oob) {
                max_bitmask[(block_start_row + 1) * mat_cols + block_start_col] = (float)(result == block_10);
            }
            if (!block_11_oob) {
                max_bitmask[(block_start_row + 1) * mat_cols + block_start_col + 1] = (float)(result == block_11);
            }

            // Write maxpool result
            int output_index = tidY * out_cols + tidX;
            out_buffer[output_index] = result;
            // printf("Set result to %f at row %d col %d for mat #%d at %d\n", result, tidY, tidX, current_matrix, output_index);

            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

void cuda_max_pool_packed(size_t* mat_ids, Tuple* out_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows / 2 + mat_rows % 2;
    int out_cols = mat_cols / 2 + mat_cols % 2;

    std::vector<size_t> out_mat_ids(num_matrices);
    std::vector<size_t> max_bitmask_ids(num_matrices);
    register_matrix_group(out_rows, out_cols, num_matrices, &out_mat_ids[0]);
    register_matrix_group(mat_rows, mat_cols, num_matrices, &max_bitmask_ids[0]);

    for (int i = 0; i < num_matrices; i++) {
        size_t out_mat_id = out_mat_ids[i];
        size_t max_bitmask = max_bitmask_ids[i];
        out_ids[i] = Tuple{out_mat_id, max_bitmask};
    }

    // Get gpu buffers to oeprate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_max_bitmasks_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_ids[i].a);
        pinned_max_bitmasks_ptr[i] = get_matrix_gpu_address(out_ids[i].b);
    }

    // Upload the pointers to a gpu array
    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_max_bitmasks_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_max_bitmasks_dp, pinned_max_bitmasks_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_max_pool_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_out_buffers_dp, gpu_max_bitmasks_dp, mat_rows, mat_cols, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
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
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK_X = 16;
    const int THREADS_PER_BLOCK_Y = 16;

    dim3 block_dim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);
    cuda_nearest_neighbor_2x_upsample_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_nearest_neighbor_2x_upsample_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidY < out_rows) {
        while (tidX < out_cols) {
            // O[i][j] = mat[i/2][j/2]
            int mat_index = (tidY / 2) * mat_cols + (tidX / 2);
            int output_index = tidY * out_cols + tidX;

            out_buffer[output_index] = mat_buffer[mat_index];

            tidX += blockDim.x;
        }
        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

void cuda_nearest_neighbor_2x_upsample_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, bool odd_upsample) {
    // Create output buffer
    int out_rows = mat_rows * 2 - (int)odd_upsample;
    int out_cols = mat_cols * 2 - (int)odd_upsample;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_nearest_neighbor_2x_upsample_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void cuda_rotate_180_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    const int mat_length = mat1_rows * mat1_cols;

    if (tidX < mat_length) {
        // Rotating an array 180 means
        // Reversing the linearized array
        const int reversed_index = mat_length - tidX - 1;
        const float input = mat1_buffer[reversed_index];

        const int output_index = tidX;
        out_buffer[output_index] = input;
    }
}

size_t cuda_rotate_180(size_t mat1_id, size_t mat1_rows, size_t mat1_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    int out_length = out_rows * out_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK_X = 256;
    dim3 block_dim(THREADS_PER_BLOCK_X, 1, 1);
    dim3 grid_dim((out_length + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_rotate_180_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Each block handles one matrix
__global__ void cuda_rotate_180_packed_kernel(float** mat_buffers, float** out_buffers, int mat_rows, int mat_cols, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    const int mat_length = out_rows * out_cols;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among threads in the block
    while (tidX < mat_length) {
        // Rotating an array 180 means
        // Reversing the linearized array
        const int reversed_index = mat_length - tidX - 1;
        const float input = mat_buffer[reversed_index];

        int output_index = tidX;
        out_buffer[output_index] = input;

        tidX += blockDim.x;
    }
}

void cuda_rotate_180_packed(size_t* mat_ids, size_t* out_mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(2);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[1];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK_X = 256;
    dim3 block_dim(THREADS_PER_BLOCK_X, 1, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_rotate_180_packed_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, gpu_out_buffers_dp, mat_rows, mat_cols, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
}

// Naive implementation
__global__ void cuda_correlate_kernel_valid_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

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

// Partial Shared Memory implementation
template <const int block_x, const int block_y>
__global__ void cuda_correlate_kernel_valid_2(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
    // Create shared memory
    __shared__ float mat1_shared[block_x * block_y];

    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;
    int threadIdWithinBlock = threadIdx.y * blockDim.x + threadIdx.x;

    // Load data into shared memory
    mat1_shared[threadIdWithinBlock] = mat1_buffer[tidY * mat1_cols + tidX];
    __syncthreads();

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = weighted sum of kernel with input, where kernel is kept within bounds of input
        float result = 0.0;
        const int kernel_top_left_row = tidY;
        const int kernel_top_left_col = tidX;

#pragma unroll 3
        for (int m = 0; m < kernel_rows; m++) {
#pragma unroll 3
            for (int n = 0; n < kernel_cols; n++) {
                int index_in_shared = (threadIdx.y + m) * blockDim.x + (threadIdx.x + n);
                int index_in_global = (kernel_top_left_row + m) * mat1_cols + (kernel_top_left_col + n);
                bool in_shared_bounds = index_in_shared < (block_x * block_y);

                const float mat1_val = in_shared_bounds ? mat1_shared[index_in_shared] : mat1_buffer[index_in_global];
                const float kernel_val = kernel_buffer[m * kernel_cols + n];
                result += mat1_val * kernel_val;
            }
        }

        const int out_index = tidY * out_cols + tidX;
        out_buffer[out_index] = result;
    }
}

// Be careful, this needs to be optimized or your CNN will suffer
size_t cuda_correlate_valid(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    // Dimension of output is input - kernel + 1
    int out_rows = mat1_rows - kernel_rows + 1;
    int out_cols = mat1_cols - kernel_cols + 1;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_kernel_buffer = get_matrix_gpu_address(kernel_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_correlate_kernel_valid_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    // cuda_correlate_kernel_valid_2<THREADS_PER_BLOCK, THREADS_PER_BLOCK><<<grid_dim, block_di>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Naive implementation
__global__ void cuda_correlate_kernel_same_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
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

// correlation is zero-padded (Output is the same size as input)
// Expects odd size, square kernels ONLY
// Be careful, this needs to be optimized or your CNN will suffer
size_t cuda_correlate_same(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_kernel_buffer = get_matrix_gpu_address(kernel_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_correlate_kernel_same_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Naive implementation
__global__ void cuda_correlate_kernel_full_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
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
size_t cuda_correlate_full(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    // Dimension of output is input + kernel - 1
    int out_rows = mat1_rows + kernel_rows - 1;
    int out_cols = mat1_cols + kernel_cols - 1;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_kernel_buffer = get_matrix_gpu_address(kernel_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_correlate_kernel_full_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

size_t cuda_correlate(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols, PaddingType padding_type) {
    if (padding_type == PaddingType::VALID) {
        return cuda_correlate_valid(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    } else if (padding_type == PaddingType::SAME) {
        return cuda_correlate_same(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    } else if (padding_type == PaddingType::FULL) {
        return cuda_correlate_full(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    }
}

// Each block handles one matrix
__global__ void cuda_correlate_kernel_packed_valid_1(float** mat_buffers, int num_matrices, int mat_rows, int mat_cols, float** kernel_buffers, int kernel_rows, int kernel_cols, float** out_buffers, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    const float* kernel_buffer = kernel_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among the threads in the block
    // Each thread will work until of tidX or tidY is out of bounds
    while (tidY < out_rows) {
        while (tidX < out_cols) {
            // Now perform correlation at this location
            float result = 0.0;
            const int kernel_top_left_row = tidY;
            const int kernel_top_left_col = tidX;

#pragma unroll 3
            for (int m = 0; m < kernel_rows; m++) {
#pragma unroll 3
                for (int n = 0; n < kernel_cols; n++) {
                    const float mat1_val = mat_buffer[(kernel_top_left_row + m) * mat_cols + (kernel_top_left_col + n)];
                    const float kernel_val = kernel_buffer[m * kernel_cols + n];
                    result += mat1_val * kernel_val;
                }
            }

            const int out_index = tidY * out_cols + tidX;
            out_buffer[out_index] = result;
            tidX += blockDim.x;
        }

        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

// Should be used when you have a lot of small matrices to convolve
void cuda_correlate_valid_packed(size_t* matrices_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_mat_ids) {
    // Create output buffer
    // Dimension of output is input - kernel + 1
    int out_rows = mat_rows - kernel_rows + 1;
    int out_cols = mat_cols - kernel_cols + 1;

    // Register output matrices
    // Register large buffer
    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_kernel_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(matrices_ids[i]);
        pinned_kernel_buffers_ptr[i] = get_matrix_gpu_address(kernel_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_kernel_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_kernel_buffers_dp, pinned_kernel_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 8;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_correlate_kernel_packed_valid_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, num_matrices, mat_rows, mat_cols, gpu_kernel_buffers_dp, kernel_rows, kernel_cols, gpu_out_buffers_dp, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
}

// Each block handles one correlation
__global__ void cuda_correlate_kernel_packed_same_1(float** mat_buffers, int num_matrices, int mat_rows, int mat_cols, float** kernel_buffers, int kernel_rows, int kernel_cols, float** out_buffers, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    const float* kernel_buffer = kernel_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among the threads in the block
    // Each thread will work until of tidX or tidY is out of bounds
    while (tidY < out_rows) {
        while (tidX < out_cols) {
            float result = 0.0;
            const int apothem = kernel_rows / 2;
#pragma unroll 3
            for (int m = 0; m < kernel_rows; m++) {
#pragma unroll 3
                for (int n = 0; n < kernel_cols; n++) {
                    int input_row = m - apothem + tidY;
                    int input_col = n - apothem + tidX;
                    bool input_row_in_bounds = input_row >= 0 && input_row < mat_rows;
                    bool input_col_in_bounds = input_col >= 0 && input_col < mat_cols;

                    if (input_row_in_bounds && input_col_in_bounds) {
                        const int curr_mat1_index = input_row * mat_cols + input_col;
                        const int curr_kernel_index = m * kernel_cols + n;
                        result += mat_buffer[curr_mat1_index] * kernel_buffer[curr_kernel_index];
                    }
                }
            }

            int output_index = tidY * out_cols + tidX;
            out_buffer[output_index] = result;
            tidX += blockDim.x;
        }

        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

// Should be used when you have a lot of small matrices to convolve
void cuda_correlate_same_packed(size_t* matrices_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_mat_ids) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Register output matrices
    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_kernel_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(matrices_ids[i]);
        pinned_kernel_buffers_ptr[i] = get_matrix_gpu_address(kernel_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_kernel_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_kernel_buffers_dp, pinned_kernel_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 8;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_correlate_kernel_packed_same_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, num_matrices, mat_rows, mat_cols, gpu_kernel_buffers_dp, kernel_rows, kernel_cols, gpu_out_buffers_dp, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
}

// Each block handles one correlation
__global__ void cuda_correlate_kernel_packed_full_1(float** mat_buffers, int num_matrices, int mat_rows, int mat_cols, float** kernel_buffers, int kernel_rows, int kernel_cols, float** out_buffers, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    const float* kernel_buffer = kernel_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among the threads in the block
    // Each thread will work until of tidX or tidY is out of bounds
    while (tidY < out_rows) {
        while (tidX < out_cols) {
            // O[i][j] = weighted sum of kernel with input, where kernel is centered at i,j
            float result = 0.0;
            const int input_start_row = (-kernel_rows + 1) + tidY;
            const int input_start_col = (-kernel_cols + 1) + tidX;
            for (int m = 0; m < kernel_rows; m++) {
                for (int n = 0; n < kernel_cols; n++) {
                    int input_row = input_start_row + m;
                    int input_col = input_start_col + n;
                    bool input_row_in_bounds = input_row >= 0 && input_row < mat_rows;
                    bool input_col_in_bounds = input_col >= 0 && input_col < mat_cols;

                    if (input_row_in_bounds && input_col_in_bounds) {
                        const int curr_mat1_index = input_row * mat_cols + input_col;
                        const int curr_kernel_index = m * kernel_cols + n;
                        result += mat_buffer[curr_mat1_index] * kernel_buffer[curr_kernel_index];
                    }
                }
            }

            int output_index = tidY * out_cols + tidX;
            out_buffer[output_index] = result;
            tidX += blockDim.x;
        }

        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

// Should be used when you have a lot of small matrices to convolve
void cuda_correlate_full_packed(size_t* matrices_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_mat_ids) {
    // Create output buffer
    // Dimension of output is input + kernel - 1
    int out_rows = mat_rows + kernel_rows - 1;
    int out_cols = mat_cols + kernel_cols - 1;

    // Register output matrices
    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_kernel_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(matrices_ids[i]);
        pinned_kernel_buffers_ptr[i] = get_matrix_gpu_address(kernel_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    // Upload the pointers to a gpu array
    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_kernel_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_kernel_buffers_dp, pinned_kernel_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 8;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_correlate_kernel_packed_full_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, num_matrices, mat_rows, mat_cols, gpu_kernel_buffers_dp, kernel_rows, kernel_cols, gpu_out_buffers_dp, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
}
void cuda_correlate_packed(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_ids, PaddingType padding_type) {
    if (padding_type == PaddingType::VALID) {
        return cuda_correlate_valid_packed(mat_ids, num_matrices, mat_rows, mat_cols, kernel_ids, kernel_rows, kernel_cols, out_ids);
    } else if (padding_type == PaddingType::SAME) {
        return cuda_correlate_same_packed(mat_ids, num_matrices, mat_rows, mat_cols, kernel_ids, kernel_rows, kernel_cols, out_ids);
    } else if (padding_type == PaddingType::FULL) {
        return cuda_correlate_full_packed(mat_ids, num_matrices, mat_rows, mat_cols, kernel_ids, kernel_rows, kernel_cols, out_ids);
    }
}

// Naive implementation
__global__ void cuda_convolve_kernel_valid_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;
    const int kernel_length = kernel_rows * kernel_cols;

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
                const int rotated_kernel_position = kernel_length - (m * kernel_cols + n) - 1;  // Equivalent to reversing linearized kernel
                const float kernel_val = kernel_buffer[rotated_kernel_position];
                result += mat1_val * kernel_val;
            }
        }

        const int out_index = tidY * out_cols + tidX;
        out_buffer[out_index] = result;
    }
}

// Be careful, this needs to be optimized or your CNN will suffer
size_t cuda_convolve_valid(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    // Dimension of output is input - kernel + 1
    int out_rows = mat1_rows - kernel_rows + 1;
    int out_cols = mat1_cols - kernel_cols + 1;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_kernel_buffer = get_matrix_gpu_address(kernel_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_convolve_kernel_valid_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Naive implementation
__global__ void cuda_convolve_kernel_same_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;
    const int kernel_length = kernel_rows * kernel_cols;

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
                    const int rotated_kernel_position = kernel_length - (m * kernel_cols + n) - 1;  // Equivalent to reversing linearized kernel
                    result += mat1_buffer[curr_mat1_index] * kernel_buffer[rotated_kernel_position];
                }
            }
        }

        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = result;
    }
}

// correlation is zero-padded (Output is the same size as input)
// Expects odd size, square kernels ONLY
// Be careful, this needs to be optimized or your CNN will suffer
size_t cuda_convolve_same(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat1_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_kernel_buffer = get_matrix_gpu_address(kernel_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_convolve_kernel_same_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Naive implementation
__global__ void cuda_convolve_kernel_full_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* kernel_buffer, int kernel_rows, int kernel_cols, float* out_buffer, int out_rows, int out_cols) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;
    const int kernel_length = kernel_rows * kernel_cols;

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
                    const int rotated_kernel_position = kernel_length - (m * kernel_cols + n) - 1;  // Equivalent to reversing linearized kernel
                    result += mat1_buffer[curr_mat1_index] * kernel_buffer[rotated_kernel_position];
                }
            }
        }

        int output_index = tidY * out_cols + tidX;
        out_buffer[output_index] = result;
    }
}

// Be careful, this needs to be optimized or your CNN will suffer
size_t cuda_convolve_full(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols) {
    // Create output buffer
    // Dimension of output is input + kernel - 1
    int out_rows = mat1_rows + kernel_rows - 1;
    int out_cols = mat1_cols + kernel_cols - 1;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = get_matrix_gpu_address(mat1_id);
    float* gpu_kernel_buffer = get_matrix_gpu_address(kernel_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_convolve_kernel_full_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_kernel_buffer, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

size_t cuda_convolve(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t kernel_id, size_t kernel_rows, size_t kernel_cols, PaddingType padding_type) {
    if (padding_type == PaddingType::VALID) {
        return cuda_convolve_valid(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    } else if (padding_type == PaddingType::SAME) {
        return cuda_convolve_same(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    } else if (padding_type == PaddingType::FULL) {
        return cuda_convolve_full(mat1_id, mat1_rows, mat1_cols, kernel_id, kernel_rows, kernel_cols);
    }
}

// Each block handles one matrix
__global__ void cuda_convolve_kernel_packed_valid_1(float** mat_buffers, int num_matrices, int mat_rows, int mat_cols, float** kernel_buffers, int kernel_rows, int kernel_cols, float** out_buffers, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;
    const int kernel_length = kernel_rows * kernel_cols;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    const float* kernel_buffer = kernel_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among the threads in the block
    // Each thread will work until of tidX or tidY is out of bounds
    while (tidY < out_rows) {
        while (tidX < out_cols) {
            // Now perform correlation at this location
            float result = 0.0;
            const int kernel_top_left_row = tidY;
            const int kernel_top_left_col = tidX;

#pragma unroll 3
            for (int m = 0; m < kernel_rows; m++) {
#pragma unroll 3
                for (int n = 0; n < kernel_cols; n++) {
                    const float mat1_val = mat_buffer[(kernel_top_left_row + m) * mat_cols + (kernel_top_left_col + n)];
                    const int rotated_kernel_position = kernel_length - (m * kernel_cols + n) - 1;  // Equivalent to reversing linearized kernel
                    const float kernel_val = kernel_buffer[rotated_kernel_position];
                    result += mat1_val * kernel_val;
                }
            }

            const int out_index = tidY * out_cols + tidX;
            out_buffer[out_index] = result;
            tidX += blockDim.x;
        }

        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

// Should be used when you have a lot of small matrices to convolve
void cuda_convolve_valid_packed(size_t* matrices_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_mat_ids) {
    // Create output buffer
    // Dimension of output is input - kernel + 1
    int out_rows = mat_rows - kernel_rows + 1;
    int out_cols = mat_cols - kernel_cols + 1;

    // Register output matrices
    // Register large buffer
    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_kernel_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(matrices_ids[i]);
        pinned_kernel_buffers_ptr[i] = get_matrix_gpu_address(kernel_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    // Upload the pointers to a gpu array
    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_kernel_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_kernel_buffers_dp, pinned_kernel_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 8;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_convolve_kernel_packed_valid_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, num_matrices, mat_rows, mat_cols, gpu_kernel_buffers_dp, kernel_rows, kernel_cols, gpu_out_buffers_dp, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
}

// Each block handles one correlation
__global__ void cuda_convolve_kernel_packed_same_1(float** mat_buffers, int num_matrices, int mat_rows, int mat_cols, float** kernel_buffers, int kernel_rows, int kernel_cols, float** out_buffers, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;
    const int kernel_length = kernel_rows * kernel_cols;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    const float* kernel_buffer = kernel_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among the threads in the block
    // Each thread will work until of tidX or tidY is out of bounds
    while (tidY < out_rows) {
        while (tidX < out_cols) {
            float result = 0.0;
            const int apothem = kernel_rows / 2;
#pragma unroll 3
            for (int m = 0; m < kernel_rows; m++) {
#pragma unroll 3
                for (int n = 0; n < kernel_cols; n++) {
                    int input_row = m - apothem + tidY;
                    int input_col = n - apothem + tidX;
                    bool input_row_in_bounds = input_row >= 0 && input_row < mat_rows;
                    bool input_col_in_bounds = input_col >= 0 && input_col < mat_cols;

                    if (input_row_in_bounds && input_col_in_bounds) {
                        const int curr_mat1_index = input_row * mat_cols + input_col;
                        const int rotated_kernel_position = kernel_length - (m * kernel_cols + n) - 1;  // Equivalent to reversing linearized kernel
                        result += mat_buffer[curr_mat1_index] * kernel_buffer[rotated_kernel_position];
                    }
                }
            }

            int output_index = tidY * out_cols + tidX;
            out_buffer[output_index] = result;
            tidX += blockDim.x;
        }

        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

// Should be used when you have a lot of small matrices to convolve
void cuda_convolve_same_packed(size_t* matrices_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_mat_ids) {
    // Create output buffer
    int out_rows = mat_rows;
    int out_cols = mat_cols;

    // Register output matrices
    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_kernel_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(matrices_ids[i]);
        pinned_kernel_buffers_ptr[i] = get_matrix_gpu_address(kernel_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    // Upload the pointers to a gpu array
    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_kernel_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_kernel_buffers_dp, pinned_kernel_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 8;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_convolve_kernel_packed_same_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, num_matrices, mat_rows, mat_cols, gpu_kernel_buffers_dp, kernel_rows, kernel_cols, gpu_out_buffers_dp, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
}

// Each block handles one correlation
__global__ void cuda_convolve_kernel_packed_full_1(float** mat_buffers, int num_matrices, int mat_rows, int mat_cols, float** kernel_buffers, int kernel_rows, int kernel_cols, float** out_buffers, int out_rows, int out_cols) {
    const int current_matrix = blockIdx.x;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;
    const int kernel_length = kernel_rows * kernel_cols;

    // Grab the buffers
    const float* mat_buffer = mat_buffers[current_matrix];
    const float* kernel_buffer = kernel_buffers[current_matrix];
    float* out_buffer = out_buffers[current_matrix];

    // The work will be split among the threads in the block
    // Each thread will work until of tidX or tidY is out of bounds
    while (tidY < out_rows) {
        while (tidX < out_cols) {
            // O[i][j] = weighted sum of kernel with input, where kernel is centered at i,j
            float result = 0.0;
            const int input_start_row = (-kernel_rows + 1) + tidY;
            const int input_start_col = (-kernel_cols + 1) + tidX;
            for (int m = 0; m < kernel_rows; m++) {
                for (int n = 0; n < kernel_cols; n++) {
                    int input_row = input_start_row + m;
                    int input_col = input_start_col + n;
                    bool input_row_in_bounds = input_row >= 0 && input_row < mat_rows;
                    bool input_col_in_bounds = input_col >= 0 && input_col < mat_cols;

                    if (input_row_in_bounds && input_col_in_bounds) {
                        const int curr_mat1_index = input_row * mat_cols + input_col;
                        const int rotated_kernel_position = kernel_length - (m * kernel_cols + n) - 1;  // Equivalent to reversing linearized kernel
                        result += mat_buffer[curr_mat1_index] * kernel_buffer[rotated_kernel_position];
                    }
                }
            }

            int output_index = tidY * out_cols + tidX;
            out_buffer[output_index] = result;
            tidX += blockDim.x;
        }

        tidX = threadIdx.x;
        tidY += blockDim.y;
    }
}

// Should be used when you have a lot of small matrices to convolve
void cuda_convolve_full_packed(size_t* matrices_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_mat_ids) {
    // Create output buffer
    // Dimension of output is input + kernel - 1
    int out_rows = mat_rows + kernel_rows - 1;
    int out_cols = mat_cols + kernel_cols - 1;

    // Register output matrices
    register_matrix_group(out_rows, out_cols, num_matrices, out_mat_ids);

    // Get the gpu buffers to operate on
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_kernel_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    float** pinned_out_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(matrices_ids[i]);
        pinned_kernel_buffers_ptr[i] = get_matrix_gpu_address(kernel_ids[i]);
        pinned_out_buffers_ptr[i] = get_matrix_gpu_address(out_mat_ids[i]);
    }

    // Upload the pointers to a gpu array
    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(3);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];
    float** gpu_kernel_buffers_dp = (float**)kernel_arg_device_pointers[1];
    float** gpu_out_buffers_dp = (float**)kernel_arg_device_pointers[2];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_kernel_buffers_dp, pinned_kernel_buffers_ptr, sizeof(float*) * num_matrices);
    memory_manager_upload_from_pinned_buffer(gpu_out_buffers_dp, pinned_out_buffers_ptr, sizeof(float*) * num_matrices);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 8;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim(num_matrices, 1, 1);

    // Run the kernels
    cuda_convolve_kernel_packed_full_1<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, num_matrices, mat_rows, mat_cols, gpu_kernel_buffers_dp, kernel_rows, kernel_cols, gpu_out_buffers_dp, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());
}
void cuda_convolve_packed(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t* kernel_ids, size_t kernel_rows, size_t kernel_cols, size_t* out_ids, PaddingType padding_type) {
    if (padding_type == PaddingType::VALID) {
        return cuda_convolve_valid_packed(mat_ids, num_matrices, mat_rows, mat_cols, kernel_ids, kernel_rows, kernel_cols, out_ids);
    } else if (padding_type == PaddingType::SAME) {
        return cuda_convolve_same_packed(mat_ids, num_matrices, mat_rows, mat_cols, kernel_ids, kernel_rows, kernel_cols, out_ids);
    } else if (padding_type == PaddingType::FULL) {
        return cuda_convolve_full_packed(mat_ids, num_matrices, mat_rows, mat_cols, kernel_ids, kernel_rows, kernel_cols, out_ids);
    }
}

__global__ void cuda_img2col_valid(float** mat_buffers, int input_depth, int input_rows, int input_cols, int filter_depth, int filter_rows, int filter_cols, float* out_buffer, int out_rows, int out_cols) {
    const int tidX = blockDim.x * blockIdx.x + threadIdx.x;

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
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(size_t) * num_matrices);
    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    // Upload the pointers to a gpu array
    // Each allocation pair contains block_id, block_offset
    float** gpu_mat_buffers = (float**)get_device_kernel_args_pointers(1)[0];
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);

    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    // Let's have each threads handle its own patch between all the kernels
    // So we will calculate the number of patches == number of columns
    // Data access should be coalesced this way
    const int THREADS_PER_BLOCK = 1024;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_img2col_valid<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers, num_matrices, mat_rows, mat_cols, kernel_count, kernel_rows, kernel_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

// Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
size_t cuda_img2col(size_t* mat_ids, size_t num_matrices, size_t mat_rows, size_t mat_cols, size_t kernel_rows, size_t kernel_cols, PaddingType padding_type) {
    if (padding_type == PaddingType::VALID) {
        return cuda_img2col_valid(mat_ids, num_matrices, mat_rows, mat_cols, kernel_rows, kernel_cols);
    } else if (padding_type == PaddingType::SAME) {
        return 0;
    } else if (padding_type == PaddingType::FULL) {
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
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);

    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_flatten_array_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffers_dp, mat_rows, mat_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

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
    register_matrix_group(mat_rows, mat_cols, num_matrices, mat_ids);
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);

    // Get the flattened array
    float* gpu_array_buffer = get_matrix_gpu_address(array_id);

    // Kernel launch parameters
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((arr_size + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_unflatten_array_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_array_buffer, arr_size, mat_rows, mat_cols, gpu_mat_buffers_dp);
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
    register_matrix_group(mat_rows, mat_cols, num_matrices, mat_ids);
    float** pinned_mat_buffers_ptr = (float**)memory_manager_get_pinned_allocation(sizeof(float*) * num_matrices);
    for (int i = 0; i < num_matrices; i++) {
        pinned_mat_buffers_ptr[i] = get_matrix_gpu_address(mat_ids[i]);
    }

    auto kernel_arg_device_pointers = get_device_kernel_args_pointers(1);
    float** gpu_mat_buffers_dp = (float**)kernel_arg_device_pointers[0];

    // Upload the pointers to a gpu array
    memory_manager_upload_from_pinned_buffer(gpu_mat_buffers_dp, pinned_mat_buffers_ptr, sizeof(float*) * num_matrices);

    // Get the flattened array
    float* gpu_array_buffer = get_matrix_gpu_address(array_id);

    // Kernel launch parameters
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((arr_size + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_unflatten_array_strided_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_array_buffer, arr_size, num_matrices, mat_rows, mat_cols, gpu_mat_buffers_dp);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void cuda_center_pad_kernel(float* mat_buffer, int mat_rows, int mat_cols, int pad_rows, int pad_cols, float* out_buffer, int out_rows, int out_cols) {
    const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < out_cols && tidY < out_rows) {
        // O[i][j] = I[i - pad_rows][j - pad_cols] if in bounds, else 0
        const int input_row = tidY - pad_rows;
        const int input_col = tidX - pad_cols;
        const bool input_row_in_bounds = input_row >= 0 && input_row < mat_rows;
        const bool input_col_in_bounds = input_col >= 0 && input_col < mat_cols;
        const bool in_bounds = input_row_in_bounds && input_col_in_bounds;

        const float result = in_bounds ? mat_buffer[input_row * mat_cols + input_col] : 0.0;
        out_buffer[tidY * out_cols + tidX] = result;
    }
}

size_t cuda_center_pad(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t pad_rows, size_t pad_cols) {
    // Create output buffer
    int out_rows = mat_rows + 2 * pad_rows;
    int out_cols = mat_cols + 2 * pad_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_center_pad_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, mat_rows, mat_cols, pad_rows, pad_cols, gpu_out_buffer, out_rows, out_cols);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void cuda_softmax_kernel(float* mat_buffer, int mat_rows, int mat_cols, float* out_buffer) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < mat_cols) {
        // Go down all the rows, find the max
        float max = -INFINITY;
        for (int row = 0; row < mat_rows; row++) {
            const float val = mat_buffer[row * mat_cols + col];
            max = val > max ? val : max;
        }

        // Now go down all the rows and subtract the max, then exponentiate
        float sum = 0.0;
        for (int row = mat_rows - 1; row >= 0; row--) {
            const float val = mat_buffer[row * mat_cols + col];
            const float exp_val = __expf(val - max);
            out_buffer[row * mat_cols + col] = exp_val;
            sum += exp_val;
        }

        // Now go down all the rows and divide by the sum
        for (int row = 0; row < mat_rows; row++) {
            out_buffer[row * mat_cols + col] /= sum;
        }
    }
}

size_t cuda_softmax(size_t mat_id, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    size_t out_mat_id = register_matrix(mat_rows, mat_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 128;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((mat_cols + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_softmax_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, mat_rows, mat_cols, gpu_out_buffer);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

__global__ void cuda_crop_kernel(float* mat_buffer, int mat_rows, int mat_cols, int crop_offset_rows, int crop_offset_cols, int crop_rows, int crop_cols, float* out_buffer) {
    const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < crop_cols && tidY < crop_rows) {
        // O[i][j] = I[i + crop_offset_rows][j + crop_offset_cols]
        const int input_row = tidY + crop_offset_rows;
        const int input_col = tidX + crop_offset_cols;

        const float result = mat_buffer[input_row * mat_cols + input_col];
        out_buffer[tidY * crop_cols + tidX] = result;
    }
}

size_t cuda_crop(size_t mat_id, size_t mat_rows, size_t mat_cols, size_t crop_offset_rows, size_t crop_offset_cols, size_t crop_rows, size_t crop_cols) {
    // Create output buffer
    size_t out_mat_id = register_matrix(crop_rows, crop_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 16;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((crop_cols + block_dim.x - 1) / block_dim.x, (crop_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    cuda_crop_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, mat_rows, mat_cols, crop_offset_rows, crop_offset_cols, crop_rows, crop_cols, gpu_out_buffer);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}

size_t cuda_copy(size_t mat_id, size_t mat_rows, size_t mat_cols) {
    // We will just use the crop function
    return cuda_crop(mat_id, mat_rows, mat_cols, 0, 0, mat_rows, mat_cols);
}

template <const int block_size>
__global__ void cuda_sum_all_matrix_elements_kernel(float* mat_buffer, int elements_to_sum) {
    // Shared memory for each block. each block handles blockDim elements
    __shared__ float sdata[block_size];

    // Load element into shared
    const int input_index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = input_index < elements_to_sum ? mat_buffer[input_index] : 0.0;
    __syncthreads();

    // Do reduction in shared memory
    const int sdata_index = threadIdx.x;
    int active_threads = blockDim.x / 2;
    while (active_threads > 0 && sdata_index < active_threads) {
        sdata[sdata_index] += sdata[sdata_index + active_threads];
        __syncthreads();
        active_threads /= 2;
    }

    // Write result for this block to global memory
    if (sdata_index == 0) {
        mat_buffer[blockIdx.x] = sdata[0];
    }
}

size_t cuda_sum_all_matrix_elements(size_t mat_id, size_t mat_rows, size_t mat_cols) {
    // Create output buffer
    size_t mat_copy_id = cuda_copy(mat_id, mat_rows, mat_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_copy_id);

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 256;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);

    // Run the kernels
    int elements_to_sum = mat_rows * mat_cols;
    while (elements_to_sum > 1) {
        dim3 grid_dim((elements_to_sum + block_dim.x - 1) / block_dim.x, 1, 1);
        cuda_sum_all_matrix_elements_kernel<THREADS_PER_BLOCK><<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, elements_to_sum);
        gpuErrchk(cudaPeekAtLastError());

        elements_to_sum = grid_dim.x;
    }

    // Return just the first element
    size_t result_id = cuda_crop(mat_copy_id, mat_rows, mat_cols, 0, 0, 1, 1);

    // Free the copy
    unregister_matrix(mat_copy_id);

    return result_id;
}

__global__ void cuda_max_by_column_kernel(float* mat_buffer, int mat_rows, int mat_cols, float* out_buffer) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < mat_cols) {
        // Go down all the rows, find the max
        float max = -INFINITY;
        for (int row = 0; row < mat_rows; row++) {
            const float val = mat_buffer[row * mat_cols + col];
            max = val > max ? val : max;
        }

        // Write result
        out_buffer[col] = max;
    }
}
size_t cuda_max_by_column(size_t mat_id, size_t mat_rows, size_t mat_cols) {
    size_t out_mat_id = register_matrix(1, mat_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 256;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((mat_cols + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_max_by_column_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, mat_rows, mat_cols, gpu_out_buffer);

    return out_mat_id;
}

__global__ void cuda_max_by_row_kernel(float* mat_buffer, int mat_rows, int mat_cols, float* out_buffer) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < mat_rows) {
        // Go down all the rows, find the max
        float max = -INFINITY;
        for (int col = 0; col < mat_cols; col++) {
            const float val = mat_buffer[row * mat_cols + col];
            max = val > max ? val : max;
        }

        // Write result
        out_buffer[row] = max;
    }
}
size_t cuda_max_by_row(size_t mat_id, size_t mat_rows, size_t mat_cols) {
    size_t out_mat_id = register_matrix(mat_rows, 1);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 256;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((mat_rows + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_max_by_row_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, mat_rows, mat_cols, gpu_out_buffer);

    return out_mat_id;
}

__global__ void cuda_argmax_by_column_kernel(float* mat_buffer, int mat_rows, int mat_cols, float* out_buffer) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < mat_cols) {
        // Go down all the rows, find the max
        float max = -INFINITY;
        float max_index = 0;
        for (int row = 0; row < mat_rows; row++) {
            const float val = mat_buffer[row * mat_cols + col];
            max_index = val > max ? row : max_index;
            max = val > max ? val : max;
        }

        // Write result
        out_buffer[col] = max_index;
    }
}
size_t cuda_argmax_by_column(size_t mat_id, size_t mat_rows, size_t mat_cols) {
    size_t out_mat_id = register_matrix(1, mat_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 256;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((mat_cols + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_argmax_by_column_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, mat_rows, mat_cols, gpu_out_buffer);

    return out_mat_id;
}

__global__ void cuda_argmax_by_row_kernel(float* mat_buffer, int mat_rows, int mat_cols, float* out_buffer) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < mat_rows) {
        // Go down all the rows, find the max
        float max = -INFINITY;
        float max_index = 0;
        for (int col = 0; col < mat_cols; col++) {
            const float val = mat_buffer[row * mat_cols + col];
            max_index = val > max ? col : max_index;
            max = val > max ? val : max;
        }

        // Write result
        out_buffer[row] = max_index;
    }
}
size_t cuda_argmax_by_row(size_t mat_id, size_t mat_rows, size_t mat_cols) {
    size_t out_mat_id = register_matrix(mat_rows, 1);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 256;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((mat_rows + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_argmax_by_row_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, mat_rows, mat_cols, gpu_out_buffer);

    return out_mat_id;
}

__global__ void cuda_one_hot_encode_kernel(float* data_buffer, int data_size, int num_classes, float* out_buffer) {
    const int tidX = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidX < data_size) {
        // Each thread handles one row
        // O[i][j] = 1 if j == data[i], else 0
        const int column_to_write_to = data_buffer[tidX];
        out_buffer[tidX * num_classes + column_to_write_to] = 1.0;
    }
}
size_t cuda_one_hot_encode(float* data, size_t data_size, size_t num_classes) {
    // Create output buffer
    size_t out_mat_id = register_matrix(data_size, num_classes);

    // Used pinned transfer for data
    float* pinned_buffer = (float*)memory_manager_get_pinned_allocation(sizeof(float) * data_size);
    memcpy(pinned_buffer, data, sizeof(float) * data_size);

    // Upload the data
    float* gpu_data_buffer = (float*)get_device_kernel_args_pointers(1)[0];
    memory_manager_upload_from_pinned_buffer(gpu_data_buffer, pinned_buffer, sizeof(float) * data_size);

    // Get buffers
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Set the output buffer to 0
    cudaMemsetAsync(gpu_out_buffer, 0, sizeof(float) * data_size * num_classes, get_stream());

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 256;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((data_size + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_one_hot_encode_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_data_buffer, data_size, num_classes, gpu_out_buffer);

    return out_mat_id;
}

size_t cuda_one_hot_encode_vector(size_t mat_id, size_t mat_len, size_t num_classes) {
    // Create output buffer
    size_t out_mat_id = register_matrix(mat_len, num_classes);

    // Get buffers
    float* gpu_data_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Set the output buffer to 0
    cudaMemsetAsync(gpu_out_buffer, 0, sizeof(float) * mat_len * num_classes, get_stream());

    // Kernel launch parameters, each thread handles one column
    const int THREADS_PER_BLOCK = 256;
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((mat_len + block_dim.x - 1) / block_dim.x, 1, 1);

    // Run the kernels
    cuda_one_hot_encode_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_data_buffer, mat_len, num_classes, gpu_out_buffer);

    return out_mat_id;
}

__global__ void element_ln_kernel(float* mat_buffer, int mat_rows, int mat_cols, float* out_buffer) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidX < mat_cols && tidY < mat_rows) {
        // O[i][j] = ln(mat1[i][j])

        int index = tidY * mat_cols + tidX;
        const float val = mat_buffer[index];
        out_buffer[index] = val <= 0.0 ? 0.0 : log(mat_buffer[index]);  // Handle <= 0 to avoid NAN
    }
}

size_t cuda_element_ln(size_t mat_id, size_t mat_rows, size_t mat_cols, bool inplace) {
    // Create output buffer
    size_t out_rows = mat_rows;
    size_t out_cols = mat_cols;
    size_t out_mat_id = inplace ? mat_id : register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat_buffer = get_matrix_gpu_address(mat_id);
    float* gpu_out_buffer = get_matrix_gpu_address(out_mat_id);

    // Kernel launch parameters
    const int THREADS_PER_BLOCK = 32;
    dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 grid_dim((out_cols + block_dim.x - 1) / block_dim.x, (out_rows + block_dim.y - 1) / block_dim.y, 1);

    // Run the kernels
    element_ln_kernel<<<grid_dim, block_dim, 0, get_stream()>>>(gpu_mat_buffer, mat_rows, mat_cols, gpu_out_buffer);
    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}
