#include <chrono>
#include <queue>

#include "./cuda_exec_memory_manager.cuh"

struct MatrixBlock {
    size_t block_id;
    void* device_address;
    size_t block_size_requested;
    size_t ref_count;
    size_t rows;
    size_t columns;
};

struct MemBlock {
    char* address;
    size_t requested_size_used;
};

bool library_init = false;
cudaMemPool_t mempool;

cudaStream_t main_exec_stream;
cudaStream_t io_to_device_stream;

void* pinned_buffer;
const size_t pinned_buffer_size = 1024 * 1024 * 512;  // 512 MB
size_t pinned_buffer_offset = 0;
size_t pinned_buffer_free_space = pinned_buffer_size;
std::queue<std::pair<cudaEvent_t, size_t>> pinned_buffer_events;

cublasHandle_t handle;

// Idea, use a hashmap of <device_pointer -> MatrixBlock>
// ID would be the device pointer instead of storing separately
// Then matrix api can direct memcpy input ids to device pointer
size_t matrices_generated_count(0);
std::vector<MatrixBlock> matrix_blocks;
std::vector<size_t> free_mat_ids;
size_t blocks_generated_count(0);
std::vector<MemBlock> mem_blocks;
std::vector<size_t> free_block_ids;

// Optimization, malloc a buffer for arguments to kernels
// Reduces calls to malloc and free for each kernel call
void* kernel_args_buffer;
const size_t kernel_args_buffer_size = 1024 * 1024 * 128;  // 128 MB

/////////////////////
/// INIT API
/////////////////////
void init_cublas_handle() {
    cublasCreate(&handle);
    cublasSetStream(handle, 0);
}
void init_min_pool_size() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetDefaultMemPool(&mempool, device);
    size_t threshold = UINT64_MAX;  // Since exclusive to one process, we can max out threshold
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
}
void init_library() {
    // Init cublas
    // init_cublas_handle();

    // Init pool
    init_min_pool_size();

    // Init streams
    cudaStreamCreate(&main_exec_stream);
    cudaStreamCreate(&io_to_device_stream);

    // Init pinned buffer
    cudaMallocHost(&pinned_buffer, (size_t)(pinned_buffer_size));

    // Init kernel args device buffer
    cudaMallocAsync(&kernel_args_buffer, kernel_args_buffer_size, main_exec_stream);

    library_init = true;
}
/////////////////////
/// Stream Management
/////////////////////
cudaStream_t get_stream() {
    return main_exec_stream;
}
/////////////////////
/// Memory Allocation
/////////////////////
void* get_block_gpu_address(size_t block_id, size_t block_offset) {
    MemBlock* block = &mem_blocks[block_id];
    return block->address + block_offset;
}
void memory_manager_delete(size_t block_id) {
    MemBlock* block = &mem_blocks[block_id];
    gpuErrchk(cudaFreeAsync(block->address, main_exec_stream));
    free_block_ids.emplace_back(block_id);
}
void memory_manager_free(size_t block_id, size_t size) {
    MemBlock* block = &mem_blocks[block_id];
    block->requested_size_used -= size;

    if (block->requested_size_used == 0) {
        memory_manager_delete(block_id);
    }
}
void memory_manager_upload_from_pinned_buffer(void* pinned_data, void* device_address, size_t size) {
    gpuErrchk(cudaMemcpyAsync(device_address, pinned_data, size, cudaMemcpyHostToDevice, main_exec_stream));

    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, main_exec_stream);

    // Store the event and pinned buffer address
    pinned_buffer_events.emplace(event, size);
}
void memory_manager_upload_async_from_pinned_buffer(void* pinned_data, void* device_address, size_t size) {
    gpuErrchk(cudaMemcpyAsync(device_address, pinned_data, size, cudaMemcpyHostToDevice, io_to_device_stream));

    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, io_to_device_stream);
    cudaStreamWaitEvent(main_exec_stream, event, 0);

    // Store the event and pinned buffer address
    pinned_buffer_events.emplace(event, size);
}
void memory_manager_upload_to_allocation(void* address, void* data, size_t size) {
    gpuErrchk(cudaMemcpy(address, data, size, cudaMemcpyHostToDevice));
}
void* memory_get_pinned_allocation(size_t size) {
    if (pinned_buffer_offset + size > pinned_buffer_size) {
        // Adjust size to wait for space on wrap around
        size += pinned_buffer_size - pinned_buffer_offset;

        // Reset the offset
        pinned_buffer_offset = 0;
    }

    while (pinned_buffer_free_space < size) {
        // Wait for space to be available
        std::pair<cudaEvent_t, size_t> event = pinned_buffer_events.front();
        cudaEventSynchronize(event.first);
        pinned_buffer_free_space += event.second;
        pinned_buffer_events.pop();
    }

    void* address = (char*)pinned_buffer + pinned_buffer_offset;
    pinned_buffer_offset += size;

    return address;
}

// The default maximum number size of each buffer is 128 MB / num_buffers
// That means in most cases (3 buffers), the maximum size is 42 MB, which is 5.5 million matrix pointers
// MUST BE USED WITH ONE THREAD AND cudaMallocAsync in execution stream!
std::vector<void*> get_device_kernel_args_pointers(size_t num_buffers) {
    std::vector<void*> pointers;
    for (size_t i = 0; i < num_buffers; i++) {
        // Align to 256 bytes
        size_t offset = (i * kernel_args_buffer_size / num_buffers);
        offset = ((offset + 255) / 256) * 256;

        void* pointer = (char*)kernel_args_buffer + offset;
        pointers.emplace_back(pointer);
    }
    return pointers;
}
size_t memory_manager_allocate(size_t size) {
    if (!library_init) {
        init_library();
    }

    char* address;
    gpuErrchk(cudaMallocAsync(&address, size, mempool, main_exec_stream));

    MemBlock block;
    block.address = address;
    block.requested_size_used = size;

    // Store the block
    size_t block_id = free_block_ids.size() > 0 ? free_block_ids.back() : blocks_generated_count++;
    if (free_block_ids.size() > 0) {
        mem_blocks[block_id] = block;
        free_block_ids.pop_back();
    } else {
        mem_blocks.emplace_back(block);
    }

    return block_id;
}

//////////////////////
/// Matrix Information
//////////////////////
size_t get_matrix_rows(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];
    return mat->rows;
}
size_t get_matrix_columns(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];
    return mat->columns;
}
size_t get_matrix_length(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];
    return mat->rows * mat->columns;
}
void reshape_matrix(size_t mat_id, size_t rows, size_t columns) {
    MatrixBlock* mat = &matrix_blocks[mat_id];

    if (get_matrix_length(mat_id) != rows * columns) {
        printf("Reshape error: new shape must have same number of elements\n");
        abort();
    }

    mat->rows = rows;
    mat->columns = columns;
}

/////////////////////
/// Matrix Allocation
/////////////////////
float* get_matrix_gpu_address(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];
    return (float*)mat->device_address;
}
size_t register_matrix_block(MatrixBlock& mat) {
    if (free_mat_ids.size() > 0) {
        size_t mat_id = free_mat_ids.back();
        matrix_blocks[mat_id] = mat;
        free_mat_ids.pop_back();
        return mat_id;
    }

    matrix_blocks.emplace_back(mat);
    return matrices_generated_count++;
}
size_t register_matrix(size_t rows, size_t columns) {
    // Allocate the memory
    size_t block_size = sizeof(float) * rows * columns;
    size_t block_id = memory_manager_allocate(block_size);

    // Create the matrix block
    MatrixBlock mat;
    mat.block_id = block_id;
    mat.device_address = get_block_gpu_address(block_id, 0);
    mat.block_size_requested = block_size;
    mat.ref_count = 1;
    mat.rows = rows;
    mat.columns = columns;

    return register_matrix_block(mat);
}
void register_matrix_group(size_t rows, size_t columns, size_t count, size_t* mat_ids) {
    size_t requested_size = sizeof(float) * rows * columns;

    // Ensure alignment of matrices to 256 bytes
    size_t alignment = 16;
    size_t aligned_requested_size = requested_size + (alignment - (requested_size % alignment));
    size_t real_block_size = aligned_requested_size * count;

    // Allocate the memory
    size_t block_id = memory_manager_allocate(real_block_size);

    for (size_t i = 0; i < count; i++) {
        const size_t block_offset = i * aligned_requested_size;

        // Create the matrix block
        MatrixBlock mat;
        mat.block_id = block_id;
        mat.device_address = get_block_gpu_address(block_id, block_offset);
        mat.block_size_requested = aligned_requested_size;
        mat.ref_count = 1;
        mat.rows = rows;
        mat.columns = columns;

        mat_ids[i] = register_matrix_block(mat);
    }
}
void upload_matrix_data(size_t mat_id, float* data) {
    // Block information
    MatrixBlock* mat = &matrix_blocks[mat_id];
    void* device_address = mat->device_address;
    size_t data_size = get_matrix_length(mat_id) * sizeof(float);

    // Upload the data
    memory_manager_upload_to_allocation(device_address, data, data_size);
}
size_t register_matrix_with_data(float* data, size_t rows, size_t columns) {
    // Create the matrix block
    size_t matrix_id = register_matrix(rows, columns);

    // Upload the data
    upload_matrix_data(matrix_id, data);
    return matrix_id;
}
void unregister_matrix(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];

    size_t block_id = mat->block_id;
    size_t matrix_size = mat->block_size_requested;
    memory_manager_free(block_id, matrix_size);
    free_mat_ids.emplace_back(mat_id);
}
void increase_matrix_ref_count(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];
    mat->ref_count++;
}
void decrease_matrix_ref_count(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];
    mat->ref_count--;
    if (mat->ref_count == 0) {
        unregister_matrix(mat_id);
    }
}
void get_matrix_data(size_t mat_id, float* data_buffer) {
    float* gpu_address = get_matrix_gpu_address(mat_id);
    size_t data_size = get_matrix_length(mat_id) * sizeof(float);
    gpuErrchk(cudaMemcpy(data_buffer, gpu_address, data_size, cudaMemcpyDeviceToHost));
}