#include <chrono>
#include <queue>

#include "./cuda_exec_memory_manager.cuh"

struct MemBlock {
    char* address;
    size_t mat_count;
    size_t mat_rows;
    size_t mat_columns;
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
std::vector<cudaEvent_t> free_pinned_buffer_events;

cublasHandle_t handle;

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
    cudaMalloc(&kernel_args_buffer, kernel_args_buffer_size);

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
void* get_block_gpu_address(size_t block_id) {
    MemBlock* block = &mem_blocks[block_id];
    return block->address;
}
void memory_manager_delete(size_t block_id) {
    MemBlock* block = &mem_blocks[block_id];
    gpuErrchk(cudaFreeAsync(block->address, main_exec_stream));
    free_block_ids.emplace_back(block_id);
}
void memory_manager_free(size_t block_id) {
    MemBlock* block = &mem_blocks[block_id];
    block->mat_count--;

    if (block->mat_count == 0) {
        memory_manager_delete(block_id);
    }
}
cudaEvent_t get_event() {
    if (free_pinned_buffer_events.size() > 0) {
        cudaEvent_t event = free_pinned_buffer_events.back();
        free_pinned_buffer_events.pop_back();
        return event;
    }

    cudaEvent_t event;
    cudaEventCreate(&event, cudaEventDisableTiming);
    return event;
}
void memory_manager_upload_from_pinned_buffer(void* device_address, void* pinned_data, size_t size) {
    gpuErrchk(cudaMemcpyAsync(device_address, pinned_data, size, cudaMemcpyHostToDevice, main_exec_stream));

    cudaEvent_t event = get_event();
    cudaEventRecord(event, main_exec_stream);

    // Store the event and pinned buffer address
    pinned_buffer_events.emplace(event, size);
}
void memory_manager_upload_async_from_pinned_buffer(void* device_address, void* pinned_data, size_t size) {
    return memory_manager_upload_from_pinned_buffer(device_address, pinned_data, size);

    gpuErrchk(cudaMemcpyAsync(device_address, pinned_data, size, cudaMemcpyHostToDevice, io_to_device_stream));  // TODO: USE IO STREAM

    cudaEvent_t event = get_event();
    cudaEventRecord(event, main_exec_stream);
    cudaStreamWaitEvent(io_to_device_stream, event, 0);

    // Store the event and pinned buffer address
    pinned_buffer_events.emplace(event, size);
}
void memory_manager_upload_to_allocation(void* address, void* data, size_t size) {
    gpuErrchk(cudaMemcpy(address, data, size, cudaMemcpyHostToDevice));
}
void* memory_manager_get_pinned_allocation(size_t size) {
    // Align allocation to 256 bytes
    size_t aligned_size = ((size + 255) / 256) * 256;
    size = aligned_size;

    if (pinned_buffer_offset + size > pinned_buffer_size) {
        // Adjust size to wait for space on wrap around
        size += pinned_buffer_size - pinned_buffer_offset;

        // Reset the offset
        pinned_buffer_offset = 0;
    }

    while (pinned_buffer_free_space < size) {
        // Wait for space to be available
        printf("Waiting for pinned buffer space\n");
        std::pair<cudaEvent_t, size_t> event = pinned_buffer_events.front();
        cudaEventSynchronize(event.first);
        pinned_buffer_free_space += event.second;
        free_pinned_buffer_events.emplace_back(event.first);
        pinned_buffer_events.pop();
    }

    void* address = (char*)pinned_buffer + pinned_buffer_offset;
    pinned_buffer_offset += size;

    return address;
}

// The default maximum number size of each buffer is 128 MB / num_buffers
// That means in most cases (3 buffers), the maximum size is 42 MB, which is 5.5 million matrix pointers
// MUST BE USED WITH ONE THREAD AND cudaMallocAsync in execution stream!
std::vector<Matrix*> get_device_kernel_args_pointers(size_t num_buffers) {
    std::vector<Matrix*> pointers;
    for (size_t i = 0; i < num_buffers; i++) {
        // Align to 256 bytes
        size_t offset = (i * kernel_args_buffer_size / num_buffers);
        offset = ((offset + 255) / 256) * 256;

        void* pointer = (char*)kernel_args_buffer + offset;
        pointers.emplace_back((Matrix*)pointer);
    }
    return pointers;
}
size_t memory_manager_device_allocate(size_t size) {
    if (!library_init) {
        init_library();
    }

    char* address;
    gpuErrchk(cudaMallocAsync(&address, size, mempool, main_exec_stream));

    MemBlock block;
    block.address = address;
    block.mat_rows = 1;
    block.mat_columns = size / sizeof(float);
    block.mat_count = 1;

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
size_t get_matrix_rows(Matrix* matrix) {
    MemBlock* block = &mem_blocks[matrix->block_id];
    return block->mat_rows;
}
size_t get_matrix_columns(Matrix* matrix) {
    MemBlock* block = &mem_blocks[matrix->block_id];
    return block->mat_columns;
}
size_t get_matrix_length(Matrix* matrix) {
    MemBlock* block = &mem_blocks[matrix->block_id];
    return block->mat_rows * block->mat_columns;
}
void reshape_matrix(Matrix* matrix, size_t rows, size_t columns) {
    MemBlock* block = &mem_blocks[matrix->block_id];

    // if (block->mat_count > 1) {
    //     printf("Warning, reshape will affect all matrices in this mem group. If you do not want this, consider deep cloning to its own mem block.");
    //     abort();
    // }

    if (get_matrix_length(matrix) != rows * columns) {
        printf("Reshape error: new shape must have same number of elements\n");
        abort();
    }

    block->mat_rows = rows;
    block->mat_columns = columns;
}

/////////////////////
/// Matrix Allocation
/////////////////////
Matrix register_matrix(size_t rows, size_t columns) {
    // Allocate the memory
    size_t block_size = sizeof(float) * rows * columns;
    size_t block_id = memory_manager_device_allocate(block_size);

    // Modify the block to include matrix information
    MemBlock* block = &mem_blocks[block_id];
    block->mat_rows = rows;
    block->mat_columns = columns;

    // Cast the address to a size_t
    return Matrix{
        .address = reinterpret_cast<float*>(block->address),
        .block_id = block_id,
    };
}
void register_matrix_group(size_t rows, size_t columns, size_t count, Matrix* matrices) {
    size_t requested_size = sizeof(float) * rows * columns;

    size_t alignment = 256;
    size_t aligned_requested_size = ((requested_size + alignment - 1) / alignment) * alignment;
    size_t real_block_size = aligned_requested_size * count;

    // Allocate the memory
    size_t block_id = memory_manager_device_allocate(real_block_size);

    // Modify the block to include matrix information
    MemBlock* block = &mem_blocks[block_id];
    block->mat_rows = rows;
    block->mat_columns = columns;
    block->mat_count = count;

    for (size_t i = 0; i < count; i++) {
        const size_t block_offset = i * aligned_requested_size;

        matrices[i] = Matrix{
            .address = reinterpret_cast<float*>(block->address + block_offset),
            .block_id = block_id,
        };
    }
}
void upload_matrix_data(Matrix* matrix, float* data) {
    void* device_address = reinterpret_cast<void*>(matrix->address);
    size_t data_size = get_matrix_length(matrix) * sizeof(float);

    // Upload the data
    memory_manager_upload_to_allocation(device_address, data, data_size);
}
void upload_matrix_data_async(Matrix* matrix, float* data) {
    void* device_address = reinterpret_cast<void*>(matrix->address);
    size_t data_size = get_matrix_length(matrix) * sizeof(float);

    // Upload the data
    memory_manager_upload_async_from_pinned_buffer(device_address, data, data_size);
}
Matrix register_matrix_with_data(float* data, size_t rows, size_t columns) {
    // Create the matrix block
    Matrix matrix = register_matrix(rows, columns);

    // Upload the data
    upload_matrix_data(&matrix, data);
    return matrix;
}
void unregister_matrix(Matrix* matrix) {
    memory_manager_free(matrix->block_id);
}
void increase_matrix_ref_count(Matrix* matrix) {
    MemBlock* block = &mem_blocks[matrix->block_id];
    block->mat_count++;
}
void decrease_matrix_ref_count(Matrix* matrix) {
    // Attempting to free is okay, because this matrix will only be freed if whole block is freed
    memory_manager_free(matrix->block_id);
}
void get_matrix_data(Matrix* matrix, float* data_buffer) {
    void* gpu_address = reinterpret_cast<void*>(matrix->address);
    size_t data_size = get_matrix_length(matrix) * sizeof(float);
    gpuErrchk(cudaMemcpy(data_buffer, gpu_address, data_size, cudaMemcpyDeviceToHost));
}