#include "./cuda_exec_memory_manager.cuh"

struct MatrixBlock {
    size_t block_id;
    size_t block_offset;
    size_t block_size_requested;
};

struct MemBlock {
    char* address;
    size_t requested_size_used;
};

bool library_init = false;
cudaMemPool_t mempool;

cublasHandle_t handle;
size_t matrices_generated_count(0);
std::vector<MatrixBlock> matrix_blocks;
std::vector<size_t> free_mat_ids;
size_t blocks_generated_count(0);
std::vector<MemBlock> mem_blocks;
std::vector<size_t> free_block_ids;

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

    library_init = true;
}
/////////////////////
/// Stream Management
/////////////////////
cudaStream_t get_stream() {
    return 0;
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
    gpuErrchk(cudaFreeAsync(block->address, 0));
    free_block_ids.push_back(block_id);
}
void memory_manager_free(size_t block_id, size_t size) {
    MemBlock* block = &mem_blocks[block_id];
    block->requested_size_used -= size;

    if (block->requested_size_used == 0) {
        memory_manager_delete(block_id);
    }
}
void memory_manager_upload_to_allocation(size_t block_id, size_t block_offset, void* data, size_t size) {
    // TODO: Copy data to pinned buffer, then from pinned buffer to gpu
    // This should allow pipeline to be more asynchronous
    void* address = get_block_gpu_address(block_id, block_offset);
    gpuErrchk(cudaMemcpy(address, data, size, cudaMemcpyHostToDevice));
}
size_t memory_manager_allocate(size_t size) {
    if (!library_init) {
        init_library();
    }

    // Allocate a new chunk
    char* address;
    gpuErrchk(cudaMallocAsync(&address, size, mempool, 0));

    MemBlock block;
    block.address = address;
    block.requested_size_used = size;

    // Store the block
    size_t block_id = free_block_ids.size() > 0 ? free_block_ids.back() : blocks_generated_count++;
    if (free_block_ids.size() > 0) {
        mem_blocks[block_id] = block;
        free_block_ids.pop_back();
    } else {
        mem_blocks.push_back(block);
    }

    return block_id;
}
/////////////////////
/// Matrix Allocation
/////////////////////
float* get_matrix_gpu_address(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];
    size_t block_id = mat->block_id;
    size_t offset = mat->block_offset;
    return (float*)get_block_gpu_address(block_id, offset);
}
size_t register_matrix_block(MatrixBlock mat) {
    // Register with the map for retrieval later
    size_t mat_id = free_mat_ids.size() > 0 ? free_mat_ids.back() : matrices_generated_count++;
    if (free_mat_ids.size() > 0) {
        matrix_blocks[mat_id] = mat;
        free_mat_ids.pop_back();
    } else {
        matrix_blocks.push_back(mat);
    }
    return mat_id;
}
size_t register_matrix(size_t rows, size_t columns) {
    // Allocate the memory
    size_t block_size = sizeof(float) * rows * columns;
    size_t block_id = memory_manager_allocate(block_size);

    // Create the matrix block
    MatrixBlock mat;
    mat.block_id = block_id;
    mat.block_offset = 0;
    mat.block_size_requested = block_size;

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
        // Create the matrix block
        MatrixBlock mat;
        mat.block_id = block_id;
        mat.block_offset = i * aligned_requested_size;
        mat.block_size_requested = aligned_requested_size;

        mat_ids[i] = register_matrix_block(mat);
    }
}
void upload_matrix_data(size_t mat_id, float* data, size_t rows, size_t columns) {
    // Block information
    MatrixBlock* mat = &matrix_blocks[mat_id];
    size_t block_id = mat->block_id;
    size_t block_offset = mat->block_offset;
    size_t data_size = sizeof(float) * rows * columns;

    // Upload the data
    memory_manager_upload_to_allocation(block_id, block_offset, data, data_size);
}
size_t register_matrix_with_data(float* data, size_t rows, size_t columns) {
    // Create the matrix block
    size_t matrix_id = register_matrix(rows, columns);

    // Upload the data
    upload_matrix_data(matrix_id, data, rows, columns);
    return matrix_id;
}
void unregister_matrix(size_t mat_id) {
    MatrixBlock* mat = &matrix_blocks[mat_id];
    size_t block_id = mat->block_id;
    size_t matrix_size = mat->block_size_requested;
    memory_manager_free(block_id, matrix_size);
    free_mat_ids.push_back(mat_id);
}
void get_matrix_data(size_t mat_id, int rows, int columns, float* data_buffer) {
    float* gpu_address = get_matrix_gpu_address(mat_id);
    gpuErrchk(cudaMemcpy(data_buffer, gpu_address, sizeof(float) * rows * columns, cudaMemcpyDeviceToHost));
}