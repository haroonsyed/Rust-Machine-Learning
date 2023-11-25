#include "./cuda_exec_memory_manager.cuh"

struct MatrixMemBlock {
    size_t chunk_id;
    size_t chunk_offset;
    size_t rows;
    size_t columns;
};

struct ChunkMemBlock {
    char* address;
    size_t used_size;             // Size of the chunk allocated to matrices
    size_t allocation_size_left;  // The amount of contiguous free space at the end of the chunk.
    size_t total_size;            // Not necessarily the same as used_size + allocation_size_left. This is the size of the chunk.
};

bool library_init = false;
cudaStream_t mem_stream;
cudaMemPool_t mempool;
std::vector<cudaStream_t> exec_streams;
bool parallel_stream_execution = false;

cublasHandle_t handle;
size_t mat_generated_count(0);
size_t chunks_generated_count(0);
const size_t chunk_size(sizeof(char) * 1024 * 1024 * 1);  // 1 MB
std::unordered_map<size_t, MatrixMemBlock> matrix_map;
std::unordered_map<size_t, ChunkMemBlock> gpu_mem_blocks;
ChunkMemBlock* current_chunk = nullptr;

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
    // Init streams
    int exec_stream_count = 4;
    for (int i = 0; i < exec_stream_count; i++) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        exec_streams.push_back(stream);
    }

    // Init mem stream, for now we will just use stream 0
    mem_stream = 0;
    // cudaStreamCreate(&mem_stream);

    // Init cublas
    init_cublas_handle();

    // Init pool
    init_min_pool_size();

    library_init = true;
}
void enable_parallel_stream_execution() {
    // Wait for all streams to finish
    cudaDeviceSynchronize();
    parallel_stream_execution = true;
}
void disable_parallel_stream_execution() {
    // Wait for all streams to finish
    cudaDeviceSynchronize();
    parallel_stream_execution = false;
}
/////////////////////
/// Stream Management
/////////////////////
cudaStream_t get_stream() {
    if (parallel_stream_execution) {
        return exec_streams[mat_generated_count % exec_streams.size()];
    }
    return 0;
}
/////////////////////
/// Memory Allocation
/////////////////////
void* get_block_gpu_address(size_t block_id, size_t block_offset) {
    ChunkMemBlock block = gpu_mem_blocks[block_id];
    return block.address + block_offset;
}
std::pair<size_t, size_t> allocate_from_chunk(size_t size) {
    // Check if we have enough space in the current chunk
    if (current_chunk->allocation_size_left < size) {
        printf("CHUNK NOT BIG ENOUGH FOR REQUESTED ALLOCATION");
        abort();
    }
    size_t current_allocation_offset = current_chunk->total_size - current_chunk->allocation_size_left;
    current_chunk->allocation_size_left -= size;
    current_chunk->used_size += size;
    return std::make_pair(chunks_generated_count, current_allocation_offset);
}

// Allocate a chunk in multiple of chunk_size. Returns the chunkID and chunk offset
std::pair<size_t, size_t> memory_manager_allocate(size_t size) {
    if (!library_init) {
        init_library();
    }

    // Round up size to a multiple of 16 bytes to ensure alignment
    size = ((size / 16) + (size % 16 > 0 ? 1 : 0)) * 16;

    if (current_chunk != nullptr && current_chunk->allocation_size_left >= size) {
        return allocate_from_chunk(size);
    }

    // Determine the multiple of chunk_size that is greater than size
    size_t min_chunk_multiple = ((size - 1) / chunk_size) + 1;
    size_t curr_chunk_size = min_chunk_multiple * chunk_size;

    // Allocate a new chunk
    char* address;
    gpuErrchk(cudaMallocAsync(&address, curr_chunk_size, mempool, mem_stream));
    chunks_generated_count++;

    ChunkMemBlock block;
    block.address = address;
    block.used_size = 0;
    block.allocation_size_left = curr_chunk_size;
    block.total_size = curr_chunk_size;
    gpu_mem_blocks[chunks_generated_count] = block;
    current_chunk = &gpu_mem_blocks[chunks_generated_count];

    return allocate_from_chunk(size);
}
void memory_manager_free(size_t block_id, size_t size) {
    // Align size to 16 bytes
    size = ((size / 16) + (size % 16 > 0 ? 1 : 0)) * 16;

    ChunkMemBlock* block = &gpu_mem_blocks[block_id];
    block->used_size -= size;

    bool should_reset_current_chunk = block == current_chunk;

    // If the block is empty, free it
    if (block->used_size == 0) {
        gpuErrchk(cudaFreeAsync(block->address, mem_stream));
        gpu_mem_blocks.erase(block_id);
    }

    if (should_reset_current_chunk) {
        current_chunk = nullptr;
    }
}
void memory_manager_upload_to_allocation(size_t block_id, size_t block_offset, void* data, size_t size) {
    void* address = get_block_gpu_address(block_id, block_offset);
    gpuErrchk(cudaMemcpyAsync(address, data, size, cudaMemcpyHostToDevice, mem_stream));
}
/////////////////////
/// Matrix Allocation
/////////////////////
float* get_matrix_gpu_address(size_t mat_id) {
    MatrixMemBlock* mat = &matrix_map[mat_id];
    size_t chunk = mat->chunk_id;
    size_t offset = mat->chunk_offset;
    return (float*)get_block_gpu_address(chunk, offset);
}
size_t register_matrix_block(MatrixMemBlock mat) {
    // Register with the map for retrieval later
    matrix_map[mat_generated_count] = mat;
    return mat_generated_count++;
}
size_t register_matrix(size_t rows, size_t columns) {
    // Allocate the memory
    auto chunk_info = memory_manager_allocate(sizeof(float) * rows * columns);
    size_t chunk_id = chunk_info.first;
    size_t chunk_offset = chunk_info.second;

    // Create the matrix block
    MatrixMemBlock mat;
    mat.chunk_id = chunk_id;
    mat.chunk_offset = chunk_offset;
    mat.rows = rows;
    mat.columns = columns;

    return register_matrix_block(mat);
}
size_t register_matrix_with_data(float* data, size_t rows, size_t columns) {
    // Create the matrix block
    size_t matrix_id = register_matrix(rows, columns);

    // Get the gpu address
    float* gpu_address = get_matrix_gpu_address(matrix_id);

    // Upload the data
    gpuErrchk(cudaMemcpy(gpu_address, data, sizeof(float) * rows * columns, cudaMemcpyHostToDevice));
    return matrix_id;
}
void unregister_matrix(size_t mat_id) {
    MatrixMemBlock* mat = &matrix_map[mat_id];
    size_t chunk_id = mat->chunk_id;
    size_t matrix_size = mat->rows * mat->columns * sizeof(float);
    memory_manager_free(chunk_id, matrix_size);
    matrix_map.erase(mat_id);
}
void get_matrix_data(size_t mat_id, int rows, int columns, float* data_buffer) {
    float* gpu_address = get_matrix_gpu_address(mat_id);
    gpuErrchk(cudaMemcpy(data_buffer, gpu_address, sizeof(float) * rows * columns, cudaMemcpyDeviceToHost));
}