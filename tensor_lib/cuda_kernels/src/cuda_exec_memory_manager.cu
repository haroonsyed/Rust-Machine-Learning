#include "./cuda_exec_memory_manager.cuh"

struct MatrixMemBlock {
    size_t chunk_id;
    size_t chunk_offset;
    size_t rows;
    size_t columns;
};

struct ChunkMemBlock {
    char* address;
    size_t chunk_id;
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
std::vector<MatrixMemBlock> matrix_map;
std::vector<size_t> free_mat_ids;
std::vector<ChunkMemBlock> gpu_mem_blocks;
std::vector<size_t> free_chunk_ids;
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
    return std::make_pair(current_chunk->chunk_id, current_allocation_offset);
}
void memory_manager_delete_block(ChunkMemBlock* block) {
    gpuErrchk(cudaFreeAsync(block->address, mem_stream));
    free_chunk_ids.push_back(block->chunk_id);
}
void memory_manager_free(size_t block_id, size_t size) {
    // Align size to 16 bytes
    size = ((size / 16) + (size % 16 > 0 ? 1 : 0)) * 16;

    ChunkMemBlock* block = &gpu_mem_blocks[block_id];
    block->used_size -= size;

    // Keep current chunk if it has allocatable space left
    bool is_current_chunk = block == current_chunk;

    // If the block is empty, free it
    if (block->used_size == 0) {
        if (is_current_chunk) {
            current_chunk->allocation_size_left = current_chunk->total_size;
        } else {
            memory_manager_delete_block(block);
        }
    }
}
void memory_manager_upload_to_allocation(size_t block_id, size_t block_offset, void* data, size_t size) {
    // Copy data from pinned buffer to gpu
    // When performing async memcpy with page-locked memory, the behavior is same as cudaMemcpy.
    // But the call is asynchronous, and allows the cpu overhead to be lower.
    void* address = get_block_gpu_address(block_id, block_offset);
    gpuErrchk(cudaMemcpyAsync(address, data, size, cudaMemcpyHostToDevice, mem_stream));
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

    // Delete current block if empty
    if (current_chunk != nullptr && current_chunk->used_size == 0) {
        memory_manager_delete_block(current_chunk);
    }

    // Determine the multiple of chunk_size that is greater than size
    size_t min_chunk_multiple = ((size - 1) / chunk_size) + 1;
    size_t curr_chunk_size = min_chunk_multiple * chunk_size;

    // Allocate a new chunk
    char* address;
    gpuErrchk(cudaMallocAsync(&address, curr_chunk_size, mempool, mem_stream));

    ChunkMemBlock block;
    block.address = address;
    block.used_size = 0;
    block.allocation_size_left = curr_chunk_size;
    block.total_size = curr_chunk_size;

    // Determine Chunk ID
    size_t chunk_id = free_chunk_ids.size() > 0 ? free_chunk_ids.back() : chunks_generated_count++;
    if (free_chunk_ids.size() > 0) {
        gpu_mem_blocks[chunk_id] = block;
        free_chunk_ids.pop_back();
    } else {
        gpu_mem_blocks.emplace_back(block);
    }
    gpu_mem_blocks[chunk_id].chunk_id = chunk_id;
    current_chunk = &gpu_mem_blocks[chunk_id];

    return allocate_from_chunk(size);
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
    size_t mat_id = free_mat_ids.size() > 0 ? free_mat_ids.back() : mat_generated_count;
    if (free_mat_ids.size() > 0) {
        matrix_map[mat_id] = mat;
        free_mat_ids.pop_back();
    } else {
        matrix_map.emplace_back(mat);
        mat_generated_count++;
    }
    return mat_id;
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

    // Block information
    MatrixMemBlock* mat = &matrix_map[matrix_id];
    size_t chunk_id = mat->chunk_id;
    size_t chunk_offset = mat->chunk_offset;
    size_t mat_size = sizeof(float) * rows * columns;

    // Upload the data
    memory_manager_upload_to_allocation(chunk_id, chunk_offset, data, mat_size);
    return matrix_id;
}
void unregister_matrix(size_t mat_id) {
    MatrixMemBlock* mat = &matrix_map[mat_id];
    size_t chunk_id = mat->chunk_id;
    size_t matrix_size = mat->rows * mat->columns * sizeof(float);
    memory_manager_free(chunk_id, matrix_size);
    free_mat_ids.push_back(mat_id);
}
void get_matrix_data(size_t mat_id, int rows, int columns, float* data_buffer) {
    float* gpu_address = get_matrix_gpu_address(mat_id);
    gpuErrchk(cudaMemcpy(data_buffer, gpu_address, sizeof(float) * rows * columns, cudaMemcpyDeviceToHost));
}