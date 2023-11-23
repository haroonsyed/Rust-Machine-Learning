#include "./cuda_exec_memory_manager.cuh"

bool library_init = false;
cudaStream_t mem_stream;
cudaMemPool_t mempool;
std::vector<cudaStream_t> exec_streams;
bool parallel_stream_execution = false;

cublasHandle_t handle;
size_t mat_generated_count(0);
std::unordered_map<size_t, float*> mat_map;

/////////////////////
/// Matrix Setup API
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
cudaStream_t get_stream() {
    if (parallel_stream_execution) {
        return exec_streams[mat_generated_count % exec_streams.size()];
    }
    return 0;
}
size_t register_matrix_buffer(float* gpu_buffer) {
    // Register with the map for retrieval later
    mat_map[mat_generated_count] = gpu_buffer;
    return mat_generated_count++;  // Fine if this overflows
}

size_t register_matrix(size_t rows, size_t cols) {
    if (!library_init) {
        init_library();
    }

    // Upload the data
    float* gpu_buffer;
    gpuErrchk(cudaMallocFromPoolAsync(&gpu_buffer, sizeof(float) * rows * cols, mempool, mem_stream));

    return register_matrix_buffer(gpu_buffer);
}

size_t register_matrix_with_data(float* data, size_t rows, size_t cols) {
    if (!library_init) {
        init_library();
    }

    // Upload the data
    float* gpu_buffer;
    gpuErrchk(cudaMallocFromPoolAsync(&gpu_buffer, sizeof(float) * rows * cols, mempool, mem_stream));
    gpuErrchk(cudaMemcpy(gpu_buffer, data, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
    return register_matrix_buffer(gpu_buffer);
}

void unregister_matrix(size_t mat_id) {
    gpuErrchk(cudaFreeAsync(mat_map[mat_id], mem_stream));
    mat_map.erase(mat_id);
}

void get_matrix_data(size_t mat_id, int rows, int cols, float* data_buffer) {
    float* gpu_buffer = mat_map[mat_id];
    gpuErrchk(cudaMemcpy(data_buffer, gpu_buffer, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
}