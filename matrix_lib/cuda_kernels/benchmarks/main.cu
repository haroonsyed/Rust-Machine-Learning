#include <chrono>
#include <vector>
using namespace std::chrono;
#include "../cuda_kernels.cuh"

int main() {
    // This is just for timing, assumes everything is correct.
    // The tests already cover correctness.
    int dim = 4096;
    std::vector<float> data;
    for (int i = 0; i < dim * dim; i++) {
        data.push_back(23.47);
    }

    // Register
    int mat1 = register_matrix(&data[0], dim, dim);
    int mat2 = register_matrix(&data[0], dim, dim);

    auto start_host = high_resolution_clock::now();

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    int num_iter = 100;
    for (int i = 0; i < num_iter; i++) {
        // Perform multiplication
        int result_id = cuda_matrix_multiply(mat1, dim, dim, mat2, dim, dim);
        cuda_synchronize();
        unregister_matrix(result_id);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, end);

    auto end_host = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end_host - start_host);

    std::cout << "Average gpu function time was: " << gpu_time / num_iter << " ms" << std::endl;
    std::cout << "Including overhead was: " << (float)cpu_time.count() / num_iter << " ms" << std::endl;

    // Okay something is wrong with the overhead on rust benchmark. Something taking 184.3 ms here is taking 1.3 seconds there.
    // Same functions on ffi being called...
}