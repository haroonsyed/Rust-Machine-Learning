#include "./cuda_kernels.cuh"

__global__ void test_kernel() {
    printf("Hello from the kernel!\n");
}

void test() {
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    std::cout << "Finished Running Kernels." << std::endl;
}
