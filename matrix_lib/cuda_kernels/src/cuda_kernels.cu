#include "./cuda_kernels.cuh"

__global__ void test_kernel() {
    printf("Hello from the kernel!\n");
}

#include <stdio.h>

#include "./cuda_kernels.cuh"

__global__ void add_kernel(int* result) {
    printf("dsijngkjdsg");
    *result = 8;
}

void test() {
    int result;
    int* d_result;

    cudaMalloc((void**)&d_result, sizeof(int));

    add_kernel<<<1, 1>>>(d_result);

    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_result);

    printf("Result: %d\n", result);
    std::cout << "Finished Running Kernels." << std::endl;
}
