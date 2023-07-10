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

    cudaError_t err = cudaMalloc((void**)&d_result, sizeof(int));
    if (err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }

    add_kernel<<<1, 1>>>(d_result);
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
    std::cout << "Finished Running Kernels." << std::endl;
}
