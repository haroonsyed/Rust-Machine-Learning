#ifndef GPU_ERR_CHECK_UTIL
#define GPU_ERR_CHECK_UTIL

// Error checking macro: https://stackoverflow.com/a/14038590
#include <stdio.h>
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#endif