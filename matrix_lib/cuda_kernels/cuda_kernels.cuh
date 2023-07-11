#include <cuda.h>

#include <iostream>

// Make sure bindings are not mangled for rust
extern "C" {
void test();
void test_array_fill(double* buffer, size_t length);
}
