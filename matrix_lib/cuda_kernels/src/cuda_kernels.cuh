#include <iostream>

// Make sure bindings are not mangled for rust
extern "C" {
void test();
}
