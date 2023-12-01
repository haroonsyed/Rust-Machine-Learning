enum PaddingType {
    VALID,
    SAME,
    FULL
};

// Used to return tuple with interop to rust
struct Tuple {
    size_t a;
    size_t b;
};