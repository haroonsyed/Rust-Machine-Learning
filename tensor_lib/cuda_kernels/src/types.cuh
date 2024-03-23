enum PaddingType {
    VALID,
    SAME,
    FULL
};

struct Matrix {
    float* address;
    size_t block_id;
};