use std::ffi::c_double;

extern "C" {
  pub fn test();
  pub fn test_array_fill(out: *mut c_double, length: usize);
  pub fn cuda_matrix_multiply(
    mat1_buffer: *const c_double,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: *const c_double,
    mat2_rows: usize,
    mat2_cols: usize,
    out_buffer: *mut c_double,
    out_rows: usize,
    out_cols: usize,
  );
}
