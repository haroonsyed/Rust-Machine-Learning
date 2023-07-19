use std::ffi::c_float;

extern "C" {
  pub fn test();
  pub fn test_array_fill(out: *mut c_float, length: usize);
  pub fn cuda_synchronize();
  pub fn register_matrix(data: *const c_float, rows: usize, cols: usize) -> usize;
  pub fn unregister_matrix(mat_id: usize) -> usize;
  pub fn get_matrix_data(mat_id: usize, rows: usize, cols: usize, data_buffer: *mut c_float);
  pub fn cuda_element_add(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_subtract(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_multiply(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_scalar_multiply(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    scalar: c_float,
    inplace: bool,
  ) -> usize;
  pub fn cuda_matrix_multiply(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
  ) -> usize;
  pub fn cuda_add_vector(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_divide_by_vector(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_exp(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_ReLU(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_ReLU_prime(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_sum_rows(mat1_id: usize, mat1_rows: usize, mat1_cols: usize) -> usize;
  pub fn cuda_sum_columns(mat1_id: usize, mat1_rows: usize, mat1_cols: usize) -> usize;
  pub fn cuda_transpose(mat1_id: usize, mat1_rows: usize, mat1_cols: usize) -> usize;
}
