use std::ffi::c_float;
use std::ffi::c_ulonglong;

use crate::Matrix;
use crate::PaddingType;

extern "C" {
  pub fn test();
  pub fn test_array_fill(out: *mut c_float, length: usize);

  pub fn cuda_synchronize();

  pub fn memory_manager_get_pinned_allocation(size: usize) -> *mut c_float;

  pub fn register_matrix(rows: usize, cols: usize) -> Matrix;
  pub fn register_matrix_group(rows: usize, cols: usize, count: usize, matrices: *mut Matrix);
  pub fn register_matrix_with_data(data: *const c_float, rows: usize, cols: usize) -> Matrix;
  pub fn cuda_one_hot_encode(data: *const c_float, data_size: usize, num_classes: usize) -> Matrix;
  pub fn upload_matrix_data(matrix: *const Matrix, data: *const c_float);
  pub fn upload_matrix_data_async(matrix: *const Matrix, data: *const c_float);
  pub fn unregister_matrix(matrix: *const Matrix);

  pub fn increase_matrix_ref_count(matrix: *const Matrix);
  pub fn decrease_matrix_ref_count(matrix: *const Matrix);

  pub fn get_matrix_rows(matrix: *const Matrix) -> usize;
  pub fn get_matrix_columns(matrix: *const Matrix) -> usize;
  pub fn get_matrix_length(matrix: *const Matrix) -> usize;
  pub fn get_matrix_data(matrix: *const Matrix, data_buffer: *mut c_float);
  pub fn reshape_matrix(matrix: *const Matrix, rows: usize, columns: usize);

  pub fn cuda_element_add(
    matrix_1: *const Matrix,
    matrix_2: *const Matrix,
    inplace: bool,
  ) -> Matrix;
  pub fn cuda_element_add_packed(
    mat1_addresses: *const *const c_ulonglong,
    mat2_addresses: *const *const c_ulonglong,
    out_matrices: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_add_packed_inplace(
    mat1_addresses: *const *const c_ulonglong,
    mat2_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_subtract(
    matrix_1: *const Matrix,
    matrix_2: *const Matrix,
    inplace: bool,
  ) -> Matrix;
  pub fn cuda_element_subtract_packed(
    mat1_addresses: *const *const c_ulonglong,
    mat2_addresses: *const *const c_ulonglong,
    out_matrices: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_subtract_packed_inplace(
    mat1_addresses: *const *const c_ulonglong,
    mat2_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_multiply(
    matrix_1: *const Matrix,
    matrix_2: *const Matrix,
    inplace: bool,
  ) -> Matrix;
  pub fn cuda_element_multiply_packed(
    mat1_addresses: *const *const c_ulonglong,
    mat2_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_multiply_packed_inplace(
    mat1_addresses: *const *const c_ulonglong,
    mat2_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_divide(
    matrix_1: *const Matrix,
    matrix_2: *const Matrix,
    inplace: bool,
  ) -> Matrix;
  pub fn cuda_element_divide_packed(
    mat1_addresses: *const *const c_ulonglong,
    mat2_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_divide_packed_inplace(
    mat1_addresses: *const *const c_ulonglong,
    mat2_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_scalar_multiply(matrix: *const Matrix, scalar: c_float, inplace: bool) -> Matrix;
  pub fn cuda_scalar_multiply_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalar: f32,
  );
  pub fn cuda_scalar_multiply_packed_inplace(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalar: f32,
  );
  pub fn cuda_scalar_divide(matrix: *const Matrix, scalar: c_float, inplace: bool) -> Matrix;
  pub fn cuda_scalar_divide_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalar: f32,
  );
  pub fn cuda_scalar_divide_packed_inplace(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalar: f32,
  );
  pub fn cuda_scalar_add(matrix: *const Matrix, scalar: c_float, inplace: bool) -> Matrix;
  pub fn cuda_scalar_add_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalar: f32,
  );
  pub fn cuda_scalar_add_packed_inplace(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalar: f32,
  );
  pub fn cuda_scalar_subtract(matrix: *const Matrix, scalar: c_float, inplace: bool) -> Matrix;
  pub fn cuda_scalar_subtract_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalar: f32,
  );
  pub fn cuda_scalar_subtract_packed_inplace(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalar: f32,
  );
  pub fn cuda_matrix_multiply(matrix_1: *const Matrix, matrix_2: *const Matrix) -> Matrix;
  pub fn cuda_add_vector(matrix_1: *const Matrix, matrix_2: *const Matrix, inplace: bool)
    -> Matrix;
  pub fn cuda_divide_by_vector(
    matrix_1: *const Matrix,
    matrix_2: *const Matrix,
    inplace: bool,
  ) -> Matrix;
  pub fn cuda_element_sqrt(matrix: *const Matrix, inplace: bool) -> Matrix;
  pub fn cuda_element_sqrt_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_sqrt_packed_inplace(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_exp(matrix: *const Matrix, inplace: bool) -> Matrix;
  pub fn cuda_element_exp_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_exp_packed_inplace(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_ReLU(matrix: *const Matrix, inplace: bool) -> Matrix;
  pub fn cuda_element_ReLU_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_ReLU_packed_inplace(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_ReLU_prime(matrix: *const Matrix, inplace: bool) -> Matrix;
  pub fn cuda_element_ReLU_prime_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_element_ReLU_prime_packed_inplace(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_sum_rows(matrix: *const Matrix) -> Matrix;
  pub fn cuda_sum_columns(matrix: *const Matrix) -> Matrix;
  pub fn cuda_transpose(matrix: *const Matrix) -> Matrix;
  pub fn cuda_max_pool(matrix: *const Matrix, out_pooled: *mut Matrix, out_bitmask: *mut Matrix);
  pub fn cuda_max_pool_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_pooled: *mut Matrix,
    out_bitmask: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_nearest_neighbor_2x_upsample(matrix: *const Matrix, odd_upsample: bool) -> Matrix;
  pub fn cuda_nearest_neighbor_2x_upsample_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    odd_upsample: bool,
  );
  pub fn cuda_rotate_180(matrix: *const Matrix) -> Matrix;
  pub fn cuda_rotate_180_packed(
    matrix_addresses: *const *const c_ulonglong,
    out_matrices: *mut Matrix,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_correlate(
    matrix: *const Matrix,
    kernel: *const Matrix,
    conv_type: PaddingType,
  ) -> Matrix;
  pub fn cuda_correlate_packed(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    kernel_addresses: *const *const c_ulonglong,
    kernel_rows: usize,
    kernel_cols: usize,
    out_matrices: *mut Matrix,
    conv_type: PaddingType,
  );
  pub fn cuda_convolve(
    matrix: *const Matrix,
    kernel: *const Matrix,
    conv_type: PaddingType,
  ) -> Matrix;
  pub fn cuda_convolve_packed(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    kernel_addresses: *const *const c_ulonglong,
    kernel_rows: usize,
    kernel_cols: usize,
    out_matrices: *mut Matrix,
    conv_type: PaddingType,
  );
  pub fn cuda_img2col(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    kernel_rows: usize,
    kernel_cols: usize,
    conv_type: PaddingType,
  ) -> Matrix;
  pub fn cuda_flatten_array(
    matrix_addresses: *const *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  ) -> Matrix;
  pub fn cuda_unflatten_array(
    array: *const Matrix,
    out_rows: usize,
    out_cols: usize,
    mat_rows: usize,
    out_matrices: *mut Matrix,
  );
  pub fn cuda_unflatten_array_strided(
    array: *const Matrix,
    out_rows: usize,
    out_cols: usize,
    mat_rows: usize,
    out_matrices: *mut Matrix,
  );
  pub fn cuda_center_pad(matrix: *const Matrix, pad_rows: usize, pad_cols: usize) -> Matrix;
  pub fn cuda_softmax(matrix: *const Matrix) -> Matrix;
  pub fn cuda_crop(
    matrix: *const Matrix,
    crop_row_offset: usize,
    crop_col_offset: usize,
    crop_rows: usize,
    crop_cols: usize,
  ) -> Matrix;
  pub fn cuda_copy(matrix: *const Matrix) -> Matrix;
  pub fn cuda_sum_all_matrix_elements(matrix: *const Matrix) -> Matrix;
  pub fn cuda_max_by_column(matrix: *const Matrix) -> Matrix;
  pub fn cuda_max_by_row(matrix: *const Matrix) -> Matrix;
  pub fn cuda_argmax_by_column(matrix: *const Matrix) -> Matrix;
  pub fn cuda_argmax_by_row(matrix: *const Matrix) -> Matrix;
  pub fn cuda_element_ln(matrix: *const Matrix, inplace: bool) -> Matrix;
  pub fn cuda_one_hot_encode_vector(matrix: *const Matrix, num_classes: usize) -> Matrix;
}
