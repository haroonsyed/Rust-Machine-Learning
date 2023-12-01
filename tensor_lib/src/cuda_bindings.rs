use std::ffi::c_float;
use std::ffi::c_ulonglong;

use crate::PaddingType;

#[derive(Clone)]
#[repr(C)] // Used to return a tuple from C
pub struct Tuple {
  pub a: usize,
  pub b: usize,
}

extern "C" {
  pub fn test();
  pub fn test_array_fill(out: *mut c_float, length: usize);
  pub fn cuda_synchronize();
  pub fn register_matrix(rows: usize, cols: usize) -> usize;
  pub fn register_matrix_with_data(data: *const c_float, rows: usize, cols: usize) -> usize;
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
  pub fn cuda_element_add_packed(
    mat1_ids: *const c_ulonglong,
    mat2_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    inplace: bool,
  );
  pub fn cuda_element_subtract(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_subtract_packed(
    mat1_ids: *const c_ulonglong,
    mat2_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    inplace: bool,
  );
  pub fn cuda_element_multiply(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_multiply_packed(
    mat1_ids: *const c_ulonglong,
    mat2_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    inplace: bool,
  );
  pub fn cuda_element_divide(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    mat2_buffer: usize,
    mat2_rows: usize,
    mat2_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_divide_packed(
    mat1_ids: *const c_ulonglong,
    mat2_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    inplace: bool,
  );
  pub fn cuda_scalar_multiply(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    scalar: c_float,
    inplace: bool,
  ) -> usize;
  pub fn cuda_scalar_multiply_packed(
    mat_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalars: *const c_float,
    inplace: bool,
  );
  pub fn cuda_scalar_divide(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    scalar: c_float,
    inplace: bool,
  ) -> usize;
  pub fn cuda_scalar_divide_packed(
    mat_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalars: *const c_float,
    inplace: bool,
  );
  pub fn cuda_scalar_add(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    scalar: c_float,
    inplace: bool,
  ) -> usize;
  pub fn cuda_scalar_add_packed(
    mat_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalars: *const c_float,
    inplace: bool,
  );
  pub fn cuda_scalar_subtract(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    scalar: c_float,
    inplace: bool,
  ) -> usize;
  pub fn cuda_scalar_subtract_packed(
    mat_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    scalars: *const c_float,
    inplace: bool,
  );
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
  pub fn cuda_element_sqrt(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_sqrt_packed(
    mat_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    inplace: bool,
  );
  pub fn cuda_element_exp(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_exp_packed(
    mat_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    inplace: bool,
  );
  pub fn cuda_element_ReLU(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_ReLU_packed(
    mat_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    inplace: bool,
  );
  pub fn cuda_element_ReLU_prime(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    inplace: bool,
  ) -> usize;
  pub fn cuda_element_ReLU_prime_packed(
    mat_ids: *const c_ulonglong,
    out_mat_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    inplace: bool,
  );
  pub fn cuda_sum_rows(mat1_id: usize, mat1_rows: usize, mat1_cols: usize) -> usize;
  pub fn cuda_sum_columns(mat1_id: usize, mat1_rows: usize, mat1_cols: usize) -> usize;
  pub fn cuda_transpose(mat1_id: usize, mat1_rows: usize, mat1_cols: usize) -> usize;
  pub fn cuda_max_pool(mat1_id: usize, mat1_rows: usize, mat1_cols: usize) -> Tuple;
  pub fn cuda_max_pool_packed(
    mat_ids: *const c_ulonglong,
    out_ids: *mut Tuple,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_nearest_neighbor_2x_upsample(
    mat1_id: usize,
    mat1_rows: usize,
    mat2_rows: usize,
    odd_upsample: bool,
  ) -> usize;
  pub fn cuda_nearest_neighbor_2x_upsample_packed(
    mat_ids: *const c_ulonglong,
    out_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    odd_upsample: bool,
  );
  pub fn cuda_rotate_180(mat1_id: usize, mat1_rows: usize, mat1_cols: usize) -> usize;
  pub fn cuda_rotate_180_packed(
    mat_ids: *const c_ulonglong,
    out_ids: *mut c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
  );
  pub fn cuda_convolution(
    mat1_id: usize,
    mat1_rows: usize,
    mat1_cols: usize,
    kernel_id: usize,
    kernel_rows: usize,
    kernel_cols: usize,
    conv_type: PaddingType,
  ) -> usize;
  pub fn cuda_convolution_packed(
    mat_ids: *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    kernel_ids: *const c_ulonglong,
    kernel_rows: usize,
    kernel_cols: usize,
    out_ids: *mut c_ulonglong,
    conv_type: PaddingType,
  );
  pub fn cuda_img2col(
    mat_ids: *const c_ulonglong,
    num_matrices: usize,
    mat_rows: usize,
    mat_cols: usize,
    kernel_rows: usize,
    kernel_cols: usize,
    conv_type: PaddingType,
  ) -> usize;
  pub fn cuda_flatten_array(
    mat_ids: *const c_ulonglong,
    arr_size: usize,
    mat_rows: usize,
    mat_cols: usize,
  ) -> usize;
  pub fn cuda_unflatten_array(
    array_id: usize,
    arr_size: usize,
    mat_rows: usize,
    mat_cols: usize,
    mat_ids: *mut c_ulonglong,
  );
  pub fn cuda_unflatten_array_strided(
    array_id: usize,
    arr_size: usize,
    mat_rows: usize,
    mat_cols: usize,
    mat_ids: *mut c_ulonglong,
  );
  pub fn cuda_center_pad(
    mat_id: usize,
    mat_rows: usize,
    mat_cols: usize,
    pad_rows: usize,
    pad_cols: usize,
  ) -> usize;
  pub fn cuda_softmax(mat_id: usize, mat_rows: usize, mat_cols: usize) -> usize;
  pub fn cuda_crop(
    mat_id: usize,
    mat_rows: usize,
    mat_cols: usize,
    crop_row_offset: usize,
    crop_col_offset: usize,
    crop_rows: usize,
    crop_cols: usize,
  ) -> usize;
  pub fn cuda_copy(mat_id: usize, mat_rows: usize, mat_cols: usize) -> usize;
  pub fn cuda_sum_all_matrix_elements(mat_id: usize, mat_rows: usize, mat_cols: usize) -> usize;
}
