use crate::cuda_bindings::*;
use itertools::Itertools;
use rand::prelude::Distribution;
use statrs::distribution::Normal;
use std::ffi::{c_float, c_ulonglong};
use std::io::{stdout, BufWriter, Write};

// IMPORTANT NOTE: THIS API AND THE UNDERLYING CUDA API ARE TO BE USED WITH A SINGLE THREAD
#[derive(Copy, Clone)]
#[repr(C)]
pub enum PaddingType {
  VALID,
  SAME,
  FULL,
}

// Note cloning keeps the same id, used for ownership movement in rust.
// Does not actually copy the underlying matrix in cuda.
// Cloning is explicit to ensure this understanding
#[repr(C)]
pub struct Matrix {
  id: usize,
  block: usize,
}

impl Clone for Matrix {
  fn clone(&self) -> Self {
    unsafe {
      increase_matrix_ref_count(self as *const Matrix);
    }
    return Matrix {
      id: self.id,
      block: self.block,
    };
  }
}

impl Drop for Matrix {
  fn drop(&mut self) {
    unsafe {
      decrease_matrix_ref_count(self as *const Matrix);
    }
  }
}

impl Matrix {
  fn new(id: usize, block: usize) -> Self {
    return Matrix { id, block };
  }

  pub fn get_id(&self) -> usize {
    return self.id;
  }

  pub fn get_rows(&self) -> usize {
    let rows;
    unsafe {
      rows = get_matrix_rows(self as *const Matrix);
    }
    return rows;
  }

  pub fn get_columns(&self) -> usize {
    let columns;
    unsafe {
      columns = get_matrix_columns(self as *const Matrix);
    }
    return columns;
  }

  pub fn get_data_length(&self) -> usize {
    let data_length;
    unsafe {
      data_length = get_matrix_length(self as *const Matrix);
    }
    return data_length;
  }

  pub fn set_data(&self, data: &Vec<Vec<f32>>) {
    if data.len() == 0 {
      panic!("Cannot set data to empty matrix!");
    }

    if (self.get_rows(), self.get_columns()) != (data.len(), data[0].len()) {
      panic!("Data shape does not match matrix shape!");
    }

    let mut flattened = Vec::<f32>::with_capacity(self.get_data_length());
    data.iter().for_each(|row| flattened.extend(row));

    unsafe {
      upload_matrix_data(self as *const Matrix, flattened.as_ptr());
    }
  }

  pub fn set_data_1d(&self, data: &Vec<f32>) {
    if data.len() != self.get_data_length() {
      panic!("Data shape does not match matrix shape!");
    }

    unsafe {
      upload_matrix_data(self as *const Matrix, data.as_ptr());
    }
  }

  pub fn set_data_from_pinned_buffer_async(&self, pinned_buffer: *mut c_float) {
    unsafe {
      upload_matrix_data_async(self as *const Matrix, pinned_buffer);
    }
  }

  pub fn get_data(&self) -> Vec<Vec<f32>> {
    let mut data = Vec::<c_float>::with_capacity(self.get_data_length());
    unsafe {
      get_matrix_data(self as *const Matrix, data.as_mut_ptr());
      data.set_len(self.get_data_length());
    }

    let mut index = 0;
    let mut result = Vec::new();
    for _ in 0..self.get_rows() {
      let row_slice = data[index..index + self.get_columns() as usize].to_vec();
      result.push(row_slice);
      index += self.get_columns() as usize;
    }

    return result;
  }

  pub fn no_fill(rows: usize, columns: usize) -> Self {
    let result;
    unsafe {
      result = register_matrix(rows, columns);
    }
    return result;
  }

  pub fn zeros(rows: usize, columns: usize) -> Self {
    let data = vec![0.0; rows * columns];
    let result;
    unsafe {
      result = register_matrix_with_data(data.as_ptr(), rows, columns);
    }
    return result;
  }

  pub fn new_1d(data: &Vec<f32>, rows: usize, columns: usize) -> Self {
    if data.len() != rows * columns {
      panic!("Rows and Columns specified not compatible with new_1d size!");
    }

    let result;
    unsafe {
      result = register_matrix_with_data(data.as_ptr(), rows, columns);
    }

    return result;
  }

  pub fn new_2d(data: &Vec<Vec<f32>>) -> Self {
    if data.len() == 0 {
      return Self::no_fill(0, 0);
    }

    let rows = data.len();
    let columns = data[0].len();
    let mut flattened = Vec::<f32>::with_capacity(rows * columns);
    data.iter().for_each(|row| flattened.extend(row));

    let result;
    unsafe {
      result = register_matrix_with_data(flattened.as_ptr(), rows, columns);
    }

    return result;
  }

  pub fn new_random(mean: f64, std: f64, rows: usize, columns: usize) -> Self {
    let mut rng = rand::thread_rng();
    let range = Normal::new(mean, std).unwrap();

    let data = (0..rows * columns)
      .map(|_| range.sample(&mut rng) as f32)
      .collect_vec();

    return Self::new_1d(&data, rows, columns);
  }

  pub fn new_one_hot_encoded(labels: &Vec<f32>, num_classes: usize) -> Self {
    let result;
    unsafe {
      result = cuda_one_hot_encode(labels.as_ptr(), labels.len(), num_classes);
    }

    return result;
  }

  pub fn print(&self) {
    const BUFFER_CAPACITY: usize = 4096 * 1024;
    let stdout = stdout();
    let handle = stdout.lock();
    let mut handle = BufWriter::with_capacity(BUFFER_CAPACITY, handle);

    let data = self.get_data();

    handle.write(b"\n").unwrap();

    for row in 0..self.get_rows() {
      for index in 0..self.get_columns() {
        let val = data[row][index];
        let formatted = format!(" {:<5} ", val);
        handle.write(formatted.as_bytes()).unwrap();
      }
      handle.write(b"\n").unwrap();
    }

    handle.write(b"\n").unwrap();
  }

  pub fn print_shape(&self) {
    println!(
      "Shape of matrix is: {} {}",
      self.get_rows(),
      self.get_columns()
    );
  }

  pub fn same_shape(&self, other: &Matrix) -> bool {
    return self.get_rows() == other.get_rows() && self.get_columns() == other.get_columns();
  }

  fn element_add_impl(&self, other: &Matrix, inplace: bool) -> Matrix {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for addition! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result;
    unsafe { result = cuda_element_add(self as *const Matrix, other as *const Matrix, inplace) }

    return result;
  }

  pub fn element_add(&self, other: &Matrix) -> Self {
    return self.element_add_impl(other, false);
  }

  pub fn element_add_inplace(&self, other: &Matrix) -> Self {
    self.element_add_impl(other, true);
    return self.clone();
  }

  fn element_subtract_impl(&self, other: &Matrix, inplace: bool) -> Matrix {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_subtract! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result;
    unsafe {
      result = cuda_element_subtract(self as *const Matrix, other as *const Matrix, inplace)
    }

    return result;
  }

  pub fn element_subtract(&self, other: &Matrix) -> Self {
    return self.element_subtract_impl(other, false);
  }

  pub fn element_subtract_inplace(&self, other: &Matrix) -> Self {
    self.element_subtract_impl(other, true);
    return self.clone();
  }

  fn element_multiply_impl(&self, other: &Matrix, inplace: bool) -> Matrix {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_multiply! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result;
    unsafe {
      result = cuda_element_multiply(self as *const Matrix, other as *const Matrix, inplace)
    }

    return result;
  }

  pub fn element_multiply(&self, other: &Matrix) -> Self {
    return self.element_multiply_impl(other, false);
  }

  pub fn element_multiply_inplace(&self, other: &Matrix) -> Self {
    self.element_multiply_impl(other, true);
    return self.clone();
  }

  fn element_divide_impl(&self, other: &Matrix, inplace: bool) -> Matrix {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_divide! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result;
    unsafe { result = cuda_element_divide(self as *const Matrix, other as *const Matrix, inplace) }

    return result;
  }

  pub fn element_divide(&self, other: &Matrix) -> Self {
    return self.element_divide_impl(other, false);
  }

  pub fn element_divide_inplace(&self, other: &Matrix) -> Self {
    self.element_divide_impl(other, true);
    return self.clone();
  }

  fn scalar_multiply_impl(&self, scalar: f32, inplace: bool) -> Matrix {
    let result;
    unsafe { result = cuda_scalar_multiply(self as *const Matrix, scalar, inplace) }

    return result;
  }

  pub fn scalar_multiply(&self, scalar: f32) -> Self {
    return self.scalar_multiply_impl(scalar, false);
  }

  pub fn scalar_multiply_inplace(&self, scalar: f32) -> Self {
    self.scalar_multiply_impl(scalar, true);
    return self.clone();
  }

  fn scalar_divide_impl(&self, scalar: f32, inplace: bool) -> Matrix {
    let result;
    unsafe { result = cuda_scalar_divide(self as *const Matrix, scalar, inplace) }

    return result;
  }

  pub fn scalar_divide(&self, scalar: f32) -> Self {
    return self.scalar_divide_impl(scalar, false);
  }

  pub fn scalar_divide_inplace(&self, scalar: f32) -> Self {
    self.scalar_divide_impl(scalar, true);
    return self.clone();
  }

  fn scalar_add_impl(&self, scalar: f32, inplace: bool) -> Matrix {
    let result;
    unsafe { result = cuda_scalar_add(self as *const Matrix, scalar, inplace) }

    return result;
  }

  pub fn scalar_add(&self, scalar: f32) -> Self {
    return self.scalar_add_impl(scalar, false);
  }

  pub fn scalar_add_inplace(&self, scalar: f32) -> Self {
    self.scalar_add_impl(scalar, true);
    return self.clone();
  }

  fn scalar_subtract_impl(&self, scalar: f32, inplace: bool) -> Matrix {
    let result;
    unsafe { result = cuda_scalar_subtract(self as *const Matrix, scalar, inplace) }

    return result;
  }

  pub fn scalar_subtract(&self, scalar: f32) -> Self {
    return self.scalar_subtract_impl(scalar, false);
  }

  pub fn scalar_subtract_inplace(&self, scalar: f32) -> Self {
    self.scalar_subtract_impl(scalar, true);
    return self.clone();
  }

  pub fn matrix_multiply(&self, other: &Matrix) -> Matrix {
    // Bound Check
    if self.get_columns() != other.get_rows() {
      panic!(
        "Matrices not compatible shape for mat mult! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result;
    unsafe { result = cuda_matrix_multiply(self as *const Matrix, other as *const Matrix) }

    return result;
  }

  fn add_vector_impl(&self, other: &Matrix, inplace: bool) -> Matrix {
    if !((self.get_rows() == other.get_rows() && other.get_columns() == 1)
      || (self.get_columns() == other.get_columns() && other.get_rows() == 1))
    {
      panic!(
        "Matrices not the correct shape for vector add! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result;
    unsafe { result = cuda_add_vector(self as *const Matrix, other as *const Matrix, inplace) }

    return result;
  }

  pub fn add_vector(&self, other: &Matrix) -> Self {
    return self.add_vector_impl(other, false);
  }

  pub fn add_vector_inplace(&self, other: &Matrix) -> Self {
    self.add_vector_impl(other, true);
    return self.clone();
  }

  fn divide_by_vector_impl(&self, other: &Matrix, inplace: bool) -> Matrix {
    if !((self.get_rows() == other.get_rows() && other.get_columns() == 1)
      || (self.get_columns() == other.get_columns() && other.get_rows() == 1))
    {
      panic!(
        "Matrices not the correct shape for division by vector! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result;
    unsafe {
      result = cuda_divide_by_vector(self as *const Matrix, other as *const Matrix, inplace)
    }

    return result;
  }

  pub fn divide_by_vector(&self, other: &Matrix) -> Self {
    return self.divide_by_vector_impl(other, false);
  }

  pub fn divide_by_vector_inplace(&self, other: &Matrix) -> Self {
    self.divide_by_vector_impl(other, true);
    return self.clone();
  }

  fn element_sqrt_impl(&self, inplace: bool) -> Matrix {
    let result;
    unsafe { result = cuda_element_sqrt(self as *const Matrix, inplace) }
    return result;
  }

  pub fn element_sqrt(&self) -> Self {
    return self.element_sqrt_impl(false);
  }

  pub fn element_sqrt_inplace(&self) -> Self {
    self.element_sqrt_impl(true);
    return self.clone();
  }

  fn element_exp_impl(&self, inplace: bool) -> Matrix {
    let result;
    unsafe { result = cuda_element_exp(self as *const Matrix, inplace) }
    return result;
  }

  pub fn element_exp(&self) -> Self {
    return self.element_exp_impl(false);
  }

  pub fn element_exp_inplace(&self) -> Self {
    self.element_exp_impl(true);
    return self.clone();
  }

  #[allow(non_snake_case)]
  fn element_ReLU_impl(&self, inplace: bool) -> Matrix {
    let result;
    unsafe { result = cuda_element_ReLU(self as *const Matrix, inplace) }

    return result;
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU(&self) -> Self {
    return self.element_ReLU_impl(false);
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_inplace(&self) -> Self {
    self.element_ReLU_impl(true);
    return self.clone();
  }

  #[allow(non_snake_case)]
  fn element_ReLU_prime_impl(&self, inplace: bool) -> Matrix {
    let result;
    unsafe { result = cuda_element_ReLU_prime(self as *const Matrix, inplace) }

    return result;
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime(&self) -> Self {
    return self.element_ReLU_prime_impl(false);
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime_inplace(&self) -> Self {
    self.element_ReLU_prime_impl(true);
    return self.clone();
  }

  pub fn sum_rows_matrix(&self) -> Self {
    let result;
    unsafe { result = cuda_sum_rows(self as *const Matrix) }

    return result;
  }

  pub fn sum_columns_matrix(&self) -> Self {
    let result;
    unsafe { result = cuda_sum_columns(self as *const Matrix) }

    return result;
  }

  pub fn sum_columns(&self) -> Vec<f32> {
    let result;
    unsafe { result = cuda_sum_columns(self as *const Matrix) }

    let result = result.get_data()[0].to_vec();
    return result;
  }

  pub fn transpose(&self) -> Self {
    let result;

    // Fast transpose
    if self.get_rows() == 1 || self.get_columns() == 1 {
      let result = self.clone();

      unsafe {
        reshape_matrix(
          self as *const Matrix,
          result.get_columns(),
          result.get_rows(),
        );
      }

      return result;
    }

    unsafe { result = cuda_transpose(self as *const Matrix) }
    return result;
  }

  pub fn max_pool(&self) -> (Self, Self) {
    let mut pooled = Matrix { id: 0, block: 0 };
    let mut bitmask = Matrix { id: 0, block: 0 };
    unsafe {
      cuda_max_pool(
        self as *const Matrix,
        &mut pooled as *mut Matrix,
        &mut bitmask as *mut Matrix,
      );
    }
    return (pooled, bitmask);
  }

  pub fn nearest_neighbor_2x_upsample(&self, odd_upsample: bool) -> Self {
    let result;
    unsafe { result = cuda_nearest_neighbor_2x_upsample(self as *const Matrix, odd_upsample) }
    return result;
  }

  pub fn rotate_180(&self) -> Self {
    let result;
    unsafe { result = cuda_rotate_180(self as *const Matrix) }
    return result;
  }

  pub fn correlate(&self, kernel: &Matrix, padding_type: PaddingType) -> Self {
    let result;

    if matches!(padding_type, PaddingType::SAME)
      && (kernel.get_rows() != kernel.get_columns() || kernel.get_rows() % 2 == 0)
    {
      panic!("Kernel must be square and odd for same convolution!");
    }

    unsafe { result = cuda_correlate(self as *const Matrix, kernel as *const Matrix, padding_type) }
    return result;
  }

  pub fn convolve(&self, kernel: &Matrix, padding_type: PaddingType) -> Self {
    let result;

    if matches!(padding_type, PaddingType::SAME)
      && (kernel.get_rows() != kernel.get_columns() || kernel.get_rows() % 2 == 0)
    {
      panic!("Kernel must be square and odd for same convolution!");
    }

    unsafe { result = cuda_convolve(self as *const Matrix, kernel as *const Matrix, padding_type) }
    return result;
  }

  pub fn center_pad(&self, pad_rows: usize, pad_cols: usize) -> Self {
    let result;
    unsafe { result = cuda_center_pad(self as *const Matrix, pad_rows, pad_cols) }
    return result;
  }

  pub fn softmax(&self) -> Self {
    let result;
    unsafe { result = cuda_softmax(self as *const Matrix) }
    return result;
  }

  pub fn crop(&self, row_offset: usize, column_offset: usize, rows: usize, columns: usize) -> Self {
    let result;
    unsafe {
      result = cuda_crop(
        self as *const Matrix,
        row_offset,
        column_offset,
        rows,
        columns,
      )
    }
    return result;
  }

  pub fn deep_copy(&self) -> Self {
    let result;
    unsafe { result = cuda_copy(self as *const Matrix) }
    return result;
  }

  // pub fn cuda_sum_all_matrix_elements(mat_id: usize, mat_rows: usize, mat_cols: usize) -> usize;
  pub fn sum_all_matrix_elements(&self) -> Self {
    let result;
    unsafe { result = cuda_sum_all_matrix_elements(self as *const Matrix) }
    return result;
  }

  pub fn max_by_column(&self) -> Self {
    let result;
    unsafe { result = cuda_max_by_column(self as *const Matrix) }
    return result;
  }

  pub fn max_by_row(&self) -> Self {
    let result;
    unsafe { result = cuda_max_by_row(self as *const Matrix) }
    return result;
  }

  pub fn argmax_by_column(&self) -> Self {
    let result;
    unsafe { result = cuda_argmax_by_column(self as *const Matrix) }
    return result;
  }

  pub fn argmax_by_row(&self) -> Self {
    let result;
    unsafe { result = cuda_argmax_by_row(self as *const Matrix) }
    return result;
  }

  pub fn element_ln(&self) -> Self {
    let result;
    unsafe { result = cuda_element_ln(self as *const Matrix, false) }
    return result;
  }

  pub fn element_ln_inplace(&self) -> &Self {
    unsafe {
      cuda_element_ln(self as *const Matrix, true);
    }
    return self;
  }

  pub fn one_hot_encode(&self, num_classes: usize) -> Self {
    let result;

    // Check that this is a vector
    if !self.get_columns() == 1 && !self.get_rows() == 1 {
      panic!("Cannot one hot encode non vector!");
    }

    unsafe { result = cuda_one_hot_encode_vector(self as *const Matrix, num_classes) }
    return result;
  }

  pub fn reshape(&mut self, new_rows: usize, new_columns: usize) -> &Self {
    unsafe {
      reshape_matrix(self as *const Matrix, new_rows, new_columns);
    }
    return self;
  }

  pub fn to_one_dimensional(&mut self) -> &Self {
    unsafe {
      reshape_matrix(self as *const Matrix, self.get_data_length(), 1);
    }
    return self;
  }
}

pub fn create_matrix_group(rows: usize, columns: usize, count: usize) -> Vec<Matrix> {
  let mut matrices = Vec::with_capacity(count);

  unsafe {
    matrices.set_len(count);
  }

  unsafe {
    register_matrix_group(rows, columns, count, matrices.as_mut_ptr() as *mut Matrix);
  }
  return matrices;
}

// All matrices are required to be the same shape
pub fn flatten_matrix_array(to_flatten: &[usize], mat_rows: usize, mat_cols: usize) -> Matrix {
  if to_flatten.len() == 0 {
    return Matrix::zeros(0, 0);
  }

  let result;
  unsafe {
    result = cuda_flatten_array(
      to_flatten.as_ptr() as *const *const u64,
      to_flatten.len(),
      mat_rows,
      mat_cols,
    )
  };

  return result;
}

// Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
pub fn img2col(
  image_ids: &[usize],
  image_depth: usize,
  image_rows: usize,
  image_cols: usize,
  filter_rows: usize,
  filter_cols: usize,
) -> Matrix {
  let result;
  unsafe {
    result = cuda_img2col(
      image_ids.as_ptr() as *const *const u64,
      image_depth,
      image_rows,
      image_cols,
      filter_rows,
      filter_cols,
      PaddingType::VALID,
    )
  };

  return result;
}

// All matrices are required to be the same shape
pub fn unflatten_array_to_matrices(
  to_unflatten: &Matrix,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = to_unflatten.get_data_length() / (mat_rows * mat_cols);
  if to_unflatten.get_data_length() != num_matrices * mat_rows * mat_cols {
    panic!("Cannot unflatten array, matrix dimensions incorrect!")
  }

  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
  }

  unsafe {
    cuda_unflatten_array(
      to_unflatten as *const Matrix,
      to_unflatten.get_data_length(),
      mat_rows,
      mat_cols,
      results.as_mut_ptr(),
    )
  };
  return results;
}

// All matrices are required to be the same shape. Each array's first n elements are the first elements in memory. [arr1_elem1, arr2_elem1, arr3_elem1, arr1_elem2, arr2_elem2, arr3_elem2, ...]
pub fn unflatten_array_strided_to_matrices(
  to_unflatten: &Matrix,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = to_unflatten.get_data_length() / (mat_rows * mat_cols);
  if to_unflatten.get_data_length() != num_matrices * mat_rows * mat_cols {
    panic!("Cannot unflatten array, matrix dimensions incorrect!")
  }

  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
  }

  unsafe {
    cuda_unflatten_array_strided(
      to_unflatten as *const Matrix,
      to_unflatten.get_data_length(),
      mat_rows,
      mat_cols,
      results.as_mut_ptr(),
    )
  };
  return results;
}

pub fn element_add_packed(
  mat_1_addresses: &[usize],
  mat_2_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  if mat_1_addresses.len() != mat_2_addresses.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1_addresses.len(),
      mat_2_addresses.len()
    );
  }

  let num_matrices = mat_1_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_element_add_packed(
      mat_1_addresses.as_ptr() as *const *const u64,
      mat_2_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
  return results;
}

pub fn element_add_packed_inplace(
  mat_1_addresses: &[usize],
  mat_2_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) {
  if mat_1_addresses.len() != mat_2_addresses.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1_addresses.len(),
      mat_2_addresses.len()
    );
  }

  let num_matrices = mat_1_addresses.len();
  unsafe {
    cuda_element_add_packed_inplace(
      mat_1_addresses.as_ptr() as *const *const u64,
      mat_2_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn element_subtract_packed(
  mat_1_addresses: &[usize],
  mat_2_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = mat_1_addresses.len();
  if mat_1_addresses.len() != mat_2_addresses.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1_addresses.len(),
      mat_2_addresses.len()
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_element_subtract_packed(
      mat_1_addresses.as_ptr() as *const *const u64,
      mat_2_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_subtract_packed_inplace(
  mat_1_addresses: &[usize],
  mat_2_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) {
  let num_matrices = mat_1_addresses.len();
  if mat_1_addresses.len() != mat_2_addresses.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1_addresses.len(),
      mat_2_addresses.len()
    );
  }

  unsafe {
    cuda_element_subtract_packed_inplace(
      mat_1_addresses.as_ptr() as *const *const u64,
      mat_2_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn element_multiply_packed(
  mat_1_addresses: &[usize],
  mat_2_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = mat_1_addresses.len();
  if mat_1_addresses.len() != mat_2_addresses.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1_addresses.len(),
      mat_2_addresses.len()
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_element_multiply_packed(
      mat_1_addresses.as_ptr() as *const *const u64,
      mat_2_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_multiply_packed_inplace(
  mat_1_addresses: &[usize],
  mat_2_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) {
  let num_matrices = mat_1_addresses.len();
  if mat_1_addresses.len() != mat_2_addresses.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1_addresses.len(),
      mat_2_addresses.len()
    );
  }

  unsafe {
    cuda_element_multiply_packed_inplace(
      mat_1_addresses.as_ptr() as *const *const u64,
      mat_2_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn element_divide_packed(
  mat_1_addresses: &[usize],
  mat_2_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = mat_1_addresses.len();
  if mat_1_addresses.len() != mat_2_addresses.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1_addresses.len(),
      mat_2_addresses.len()
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_element_divide_packed(
      mat_1_addresses.as_ptr() as *const *const u64,
      mat_2_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_divide_packed_inplace(
  mat_1_addresses: &[usize],
  mat_2_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) {
  let num_matrices = mat_1_addresses.len();
  if mat_1_addresses.len() != mat_2_addresses.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1_addresses.len(),
      mat_2_addresses.len()
    );
  }

  unsafe {
    cuda_element_divide_packed_inplace(
      mat_1_addresses.as_ptr() as *const *const u64,
      mat_2_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn scalar_multiply_packed(
  matrix_addresses: &[usize],
  scalar: f32,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_scalar_multiply_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }

  return results;
}

pub fn scalar_multiply_packed_inplace(
  matrix_addresses: &[usize],
  scalar: f32,
  mat_rows: usize,
  mat_cols: usize,
) {
  let num_matrices = matrix_addresses.len();
  unsafe {
    cuda_scalar_multiply_packed_inplace(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }
}

pub fn scalar_divide_packed(
  matrix_addresses: &[usize],
  scalar: f32,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_scalar_divide_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }

  return results;
}

pub fn scalar_divide_packed_inplace(
  matrix_addresses: &[usize],
  scalar: f32,
  mat_rows: usize,
  mat_cols: usize,
) {
  let num_matrices = matrix_addresses.len();
  unsafe {
    cuda_scalar_divide_packed_inplace(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }
}

pub fn scalar_add_packed(
  matrix_addresses: &[usize],
  scalar: f32,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_scalar_add_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }

  return results;
}

pub fn scalar_add_packed_inplace(
  matrix_addresses: &[usize],
  scalar: f32,
  mat_rows: usize,
  mat_cols: usize,
) {
  let num_matrices = matrix_addresses.len();
  unsafe {
    cuda_scalar_add_packed_inplace(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }
}

pub fn scalar_subtract_packed(
  matrix_addresses: &[usize],
  scalar: f32,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_scalar_subtract_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }

  return results;
}

pub fn scalar_subtract_packed_inplace(
  matrix_addresses: &[usize],
  scalar: f32,
  mat_rows: usize,
  mat_cols: usize,
) {
  let num_matrices = matrix_addresses.len();
  unsafe {
    cuda_scalar_subtract_packed_inplace(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }
}

pub fn element_sqrt_packed(
  matrix_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_element_sqrt_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_sqrt_packed_inplace(matrix_addresses: &[usize], mat_rows: usize, mat_cols: usize) {
  let num_matrices = matrix_addresses.len();
  unsafe {
    cuda_element_sqrt_packed_inplace(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn element_exp_packed(
  matrix_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_element_exp_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_exp_packed_inplace(matrix_addresses: &[usize], mat_rows: usize, mat_cols: usize) {
  let num_matrices = matrix_addresses.len();
  unsafe {
    cuda_element_exp_packed_inplace(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

#[allow(non_snake_case)]
pub fn element_ReLU_packed(
  matrix_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_element_ReLU_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

#[allow(non_snake_case)]
pub fn element_ReLU_packed_inplace(matrix_addresses: &[usize], mat_rows: usize, mat_cols: usize) {
  let num_matrices = matrix_addresses.len();
  unsafe {
    cuda_element_ReLU_packed_inplace(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

#[allow(non_snake_case)]
pub fn element_ReLU_prime_packed(
  matrix_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_element_ReLU_prime_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

#[allow(non_snake_case)]
pub fn element_ReLU_prime_packed_inplace(
  matrix_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) {
  let num_matrices = matrix_addresses.len();
  unsafe {
    cuda_element_ReLU_prime_packed_inplace(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn max_pool_packed(
  matrix_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> (Vec<Matrix>, Vec<Matrix>) {
  let num_matrices = matrix_addresses.len();
  if num_matrices == 0 {
    return (Vec::new(), Vec::new());
  }

  let mut pooled_results = Vec::with_capacity(num_matrices);
  let mut bitmask_results = Vec::with_capacity(num_matrices);
  unsafe {
    cuda_max_pool_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      pooled_results.as_mut_ptr(),
      bitmask_results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return (pooled_results, bitmask_results);
}

pub fn nearest_neighbor_2x_upsample_packed(
  matrix_addresses: &[usize],
  odd_upsample: bool,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  if num_matrices == 0 {
    return Vec::new();
  }

  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    cuda_nearest_neighbor_2x_upsample_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
      odd_upsample,
    );
  }

  return results;
}

pub fn rotate_180_packed(
  matrix_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_rotate_180_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      results.as_mut_ptr(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn correlate_packed(
  matrix_addresses: &[usize],
  kernel_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
  kernel_rows: usize,
  kernel_cols: usize,
  padding_type: PaddingType,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let num_kernels = kernel_addresses.len();

  if num_matrices != num_kernels {
    panic!(
      "Number of matrices must be equal to number of kernels! {} {}",
      num_matrices, num_kernels
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  if matches!(padding_type, PaddingType::SAME)
    && (kernel_rows != kernel_cols || kernel_rows % 2 == 0)
  {
    panic!("Kernel must be square and odd for same correlation!");
  }

  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_correlate_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
      kernel_addresses.as_ptr() as *const *const u64,
      kernel_rows,
      kernel_cols,
      results.as_mut_ptr(),
      padding_type,
    );
  }
  return results;
}

pub fn convolve_packed(
  matrix_addresses: &[usize],
  kernel_addresses: &[usize],
  mat_rows: usize,
  mat_cols: usize,
  kernel_rows: usize,
  kernel_cols: usize,
  padding_type: PaddingType,
) -> Vec<Matrix> {
  let num_matrices = matrix_addresses.len();
  let num_kernels = kernel_addresses.len();

  if num_matrices != num_kernels {
    panic!(
      "Number of matrices must be equal to number of kernels! {} {}",
      num_matrices, num_kernels
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  if matches!(padding_type, PaddingType::SAME)
    && (kernel_rows != kernel_cols || kernel_rows % 2 == 0)
  {
    panic!("Kernel must be square and odd for same correlation!");
  }

  let mut results = Vec::with_capacity(num_matrices);
  unsafe {
    results.set_len(num_matrices);
    cuda_convolve_packed(
      matrix_addresses.as_ptr() as *const *const u64,
      num_matrices,
      mat_rows,
      mat_cols,
      kernel_addresses.as_ptr() as *const *const u64,
      kernel_rows,
      kernel_cols,
      results.as_mut_ptr(),
      padding_type,
    );
  }
  return results;
}
