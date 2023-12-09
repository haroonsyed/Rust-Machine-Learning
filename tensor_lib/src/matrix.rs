use crate::cuda_bindings::*;
use itertools::Itertools;
use rand::prelude::Distribution;
use statrs::distribution::Normal;
use std::ffi::{c_float, c_ulonglong};
use std::io::{stdout, BufWriter, Write};
use std::ptr::null_mut;
use std::time::Instant;

// IMPORTANT NOTE: THIS API IS THREADSAFE, BUT THE UNDERLYING CUDA LIBRARY IS NOT THREAD SAFE (yet).

#[derive(Copy, Clone)]
#[repr(C)]
pub enum PaddingType {
  VALID,
  SAME,
  FULL,
}

// Use newtype for id to ensure drop arc<id> does not apply to all usize
#[repr(C)]
struct MatrixId(usize);

impl MatrixId {
  fn new(id: usize) -> Self {
    return MatrixId(id);
  }
}

impl Clone for MatrixId {
  fn clone(&self) -> Self {
    unsafe {
      increase_matrix_ref_count(self.0);
    }
    return MatrixId(self.0);
  }
}

impl Drop for MatrixId {
  fn drop(&mut self) {
    unsafe {
      decrease_matrix_ref_count(self.0);
    }
  }
}

// Note cloning keeps the same id, used for ownership movement in rust.
// Does not actually copy the underlying matrix in cuda.
// Cloning is explicit to ensure this understanding
#[derive(Clone)]
pub struct Matrix {
  id: MatrixId,
}

impl Matrix {
  fn new(id: usize) -> Self {
    return Matrix {
      id: MatrixId::new(id),
    };
  }

  pub fn get_id(&self) -> usize {
    // return self.id.0;
    return self.id.0;
  }

  pub fn get_rows(&self) -> usize {
    let rows;
    unsafe {
      rows = get_matrix_rows(self.get_id());
    }
    return rows;
  }

  pub fn get_columns(&self) -> usize {
    let columns;
    unsafe {
      columns = get_matrix_columns(self.get_id());
    }
    return columns;
  }

  pub fn get_data_length(&self) -> usize {
    let data_length;
    unsafe {
      data_length = get_matrix_length(self.get_id());
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
      upload_matrix_data(self.get_id(), flattened.as_ptr());
    }
  }

  pub fn set_data_1d(&self, data: &Vec<f32>) {
    if data.len() != self.get_data_length() {
      panic!("Data shape does not match matrix shape!");
    }

    unsafe {
      upload_matrix_data(self.get_id(), data.as_ptr());
    }
  }

  pub fn get_data(&self) -> Vec<Vec<f32>> {
    let mut data = Vec::<c_float>::with_capacity(self.get_data_length());
    unsafe {
      get_matrix_data(self.get_id(), data.as_mut_ptr());
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
    let id;
    unsafe {
      id = register_matrix(rows, columns);
    }
    return Matrix::new(id);
  }

  pub fn zeros(rows: usize, columns: usize) -> Self {
    let data = vec![0.0; rows * columns];
    let id;
    unsafe {
      id = register_matrix_with_data(data.as_ptr(), rows, columns);
    }
    return Matrix::new(id);
  }

  pub fn new_1d(data: &Vec<f32>, rows: usize, columns: usize) -> Self {
    if data.len() != rows * columns {
      panic!("Rows and Columns specified not compatible with new_1d size!");
    }

    let id;
    unsafe {
      id = register_matrix_with_data(data.as_ptr(), rows, columns);
    }

    return Matrix::new(id);
  }

  pub fn new_2d(data: &Vec<Vec<f32>>) -> Self {
    if data.len() == 0 {
      return Self::no_fill(0, 0);
    }

    let rows = data.len();
    let columns = data[0].len();
    let mut flattened = Vec::<f32>::with_capacity(rows * columns);
    data.iter().for_each(|row| flattened.extend(row));

    let id;
    unsafe {
      id = register_matrix_with_data(flattened.as_ptr(), rows, columns);
    }

    return Matrix::new(id);
  }

  pub fn new_random(mean: f64, std: f64, rows: usize, columns: usize) -> Self {
    let mut rng = rand::thread_rng();
    let range = Normal::new(mean, std).unwrap();

    let data = (0..rows)
      .map(|_| {
        (0..columns)
          .map(|_| range.sample(&mut rng) as f32)
          .collect_vec()
      })
      .collect_vec();

    return Self::new_2d(&data);
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

  fn element_add_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for addition! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_element_add(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        other.get_id(),
        other.get_rows(),
        other.get_columns(),
        inplace,
      )
    }

    return result_id;
  }

  pub fn element_add(&self, other: &Matrix) -> Self {
    let result_id = self.element_add_impl(other, false);

    return Matrix::new(result_id);
  }

  pub fn element_add_inplace(&self, other: &Matrix) -> Self {
    self.element_add_impl(other, true);
    return self.clone();
  }

  fn element_subtract_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_subtract! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_element_subtract(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        other.get_id(),
        other.get_rows(),
        other.get_columns(),
        inplace,
      )
    }

    return result_id;
  }

  pub fn element_subtract(&self, other: &Matrix) -> Self {
    let result_id = self.element_subtract_impl(other, false);

    return Matrix::new(result_id);
  }

  pub fn element_subtract_inplace(&self, other: &Matrix) -> Self {
    self.element_subtract_impl(other, true);
    return self.clone();
  }

  fn element_multiply_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_multiply! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_element_multiply(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        other.get_id(),
        other.get_rows(),
        other.get_columns(),
        inplace,
      )
    }

    return result_id;
  }

  pub fn element_multiply(&self, other: &Matrix) -> Self {
    let result_id = self.element_multiply_impl(other, false);

    return Matrix::new(result_id);
  }

  pub fn element_multiply_inplace(&self, other: &Matrix) -> Self {
    self.element_multiply_impl(other, true);
    return self.clone();
  }

  fn element_divide_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_divide! A: {} {} B: {} {}",
        self.get_rows(),
        self.get_columns(),
        other.get_rows(),
        other.get_columns()
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_element_divide(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        other.get_id(),
        other.get_rows(),
        other.get_columns(),
        inplace,
      )
    }

    return result_id;
  }

  pub fn element_divide(&self, other: &Matrix) -> Self {
    let result_id = self.element_divide_impl(other, false);

    return Matrix::new(result_id);
  }

  pub fn element_divide_inplace(&self, other: &Matrix) -> Self {
    self.element_divide_impl(other, true);
    return self.clone();
  }

  fn scalar_multiply_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_scalar_multiply(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        scalar,
        inplace,
      )
    }

    return result_id;
  }

  pub fn scalar_multiply(&self, scalar: f32) -> Self {
    let result_id = self.scalar_multiply_impl(scalar, false);

    return Matrix::new(result_id);
  }

  pub fn scalar_multiply_inplace(&self, scalar: f32) -> Self {
    self.scalar_multiply_impl(scalar, true);
    return self.clone();
  }

  fn scalar_divide_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_scalar_divide(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        scalar,
        inplace,
      )
    }

    return result_id;
  }

  pub fn scalar_divide(&self, scalar: f32) -> Self {
    let result_id = self.scalar_divide_impl(scalar, false);

    return Matrix::new(result_id);
  }

  pub fn scalar_divide_inplace(&self, scalar: f32) -> Self {
    self.scalar_divide_impl(scalar, true);
    return self.clone();
  }

  fn scalar_add_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_scalar_add(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        scalar,
        inplace,
      )
    }

    return result_id;
  }

  pub fn scalar_add(&self, scalar: f32) -> Self {
    let result_id = self.scalar_add_impl(scalar, false);

    return Matrix::new(result_id);
  }

  pub fn scalar_add_inplace(&self, scalar: f32) -> Self {
    self.scalar_add_impl(scalar, true);
    return self.clone();
  }

  fn scalar_subtract_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_scalar_subtract(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        scalar,
        inplace,
      )
    }

    return result_id;
  }

  pub fn scalar_subtract(&self, scalar: f32) -> Self {
    let result_id = self.scalar_subtract_impl(scalar, false);

    return Matrix::new(result_id);
  }

  pub fn scalar_subtract_inplace(&self, scalar: f32) -> Self {
    self.scalar_subtract_impl(scalar, true);
    return self.clone();
  }

  pub fn matrix_multiply(&self, other: &Matrix) -> Self {
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

    let result_id: usize;
    unsafe {
      result_id = cuda_matrix_multiply(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        other.get_id(),
        other.get_rows(),
        other.get_columns(),
      )
    }

    return Matrix::new(result_id);
  }

  fn add_vector_impl(&self, other: &Matrix, inplace: bool) -> usize {
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

    let result_id: usize;
    unsafe {
      result_id = cuda_add_vector(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        other.get_id(),
        other.get_rows(),
        other.get_columns(),
        inplace,
      )
    }

    return result_id;
  }

  pub fn add_vector(&self, other: &Matrix) -> Self {
    let result_id = self.add_vector_impl(other, false);

    return Matrix::new(result_id);
  }

  pub fn add_vector_inplace(&self, other: &Matrix) -> Self {
    self.add_vector_impl(other, true);

    return self.clone();
  }

  fn divide_by_vector_impl(&self, other: &Matrix, inplace: bool) -> usize {
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

    let result_id: usize;
    unsafe {
      result_id = cuda_divide_by_vector(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        other.get_id(),
        other.get_rows(),
        other.get_columns(),
        inplace,
      )
    }

    return result_id;
  }

  pub fn divide_by_vector(&self, other: &Matrix) -> Self {
    let result_id = self.divide_by_vector_impl(other, false);

    return Matrix::new(result_id);
  }

  pub fn divide_by_vector_inplace(&self, other: &Matrix) -> Self {
    self.divide_by_vector_impl(other, true);

    return self.clone();
  }

  fn element_sqrt_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_element_sqrt(self.get_id(), self.get_rows(), self.get_columns(), inplace)
    }
    return result_id;
  }

  pub fn element_sqrt(&self) -> Self {
    let result_id = self.element_sqrt_impl(false);

    return Matrix::new(result_id);
  }

  pub fn element_sqrt_inplace(&self) -> Self {
    self.element_sqrt_impl(true);

    return self.clone();
  }

  fn element_exp_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_element_exp(self.get_id(), self.get_rows(), self.get_columns(), inplace)
    }
    return result_id;
  }

  pub fn element_exp(&self) -> Self {
    let result_id = self.element_exp_impl(false);

    return Matrix::new(result_id);
  }

  pub fn element_exp_inplace(&self) -> Self {
    self.element_exp_impl(true);

    return self.clone();
  }

  #[allow(non_snake_case)]
  fn element_ReLU_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_element_ReLU(self.get_id(), self.get_rows(), self.get_columns(), inplace)
    }

    return result_id;
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU(&self) -> Self {
    let result_id = self.element_ReLU_impl(false);

    return Matrix::new(result_id);
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_inplace(&self) -> Self {
    self.element_ReLU_impl(true);

    return self.clone();
  }

  #[allow(non_snake_case)]
  fn element_ReLU_prime_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id =
        cuda_element_ReLU_prime(self.get_id(), self.get_rows(), self.get_columns(), inplace)
    }

    return result_id;
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime(&self) -> Self {
    let result_id = self.element_ReLU_prime_impl(false);

    return Matrix::new(result_id);
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime_inplace(&self) -> Self {
    self.element_ReLU_prime_impl(true);

    return self.clone();
  }

  pub fn sum_rows_matrix(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_sum_rows(self.get_id(), self.get_rows(), self.get_columns()) }

    return Matrix::new(result_id);
  }

  pub fn sum_columns_matrix(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_sum_columns(self.get_id(), self.get_rows(), self.get_columns()) }

    return Matrix::new(result_id);
  }

  pub fn sum_columns(&self) -> Vec<f32> {
    let result_id: usize;
    unsafe { result_id = cuda_sum_columns(self.get_id(), self.get_rows(), self.get_columns()) }

    let result_matrix = Matrix::new(result_id);

    let result = result_matrix.get_data()[0].to_vec();
    return result;
  }

  pub fn transpose(&self) -> Self {
    let result_id: usize;

    // Fast transpose
    if self.get_rows() == 1 || self.get_columns() == 1 {
      // Be very careful here, we need to clone the arc or we would delete the memory associated with this
      let result = self.clone();

      unsafe {
        reshape_matrix(result.get_id(), result.get_columns(), result.get_rows());
      }

      return result;
    }

    unsafe { result_id = cuda_transpose(self.get_id(), self.get_rows(), self.get_columns()) }

    return Matrix::new(result_id);
  }

  pub fn max_pool(&self) -> (Self, Self) {
    let result_ids: Tuple;

    unsafe { result_ids = cuda_max_pool(self.get_id(), self.get_rows(), self.get_columns()) }

    let pooled_matrix = Matrix::new(result_ids.a);
    let bitmask_matrix = Matrix::new(result_ids.b);

    return (pooled_matrix, bitmask_matrix);
  }

  pub fn nearest_neighbor_2x_upsample(&self, odd_upsample: bool) -> Self {
    let result_id: usize;

    unsafe {
      result_id = cuda_nearest_neighbor_2x_upsample(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        odd_upsample,
      )
    }

    return Matrix::new(result_id);
  }

  pub fn rotate_180(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_rotate_180(self.get_id(), self.get_rows(), self.get_columns()) }

    return Matrix::new(result_id);
  }

  pub fn correlate(&self, kernel: &Matrix, padding_type: PaddingType) -> Self {
    let result_id: usize;

    if matches!(padding_type, PaddingType::SAME)
      && (kernel.get_rows() != kernel.get_columns() || kernel.get_rows() % 2 == 0)
    {
      panic!("Kernel must be square and odd for same convolution!");
    }

    unsafe {
      result_id = cuda_correlate(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        kernel.get_id(),
        kernel.get_rows(),
        kernel.get_columns(),
        padding_type,
      )
    }

    return Matrix::new(result_id);
  }

  pub fn convolve(&self, kernel: &Matrix, padding_type: PaddingType) -> Self {
    let result_id: usize;

    if matches!(padding_type, PaddingType::SAME)
      && (kernel.get_rows() != kernel.get_columns() || kernel.get_rows() % 2 == 0)
    {
      panic!("Kernel must be square and odd for same convolution!");
    }

    unsafe {
      result_id = cuda_convolve(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        kernel.get_id(),
        kernel.get_rows(),
        kernel.get_columns(),
        padding_type,
      )
    }

    return Matrix::new(result_id);
  }

  pub fn center_pad(&self, pad_rows: usize, pad_cols: usize) -> Self {
    let result_id: usize;

    unsafe {
      result_id = cuda_center_pad(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        pad_rows,
        pad_cols,
      )
    }

    return Matrix::new(result_id);
  }

  pub fn softmax(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_softmax(self.get_id(), self.get_rows(), self.get_columns()) }

    return Matrix::new(result_id);
  }

  pub fn crop(&self, row_offset: usize, column_offset: usize, rows: usize, columns: usize) -> Self {
    let result_id: usize;

    unsafe {
      result_id = cuda_crop(
        self.get_id(),
        self.get_rows(),
        self.get_columns(),
        row_offset,
        column_offset,
        rows,
        columns,
      )
    }

    return Matrix::new(result_id);
  }

  pub fn deep_copy(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_copy(self.get_id(), self.get_rows(), self.get_columns()) }

    return Matrix::new(result_id);
  }

  // pub fn cuda_sum_all_matrix_elements(mat_id: usize, mat_rows: usize, mat_cols: usize) -> usize;
  pub fn sum_all_matrix_elements(&self) -> Self {
    let result_id: usize;

    unsafe {
      result_id = cuda_sum_all_matrix_elements(self.get_id(), self.get_rows(), self.get_columns())
    }

    return Matrix::new(result_id);
  }

  pub fn reshape(&mut self, new_rows: usize, new_columns: usize) -> &Self {
    unsafe {
      reshape_matrix(self.get_id(), new_rows, new_columns);
    }

    return self;
  }

  pub fn to_one_dimensional(&mut self) -> &Self {
    unsafe {
      reshape_matrix(self.get_id(), self.get_data_length(), 1);
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
    register_matrix_group(
      rows,
      columns,
      count,
      matrices.as_mut_ptr() as *mut c_ulonglong,
    );
  }

  return matrices;
}

// All matrices are required to be the same shape
pub fn flatten_matrix_array(to_flatten: &[Matrix]) -> Matrix {
  if to_flatten.len() == 0 {
    return Matrix::zeros(0, 0);
  }

  let mat_rows = to_flatten[0].get_rows();
  let mat_cols = to_flatten[0].get_columns();

  let out_id;
  unsafe {
    out_id = cuda_flatten_array(
      to_flatten.as_ptr() as *const c_ulonglong,
      to_flatten.len(),
      mat_rows,
      mat_cols,
    )
  };

  return Matrix::new(out_id);
}

// Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
pub fn img2col(image: &[Matrix], filter_rows: usize, filter_cols: usize) -> Matrix {
  let image_depth = image.len();

  let image_rows = image[0].get_rows();
  let image_cols = image[0].get_columns();

  let out_id;
  unsafe {
    out_id = cuda_img2col(
      image.as_ptr() as *const c_ulonglong,
      image_depth,
      image_rows,
      image_cols,
      filter_rows,
      filter_cols,
      PaddingType::VALID,
    )
  };

  return Matrix::new(out_id);
}

// All matrices are required to be the same shape
pub fn unflatten_array_to_matrices(
  to_unflatten: &Matrix,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = to_unflatten.get_data_length() / (mat_rows * mat_cols);
  // let mut results = vec![Matrix::new(0); num_matrices];
  let mut results = (0..num_matrices).map(|_| Matrix::new(0)).collect_vec();

  if to_unflatten.get_data_length() != num_matrices * mat_rows * mat_cols {
    panic!("Cannot unflatten array, matrix dimensions incorrect!")
  }

  unsafe {
    cuda_unflatten_array(
      to_unflatten.get_id(),
      to_unflatten.get_data_length(),
      mat_rows,
      mat_cols,
      results.as_mut_ptr() as *mut c_ulonglong,
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
  // let mut results = vec![Matrix::new(0); num_matrices];
  let mut results = (0..num_matrices).map(|_| Matrix::new(0)).collect_vec();

  if to_unflatten.get_data_length() != num_matrices * mat_rows * mat_cols {
    panic!("Cannot unflatten array, matrix dimensions incorrect!")
  }

  unsafe {
    cuda_unflatten_array_strided(
      to_unflatten.get_id(),
      to_unflatten.get_data_length(),
      mat_rows,
      mat_cols,
      results.as_mut_ptr() as *mut c_ulonglong,
    )
  };

  return results;
}

pub fn element_add_packed(mat_1s: &[Matrix], mat_2s: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = mat_1s.len();

  if num_matrices != mat_2s.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1s.len(),
      mat_2s.len()
    );
  }

  let mat_rows = mat_1s[0].get_columns();
  let mat_cols = mat_1s[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_element_add_packed(
      mat_1s.as_ptr() as *const c_ulonglong,
      mat_2s.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_add_packed_inplace(mat_1s: &[Matrix], mat_2s: &[Matrix]) {
  let num_matrices = mat_1s.len();

  if num_matrices != mat_2s.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1s.len(),
      mat_2s.len()
    );
  }

  let mat_rows = mat_1s[0].get_columns();
  let mat_cols = mat_1s[0].get_columns();

  unsafe {
    cuda_element_add_packed_inplace(
      mat_1s.as_ptr() as *const c_ulonglong,
      mat_2s.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn element_subtract_packed(mat_1s: &[Matrix], mat_2s: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = mat_1s.len();

  if num_matrices != mat_2s.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1s.len(),
      mat_2s.len()
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = mat_1s[0].get_columns();
  let mat_cols = mat_1s[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_element_subtract_packed(
      mat_1s.as_ptr() as *const c_ulonglong,
      mat_2s.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_subtract_packed_inplace(mat_1s: &[Matrix], mat_2s: &[Matrix]) {
  let num_matrices = mat_1s.len();

  if num_matrices != mat_2s.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1s.len(),
      mat_2s.len()
    );
  }

  let mat_rows = mat_1s[0].get_columns();
  let mat_cols = mat_1s[0].get_columns();

  unsafe {
    cuda_element_subtract_packed_inplace(
      mat_1s.as_ptr() as *const c_ulonglong,
      mat_2s.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn element_multiply_packed(mat_1s: &[Matrix], mat_2s: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = mat_1s.len();

  if num_matrices != mat_2s.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1s.len(),
      mat_2s.len()
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = mat_1s[0].get_columns();
  let mat_cols = mat_1s[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_element_multiply_packed(
      mat_1s.as_ptr() as *const c_ulonglong,
      mat_2s.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_multiply_packed_inplace(mat_1s: &[Matrix], mat_2s: &[Matrix]) {
  let num_matrices = mat_1s.len();

  if num_matrices != mat_2s.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1s.len(),
      mat_2s.len()
    );
  }

  let mat_rows = mat_1s[0].get_columns();
  let mat_cols = mat_1s[0].get_columns();

  unsafe {
    cuda_element_multiply_packed_inplace(
      mat_1s.as_ptr() as *const c_ulonglong,
      mat_2s.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn element_divide_packed(mat_1s: &[Matrix], mat_2s: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = mat_1s.len();

  if num_matrices != mat_2s.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1s.len(),
      mat_2s.len()
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = mat_1s[0].get_columns();
  let mat_cols = mat_1s[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_element_divide_packed(
      mat_1s.as_ptr() as *const c_ulonglong,
      mat_2s.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_divide_packed_inplace(mat_1s: &[Matrix], mat_2s: &[Matrix]) {
  let num_matrices = mat_1s.len();

  if num_matrices != mat_2s.len() {
    panic!(
      "Number of matrices must be equal! {} {}",
      mat_1s.len(),
      mat_2s.len()
    );
  }

  let mat_rows = mat_1s[0].get_columns();
  let mat_cols = mat_1s[0].get_columns();

  unsafe {
    cuda_element_divide_packed_inplace(
      mat_1s.as_ptr() as *const c_ulonglong,
      mat_2s.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn scalar_multiply_packed(matrices: &[Matrix], scalar: f32) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_scalar_multiply_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }

  return results;
}

pub fn scalar_multiply_packed_inplace(matrices: &[Matrix], scalar: f32) {
  let num_matrices = matrices.len();

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  unsafe {
    cuda_scalar_multiply_packed_inplace(
      matrices.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }
}

pub fn scalar_divide_packed(matrices: &[Matrix], scalar: f32) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_scalar_divide_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }

  return results;
}

pub fn scalar_divide_packed_inplace(matrices: &[Matrix], scalar: f32) {
  let num_matrices = matrices.len();

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  unsafe {
    cuda_scalar_divide_packed_inplace(
      matrices.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }
}

pub fn scalar_add_packed(matrices: &[Matrix], scalar: f32) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_scalar_add_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }

  return results;
}

pub fn scalar_add_packed_inplace(matrices: &[Matrix], scalar: f32) {
  let num_matrices = matrices.len();

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  unsafe {
    cuda_scalar_add_packed_inplace(
      matrices.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }
}

pub fn scalar_subtract_packed(matrices: &[Matrix], scalar: f32) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_scalar_subtract_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }

  return results;
}

pub fn scalar_subtract_packed_inplace(matrices: &[Matrix], scalar: f32) {
  let num_matrices = matrices.len();

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  unsafe {
    cuda_scalar_subtract_packed_inplace(
      matrices.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
      scalar,
    );
  }
}

pub fn element_sqrt_packed(matrices: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_element_sqrt_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_sqrt_packed_inplace(matrices: &[Matrix]) {
  let num_matrices = matrices.len();

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  unsafe {
    cuda_element_sqrt_packed_inplace(
      matrices.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn element_exp_packed(matrices: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_element_exp_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn element_exp_packed_inplace(matrices: &[Matrix]) {
  let num_matrices = matrices.len();

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  unsafe {
    cuda_element_exp_packed_inplace(
      matrices.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

#[allow(non_snake_case)]
pub fn element_ReLU_packed(matrices: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_element_ReLU_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

#[allow(non_snake_case)]
pub fn element_ReLU_packed_inplace(matrices: &[Matrix]) {
  let num_matrices = matrices.len();

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  unsafe {
    cuda_element_ReLU_packed_inplace(
      matrices.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

#[allow(non_snake_case)]
pub fn element_ReLU_prime_packed(matrices: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_element_ReLU_prime_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

#[allow(non_snake_case)]
pub fn element_ReLU_prime_packed_inplace(matrices: &[Matrix]) {
  let num_matrices = matrices.len();

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  unsafe {
    cuda_element_ReLU_prime_packed_inplace(
      matrices.as_ptr() as *const c_ulonglong,
      null_mut(),
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }
}

pub fn max_pool_packed(matrices: &[Matrix]) -> (Vec<Matrix>, Vec<Matrix>) {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return (Vec::new(), Vec::new());
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut result_ids = vec![Tuple { a: 0, b: 0 }; num_matrices];

  unsafe {
    cuda_max_pool_packed(
      matrices.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut Tuple,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  let pooled_result = result_ids
    .iter()
    .map(|result_id| Matrix::new(result_id.a))
    .collect_vec();

  let bitmask_result = result_ids
    .iter()
    .map(|result_id| Matrix::new(result_id.b))
    .collect_vec();

  return (pooled_result, bitmask_result);
}

pub fn nearest_neighbor_2x_upsample_packed(matrices: &[Matrix], odd_upsample: bool) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  // let mut results = vec![Matrix::new(0); num_matrices];
  let mut results = (0..num_matrices).map(|_| Matrix::new(0)).collect_vec();
  unsafe {
    cuda_nearest_neighbor_2x_upsample_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      odd_upsample,
    );
  }

  return results;
}

pub fn rotate_180_packed(matrices: &[Matrix]) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_rotate_180_packed(
      matrices.as_ptr() as *const c_ulonglong,
      results.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  return results;
}

pub fn correlate_packed(
  matrices: &[Matrix],
  kernels: &[Matrix],
  padding_type: PaddingType,
) -> Vec<Matrix> {
  let start = Instant::now();
  let num_matrices = matrices.len();
  let num_kernels = kernels.len();

  if num_matrices != num_kernels {
    panic!(
      "Number of matrices must be equal to number of kernels! {} {}",
      num_matrices, num_kernels
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();
  let kernel_rows = kernels[0].get_rows();
  let kernel_cols = kernels[0].get_columns();

  if matches!(padding_type, PaddingType::SAME)
    && (kernel_rows != kernel_cols || kernel_rows % 2 == 0)
  {
    panic!("Kernel must be square and odd for same correlation!");
  }

  let setup = start.elapsed();

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_correlate_packed(
      matrices.as_ptr() as *const c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      kernels.as_ptr() as *const c_ulonglong,
      kernel_rows,
      kernel_cols,
      results.as_mut_ptr() as *mut c_ulonglong,
      padding_type,
    );
  }
  let call = start.elapsed() - setup;

  let padding_decision_time = start.elapsed() - setup - call;

  let result_id_time = start.elapsed() - setup - call - padding_decision_time;
  let total = start.elapsed();
  // println!("Correlate setup: {:?}", setup);
  // println!("Correlate call: {:?}", call);
  // println!(
  //   "Correlate padding decision time: {:?}",
  //   padding_decision_time
  // );
  // println!("Correlate result id time: {:?}", result_id_time);
  // println!("Correlate total: {:?}", total);
  return results;
}

pub fn convolve_packed(
  matrices: &[Matrix],
  kernels: &[Matrix],
  padding_type: PaddingType,
) -> Vec<Matrix> {
  let num_matrices = matrices.len();
  let num_kernels = kernels.len();

  if num_matrices != num_kernels {
    panic!(
      "Number of matrices must be equal to number of kernels! {} {}",
      num_matrices, num_kernels
    );
  }

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].get_rows();
  let mat_cols = matrices[0].get_columns();
  let kernel_rows = kernels[0].get_rows();
  let kernel_cols = kernels[0].get_columns();

  if matches!(padding_type, PaddingType::SAME)
    && (kernel_rows != kernel_cols || kernel_rows % 2 == 0)
  {
    panic!("Kernel must be square and odd for same correlation!");
  }

  let mut results = Vec::with_capacity(num_matrices);

  unsafe {
    results.set_len(num_matrices);
    cuda_convolve_packed(
      matrices.as_ptr() as *const c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      kernels.as_ptr() as *const c_ulonglong,
      kernel_rows,
      kernel_cols,
      results.as_mut_ptr() as *mut c_ulonglong,
      padding_type,
    );
  }

  return results;
}
