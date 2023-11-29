use crate::cuda_bindings::*;
use itertools::Itertools;
use rand::prelude::Distribution;
use statrs::distribution::Normal;
use std::ffi::{c_float, c_ulonglong};
use std::io::{stdout, BufWriter, Write};
use std::sync::Arc;

// IMPORTANT NOTE: THIS API IS THREADSAFE, BUT THE UNDERLYING CUDA LIBRARY IS NOT THREAD SAFE (yet).

#[derive(Copy, Clone)]
#[repr(C)]
pub enum ConvolutionType {
  VALID,
  SAME,
  FULL,
}

// Use newtype for id to ensure drop arc<id> does not apply to all usize
struct MatrixId(usize);

impl MatrixId {
  fn new(id: usize) -> Self {
    return MatrixId(id);
  }
}

impl Drop for MatrixId {
  fn drop(&mut self) {
    unsafe {
      unregister_matrix(self.0);
    }
  }
}

// Note cloning keeps the same id, used for ownership movement in rust.
// Does not actually copy the underlying matrix in cuda.
// Cloning is explicit to ensure this understanding
#[derive(Clone)]
pub struct Matrix {
  id: Arc<MatrixId>,
  pub rows: usize,
  pub columns: usize,
}

impl Matrix {
  pub fn get_data_length(&self) -> usize {
    return self.rows * self.columns;
  }

  fn new(id: usize, rows: usize, columns: usize) -> Self {
    return Matrix {
      id: Arc::new(MatrixId::new(id)),
      rows,
      columns,
    };
  }

  pub fn get_id(&self) -> usize {
    return (*self.id).0;
  }

  pub fn get_data(&self) -> Vec<Vec<f32>> {
    let mut data = Vec::<c_float>::with_capacity(self.get_data_length());
    unsafe {
      get_matrix_data(self.get_id(), self.rows, self.columns, data.as_mut_ptr());
      data.set_len(self.get_data_length());
    }

    let mut index = 0;
    let mut result = Vec::new();
    for _ in 0..self.rows {
      let row_slice = data[index..index + self.columns as usize].to_vec();
      result.push(row_slice);
      index += self.columns as usize;
    }

    return result;
  }

  pub fn no_fill(rows: usize, columns: usize) -> Self {
    let id;
    unsafe {
      id = register_matrix(rows, columns);
    }
    return Matrix::new(id, rows, columns);
  }

  pub fn zeros(rows: usize, columns: usize) -> Self {
    let data = vec![0.0; rows * columns];
    let id;
    unsafe {
      id = register_matrix_with_data(data.as_ptr(), rows, columns);
    }
    return Matrix::new(id, rows, columns);
  }

  pub fn new_1d(data: &Vec<f32>, rows: usize, columns: usize) -> Self {
    if data.len() != rows * columns {
      panic!("Rows and Columns specified not compatible with new_1d size!");
    }

    let id;
    unsafe {
      id = register_matrix_with_data(data.as_ptr(), rows, columns);
    }

    return Matrix::new(id, rows, columns);
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

    return Matrix::new(id, rows, columns);
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

    for row in 0..self.rows {
      for index in 0..self.columns {
        let val = data[row][index];
        let formatted = format!(" {:<5} ", val);
        handle.write(formatted.as_bytes()).unwrap();
      }
      handle.write(b"\n").unwrap();
    }

    handle.write(b"\n").unwrap();
  }

  pub fn print_shape(&self) {
    println!("Shape of matrix is: {} {}", self.rows, self.columns);
  }

  pub fn same_shape(&self, other: &Matrix) -> bool {
    return self.rows == other.rows && self.columns == other.columns;
  }

  fn element_add_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for addition! A: {} {} B: {} {}",
        self.rows, self.columns, other.rows, other.columns
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_element_add(
        self.get_id(),
        self.rows,
        self.columns,
        other.get_id(),
        other.rows,
        other.columns,
        inplace,
      )
    }

    return result_id;
  }

  pub fn element_add(&self, other: &Matrix) -> Self {
    let result_id = self.element_add_impl(other, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn element_add_inplace(&self, other: &Matrix) -> Self {
    self.element_add_impl(other, true);
    return self.clone();
  }

  fn element_subtract_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_subtract! A: {} {} B: {} {}",
        self.rows, self.columns, other.rows, other.columns
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_element_subtract(
        self.get_id(),
        self.rows,
        self.columns,
        other.get_id(),
        other.rows,
        other.columns,
        inplace,
      )
    }

    return result_id;
  }

  pub fn element_subtract(&self, other: &Matrix) -> Self {
    let result_id = self.element_subtract_impl(other, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn element_subtract_inplace(&self, other: &Matrix) -> Self {
    self.element_subtract_impl(other, true);
    return self.clone();
  }

  fn element_multiply_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_multiply! A: {} {} B: {} {}",
        self.rows, self.columns, other.rows, other.columns
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_element_multiply(
        self.get_id(),
        self.rows,
        self.columns,
        other.get_id(),
        other.rows,
        other.columns,
        inplace,
      )
    }

    return result_id;
  }

  pub fn element_multiply(&self, other: &Matrix) -> Self {
    let result_id = self.element_multiply_impl(other, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn element_multiply_inplace(&self, other: &Matrix) -> Self {
    self.element_multiply_impl(other, true);
    return self.clone();
  }

  fn element_divide_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !self.same_shape(other) {
      panic!(
        "Matrices not the same shape for element_divide! A: {} {} B: {} {}",
        self.rows, self.columns, other.rows, other.columns
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_element_divide(
        self.get_id(),
        self.rows,
        self.columns,
        other.get_id(),
        other.rows,
        other.columns,
        inplace,
      )
    }

    return result_id;
  }

  pub fn element_divide(&self, other: &Matrix) -> Self {
    let result_id = self.element_divide_impl(other, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn element_divide_inplace(&self, other: &Matrix) -> Self {
    self.element_divide_impl(other, true);
    return self.clone();
  }

  fn scalar_multiply_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_scalar_multiply(self.get_id(), self.rows, self.columns, scalar, inplace)
    }

    return result_id;
  }

  pub fn scalar_multiply(&self, scalar: f32) -> Self {
    let result_id = self.scalar_multiply_impl(scalar, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn scalar_multiply_inplace(&self, scalar: f32) -> Self {
    self.scalar_multiply_impl(scalar, true);
    return self.clone();
  }

  fn scalar_divide_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_scalar_divide(self.get_id(), self.rows, self.columns, scalar, inplace)
    }

    return result_id;
  }

  pub fn scalar_divide(&self, scalar: f32) -> Self {
    let result_id = self.scalar_divide_impl(scalar, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn scalar_divide_inplace(&self, scalar: f32) -> Self {
    self.scalar_divide_impl(scalar, true);
    return self.clone();
  }

  fn scalar_add_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_scalar_add(self.get_id(), self.rows, self.columns, scalar, inplace) }

    return result_id;
  }

  pub fn scalar_add(&self, scalar: f32) -> Self {
    let result_id = self.scalar_add_impl(scalar, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn scalar_add_inplace(&self, scalar: f32) -> Self {
    self.scalar_add_impl(scalar, true);
    return self.clone();
  }

  fn scalar_subtract_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe {
      result_id = cuda_scalar_subtract(self.get_id(), self.rows, self.columns, scalar, inplace)
    }

    return result_id;
  }

  pub fn scalar_subtract(&self, scalar: f32) -> Self {
    let result_id = self.scalar_subtract_impl(scalar, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn scalar_subtract_inplace(&self, scalar: f32) -> Self {
    self.scalar_subtract_impl(scalar, true);
    return self.clone();
  }

  pub fn matrix_multiply(&self, other: &Matrix) -> Self {
    // Bound Check
    if self.columns != other.rows {
      panic!(
        "Matrices not compatible shape for mat mult! A: {} {} B: {} {}",
        self.rows, self.columns, other.rows, other.columns
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_matrix_multiply(
        self.get_id(),
        self.rows,
        self.columns,
        other.get_id(),
        other.rows,
        other.columns,
      )
    }

    let output_rows = self.rows;
    let output_columns = other.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  fn add_vector_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !((self.rows == other.rows && other.columns == 1)
      || (self.columns == other.columns && other.rows == 1))
    {
      panic!(
        "Matrices not the correct shape for vector add! A: {} {} B: {} {}",
        self.rows, self.columns, other.rows, other.columns
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_add_vector(
        self.get_id(),
        self.rows,
        self.columns,
        other.get_id(),
        other.rows,
        other.columns,
        inplace,
      )
    }

    return result_id;
  }

  pub fn add_vector(&self, other: &Matrix) -> Self {
    let result_id = self.add_vector_impl(other, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn add_vector_inplace(&self, other: &Matrix) -> Self {
    self.add_vector_impl(other, true);

    return self.clone();
  }

  fn divide_by_vector_impl(&self, other: &Matrix, inplace: bool) -> usize {
    if !((self.rows == other.rows && other.columns == 1)
      || (self.columns == other.columns && other.rows == 1))
    {
      panic!(
        "Matrices not the correct shape for division by vector! A: {} {} B: {} {}",
        self.rows, self.columns, other.rows, other.columns
      );
    }

    let result_id: usize;
    unsafe {
      result_id = cuda_divide_by_vector(
        self.get_id(),
        self.rows,
        self.columns,
        other.get_id(),
        other.rows,
        other.columns,
        inplace,
      )
    }

    return result_id;
  }

  pub fn divide_by_vector(&self, other: &Matrix) -> Self {
    let result_id = self.divide_by_vector_impl(other, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn divide_by_vector_inplace(&self, other: &Matrix) -> Self {
    self.divide_by_vector_impl(other, true);

    return self.clone();
  }

  fn element_sqrt_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_element_sqrt(self.get_id(), self.rows, self.columns, inplace) }
    return result_id;
  }

  pub fn element_sqrt(&self) -> Self {
    let result_id = self.element_sqrt_impl(false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn element_sqrt_inplace(&self) -> Self {
    self.element_sqrt_impl(true);

    return self.clone();
  }

  fn element_exp_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_element_exp(self.get_id(), self.rows, self.columns, inplace) }
    return result_id;
  }

  pub fn element_exp(&self) -> Self {
    let result_id = self.element_exp_impl(false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn element_exp_inplace(&self) -> Self {
    self.element_exp_impl(true);

    return self.clone();
  }

  #[allow(non_snake_case)]
  fn element_ReLU_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_element_ReLU(self.get_id(), self.rows, self.columns, inplace) }

    return result_id;
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU(&self) -> Self {
    let result_id = self.element_ReLU_impl(false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_inplace(&self) -> Self {
    self.element_ReLU_impl(true);

    return self.clone();
  }

  #[allow(non_snake_case)]
  fn element_ReLU_prime_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_element_ReLU_prime(self.get_id(), self.rows, self.columns, inplace) }

    return result_id;
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime(&self) -> Self {
    let result_id = self.element_ReLU_prime_impl(false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime_inplace(&self) -> Self {
    self.element_ReLU_prime_impl(true);

    return self.clone();
  }

  pub fn sum_rows_matrix(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_sum_rows(self.get_id(), self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = 1;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn sum_columns_matrix(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_sum_columns(self.get_id(), self.rows, self.columns) }

    let output_rows = 1;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn sum_columns(&self) -> Vec<f32> {
    let result_id: usize;
    unsafe { result_id = cuda_sum_columns(self.get_id(), self.rows, self.columns) }

    let output_rows = 1;
    let output_columns = self.columns;

    let result_matrix = Matrix::new(result_id, output_rows, output_columns);

    let result = result_matrix.get_data()[0].to_vec();
    return result;
  }

  pub fn transpose(&self) -> Self {
    let result_id: usize;

    // Fast transpose
    if self.rows == 1 || self.columns == 1 {
      // Be very careful here, we need to clone the arc or we would delete the memory associated with this
      let mut result = self.clone();
      result.rows = self.columns;
      result.columns = self.rows;
      return result;
    }

    unsafe { result_id = cuda_transpose(self.get_id(), self.rows, self.columns) }

    let output_rows = self.columns;
    let output_columns = self.rows;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn max_pool(&self) -> (Self, Self) {
    let result_ids: Tuple;

    unsafe { result_ids = cuda_max_pool(self.get_id(), self.rows, self.columns) }

    let output_rows = self.rows / 2 + self.rows % 2;
    let output_columns = self.columns / 2 + self.columns % 2;

    let pooled_matrix = Matrix::new(result_ids.a, output_rows, output_columns);
    let bitmask_matrix = Matrix::new(result_ids.b, self.rows, self.columns);

    return (pooled_matrix, bitmask_matrix);
  }

  pub fn nearest_neighbor_2x_upsample(&self, odd_upsample: bool) -> Self {
    let result_id: usize;

    unsafe {
      result_id =
        cuda_nearest_neighbor_2x_upsample(self.get_id(), self.rows, self.columns, odd_upsample)
    }

    let output_rows = self.rows * 2 - (odd_upsample as usize);
    let output_columns = self.columns * 2 - (odd_upsample as usize);

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn rotate_180(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_rotate_180(self.get_id(), self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn convolution(&self, kernel: &Matrix, conv_type: ConvolutionType) -> Self {
    let result_id: usize;

    if matches!(conv_type, ConvolutionType::SAME)
      && (kernel.rows != kernel.columns || kernel.rows % 2 == 0)
    {
      panic!("Kernel must be square and odd for same convolution!");
    }

    unsafe {
      result_id = cuda_convolution(
        self.get_id(),
        self.rows,
        self.columns,
        kernel.get_id(),
        kernel.rows,
        kernel.columns,
        conv_type,
      )
    }

    let output_rows = match conv_type {
      ConvolutionType::VALID => self.rows - kernel.rows + 1,
      ConvolutionType::SAME => self.rows,
      ConvolutionType::FULL => self.rows + kernel.rows - 1,
    };

    let output_columns = match conv_type {
      ConvolutionType::VALID => self.columns - kernel.columns + 1,
      ConvolutionType::SAME => self.columns,
      ConvolutionType::FULL => self.columns + kernel.columns - 1,
    };

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn convolution_v2(&self, kernel: &Matrix, conv_type: ConvolutionType) -> Self {
    if matches!(conv_type, ConvolutionType::SAME)
      && (kernel.rows != kernel.columns || kernel.rows % 2 == 0)
    {
      panic!("Kernel must be square and odd for same convolution!");
    }

    // Fast version using matrix multiplication

    // First we need to flatten the kernel
    let flattened_kernel = flatten_matrix_array(&vec![kernel.clone()]);

    // Now transform the input matrix into a matrix of columns
    let transformed_img = img2col(&vec![self.clone()], kernel.rows, kernel.columns);

    // Now perform a matrix multiplication
    let mut result = flattened_kernel.matrix_multiply(&transformed_img);

    let output_rows = match conv_type {
      ConvolutionType::VALID => self.rows - kernel.rows + 1,
      ConvolutionType::SAME => self.rows,
      ConvolutionType::FULL => self.rows + kernel.rows - 1,
    };

    let output_columns = match conv_type {
      ConvolutionType::VALID => self.columns - kernel.columns + 1,
      ConvolutionType::SAME => self.columns,
      ConvolutionType::FULL => self.columns + kernel.columns - 1,
    };

    // Unflatten the result
    result.reshape(output_rows, output_columns);

    return result;
  }

  pub fn center_pad(&self, pad_rows: usize, pad_cols: usize) -> Self {
    let result_id: usize;

    unsafe {
      result_id = cuda_center_pad(self.get_id(), self.rows, self.columns, pad_rows, pad_cols)
    }

    let output_rows = self.rows + 2 * pad_rows;
    let output_columns = self.columns + 2 * pad_cols;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn softmax(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_softmax(self.get_id(), self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn crop(&self, row_offset: usize, column_offset: usize, rows: usize, columns: usize) -> Self {
    let result_id: usize;

    unsafe {
      result_id = cuda_crop(
        self.get_id(),
        self.rows,
        self.columns,
        row_offset,
        column_offset,
        rows,
        columns,
      )
    }

    let output_rows = rows;
    let output_columns = columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn deep_copy(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_copy(self.get_id(), self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  // pub fn cuda_sum_all_matrix_elements(mat_id: usize, mat_rows: usize, mat_cols: usize) -> usize;
  pub fn sum_all_matrix_elements(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_sum_all_matrix_elements(self.get_id(), self.rows, self.columns) }

    let output_rows = 1;
    let output_columns = 1;

    return Matrix::new(result_id, output_rows, output_columns);
  }

  pub fn reshape(&mut self, new_rows: usize, new_columns: usize) -> &Self {
    if new_rows * new_columns == self.get_data_length() {
      self.rows = new_rows;
      self.columns = new_columns;
    } else {
      panic!("Cannot reshape, matrices are not the same length!")
    }

    return self;
  }

  pub fn to_one_dimensional(&mut self) -> &Self {
    self.columns = self.get_data_length();
    self.rows = 1;

    return self;
  }
}

// All matrices are required to be the same shape
pub fn flatten_matrix_array(to_flatten: &Vec<Matrix>) -> Matrix {
  if to_flatten.len() == 0 {
    return Matrix::zeros(0, 0);
  }

  let mat_ids = to_flatten.iter().map(|mat| mat.get_id()).collect_vec();

  let mat_rows = to_flatten[0].rows;
  let mat_cols = to_flatten[0].columns;

  let out_id;
  unsafe {
    out_id = cuda_flatten_array(
      mat_ids.as_ptr() as *const c_ulonglong,
      to_flatten.len(),
      mat_rows,
      mat_cols,
    )
  };

  let out_rows = 1;
  let out_cols = to_flatten[0].get_data_length() * to_flatten.len();
  return Matrix::new(out_id, out_rows, out_cols);
}

// Take an image and convert it to a matrix of columns based on patches (with specified padding) the filter makes of image
pub fn img2col(image: &Vec<Matrix>, filter_rows: usize, filter_cols: usize) -> Matrix {
  let image_depth = image.len();

  let image_rows = image[0].rows;
  let image_cols = image[0].columns;

  let image_ids = image.iter().map(|image| image.get_id()).collect_vec();

  let out_id;
  unsafe {
    out_id = cuda_img2col(
      image_ids.as_ptr() as *const c_ulonglong,
      image_depth,
      image_rows,
      image_cols,
      filter_rows,
      filter_cols,
      ConvolutionType::VALID,
    )
  };

  let out_rows = image_depth * filter_rows * filter_cols;
  let out_cols = (image_rows - filter_rows + 1) * (image_cols - filter_cols + 1);
  return Matrix::new(out_id, out_rows, out_cols);
}

// All matrices are required to be the same shape
pub fn unflatten_array_to_matrices(
  to_unflatten: &Matrix,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = to_unflatten.get_data_length() / (mat_rows * mat_cols);
  let mut mat_ids = vec![0; num_matrices];

  if to_unflatten.get_data_length() != num_matrices * mat_rows * mat_cols {
    panic!("Cannot unflatten array, matrix dimensions incorrect!")
  }

  unsafe {
    cuda_unflatten_array(
      to_unflatten.get_id(),
      to_unflatten.get_data_length(),
      mat_rows,
      mat_cols,
      mat_ids.as_mut_ptr() as *mut c_ulonglong,
    )
  };

  return mat_ids
    .iter()
    .map(|mat_id| Matrix::new(*mat_id, mat_rows, mat_cols))
    .collect_vec();
}

// All matrices are required to be the same shape. Each array's first n elements are the first elements in memory. [arr1_elem1, arr2_elem1, arr3_elem1, arr1_elem2, arr2_elem2, arr3_elem2, ...]
pub fn unflatten_array_strided_to_matrices(
  to_unflatten: &Matrix,
  mat_rows: usize,
  mat_cols: usize,
) -> Vec<Matrix> {
  let num_matrices = to_unflatten.get_data_length() / (mat_rows * mat_cols);
  let mut mat_ids = vec![0; num_matrices];

  if to_unflatten.get_data_length() != num_matrices * mat_rows * mat_cols {
    panic!("Cannot unflatten array, matrix dimensions incorrect!")
  }

  unsafe {
    cuda_unflatten_array_strided(
      to_unflatten.get_id(),
      to_unflatten.get_data_length(),
      mat_rows,
      mat_cols,
      mat_ids.as_mut_ptr() as *mut c_ulonglong,
    )
  };

  return mat_ids
    .iter()
    .map(|mat_id| Matrix::new(*mat_id, mat_rows, mat_cols))
    .collect_vec();
}

pub fn element_add_packed(
  mat_1s: &Vec<Matrix>,
  mat_2s: &Vec<Matrix>,
  inplace: bool,
) -> Vec<Matrix> {
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

  let mat_rows = mat_1s[0].rows;
  let mat_cols = mat_1s[0].columns;

  let mat_1_ids = mat_1s.iter().map(|mat| mat.get_id()).collect_vec();
  let mat_2_ids = mat_2s.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_element_add_packed(
      mat_1_ids.as_ptr() as *const c_ulonglong,
      mat_2_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return mat_1s.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn element_subtract_packed(
  mat_1s: &Vec<Matrix>,
  mat_2s: &Vec<Matrix>,
  inplace: bool,
) -> Vec<Matrix> {
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

  let mat_rows = mat_1s[0].rows;
  let mat_cols = mat_1s[0].columns;

  let mat_1_ids = mat_1s.iter().map(|mat| mat.get_id()).collect_vec();
  let mat_2_ids = mat_2s.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_element_subtract_packed(
      mat_1_ids.as_ptr() as *const c_ulonglong,
      mat_2_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return mat_1s.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn element_multiply_packed(
  mat_1s: &Vec<Matrix>,
  mat_2s: &Vec<Matrix>,
  inplace: bool,
) -> Vec<Matrix> {
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

  let mat_rows = mat_1s[0].rows;
  let mat_cols = mat_1s[0].columns;

  let mat_1_ids = mat_1s.iter().map(|mat| mat.get_id()).collect_vec();
  let mat_2_ids = mat_2s.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_element_multiply_packed(
      mat_1_ids.as_ptr() as *const c_ulonglong,
      mat_2_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return mat_1s.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn element_divide_packed(
  mat_1s: &Vec<Matrix>,
  mat_2s: &Vec<Matrix>,
  inplace: bool,
) -> Vec<Matrix> {
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

  let mat_rows = mat_1s[0].rows;
  let mat_cols = mat_1s[0].columns;

  let mat_1_ids = mat_1s.iter().map(|mat| mat.get_id()).collect_vec();
  let mat_2_ids = mat_2s.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_element_divide_packed(
      mat_1_ids.as_ptr() as *const c_ulonglong,
      mat_2_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return mat_1s.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn scalar_multiply_packed(
  matrices: &Vec<Matrix>,
  scalars: &Vec<f32>,
  inplace: bool,
) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_scalar_multiply_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      scalars.as_ptr() as *const c_float,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return matrices.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn scalar_divide_packed(
  matrices: &Vec<Matrix>,
  scalars: &Vec<f32>,
  inplace: bool,
) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_scalar_divide_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      scalars.as_ptr() as *const c_float,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return matrices.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn scalar_add_packed(matrices: &Vec<Matrix>, scalars: &Vec<f32>, inplace: bool) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_scalar_add_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      scalars.as_ptr() as *const c_float,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return matrices.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn scalar_subtract_packed(
  matrices: &Vec<Matrix>,
  scalars: &Vec<f32>,
  inplace: bool,
) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_scalar_subtract_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      scalars.as_ptr() as *const c_float,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return matrices.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn element_sqrt_packed(matrices: &Vec<Matrix>, inplace: bool) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_element_sqrt_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return matrices.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn element_exp_packed(matrices: &Vec<Matrix>, inplace: bool) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_element_exp_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return matrices.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

#[allow(non_snake_case)]
pub fn element_ReLU_packed(matrices: &Vec<Matrix>, inplace: bool) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_element_ReLU_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return matrices.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

#[allow(non_snake_case)]
pub fn element_ReLU_prime_packed(matrices: &Vec<Matrix>, inplace: bool) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_element_ReLU_prime_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      inplace,
    );
  }

  if inplace {
    // Return mat1s, but clone to keep arc
    return matrices.to_owned();
  } else {
    return result_ids
      .iter()
      .map(|result_id| Matrix::new(*result_id, mat_rows, mat_cols))
      .collect_vec();
  }
}

pub fn max_pool_packed(matrices: &Vec<Matrix>) -> (Vec<Matrix>, Vec<Matrix>) {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return (Vec::new(), Vec::new());
  }

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![Tuple { a: 0, b: 0 }; num_matrices];

  unsafe {
    cuda_max_pool_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut Tuple,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  let output_rows = mat_rows / 2 + mat_rows % 2;
  let output_columns = mat_cols / 2 + mat_cols % 2;

  let pooled_result = result_ids
    .iter()
    .map(|result_id| Matrix::new(result_id.a, output_rows, output_columns))
    .collect_vec();

  let bitmask_result = result_ids
    .iter()
    .map(|result_id| Matrix::new(result_id.b, mat_rows, mat_cols))
    .collect_vec();

  return (pooled_result, bitmask_result);
}

pub fn nearest_neighbor_2x_upsample_packed(
  matrices: &Vec<Matrix>,
  odd_upsample: bool,
) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  unsafe {
    cuda_nearest_neighbor_2x_upsample_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      odd_upsample,
    );
  }

  let output_rows = mat_rows * 2 - (odd_upsample as usize);
  let output_columns = mat_cols * 2 - (odd_upsample as usize);

  return result_ids
    .iter()
    .map(|result_id| Matrix::new(*result_id, output_rows, output_columns))
    .collect_vec();
}

pub fn rotate_180_packed(matrices: &Vec<Matrix>) -> Vec<Matrix> {
  let num_matrices = matrices.len();

  if num_matrices == 0 {
    return Vec::new();
  }

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;

  unsafe {
    cuda_rotate_180_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
    );
  }

  let output_rows = mat_rows;
  let output_columns = mat_cols;

  return result_ids
    .iter()
    .map(|result_id| Matrix::new(*result_id, output_rows, output_columns))
    .collect_vec();
}

pub fn convolution_packed(
  matrices: &Vec<Matrix>,
  kernels: &Vec<Matrix>,
  conv_type: ConvolutionType,
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

  let mat_rows = matrices[0].rows;
  let mat_cols = matrices[0].columns;
  let kernel_rows = kernels[0].rows;
  let kernel_cols = kernels[0].columns;

  if matches!(conv_type, ConvolutionType::SAME)
    && (kernel_rows != kernel_cols || kernel_rows % 2 == 0)
  {
    panic!("Kernel must be square and odd for same convolution!");
  }

  let mat_ids = matrices.iter().map(|mat| mat.get_id()).collect_vec();
  let kernel_ids = kernels.iter().map(|kernel| kernel.get_id()).collect_vec();
  let mut result_ids = vec![0; num_matrices];

  unsafe {
    cuda_convolution_packed(
      mat_ids.as_ptr() as *const c_ulonglong,
      num_matrices,
      mat_rows,
      mat_cols,
      kernel_ids.as_ptr() as *const c_ulonglong,
      kernel_rows,
      kernel_cols,
      result_ids.as_mut_ptr() as *mut c_ulonglong,
      conv_type,
    );
  }

  let output_rows = match conv_type {
    ConvolutionType::VALID => mat_rows - kernel_rows + 1,
    ConvolutionType::SAME => mat_rows,
    ConvolutionType::FULL => mat_rows + kernel_rows - 1,
  };

  let output_columns = match conv_type {
    ConvolutionType::VALID => mat_cols - kernel_cols + 1,
    ConvolutionType::SAME => mat_cols,
    ConvolutionType::FULL => mat_cols + kernel_cols - 1,
  };

  return result_ids
    .iter()
    .map(|result_id| Matrix::new(*result_id, output_rows, output_columns))
    .collect_vec();
}
