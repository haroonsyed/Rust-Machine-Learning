pub mod bindings;
pub mod lib_cpu;

use bindings::*;
use itertools::Itertools;
use rand::prelude::Distribution;
use statrs::distribution::Normal;
use std::ffi::{c_float, c_ulonglong};
use std::io::{stdout, BufWriter, Write};

#[derive(Copy, Clone)]
#[repr(C)]
pub enum ConvolutionType {
  VALID,
  SAME,
  FULL,
}
pub struct Matrix {
  id: usize,
  pub rows: usize,
  pub columns: usize,
}

impl Drop for Matrix {
  fn drop(&mut self) {
    unsafe {
      unregister_matrix(self.id);
    }
  }
}

impl Matrix {
  pub fn get_data_length(&self) -> usize {
    return self.rows * self.columns;
  }

  pub fn get_data(&self) -> Vec<Vec<f32>> {
    let mut data = Vec::<c_float>::with_capacity(self.get_data_length());
    unsafe {
      get_matrix_data(self.id, self.rows, self.columns, data.as_mut_ptr());
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
    let data = Vec::<c_float>::with_capacity(rows * columns);
    let id;
    unsafe {
      id = register_matrix(data.as_ptr(), rows, columns);
    }
    return Matrix { id, rows, columns };
  }

  pub fn zeros(rows: usize, columns: usize) -> Self {
    let data = vec![0.0; rows * columns];
    let id;
    unsafe {
      id = register_matrix(data.as_ptr(), rows, columns);
    }
    return Matrix { id, rows, columns };
  }

  pub fn new_1d(data: &Vec<f32>, rows: usize, columns: usize) -> Self {
    if data.len() != rows * columns {
      panic!("Rows and Columns specified not compatible with new_1d size!");
    }

    let id;
    unsafe {
      id = register_matrix(data.as_ptr(), rows, columns);
    }

    return Matrix { id, rows, columns };
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
      id = register_matrix(flattened.as_ptr(), rows, columns);
    }

    return Matrix { id, rows, columns };
  }

  pub fn new_random(mean: f64, std: f64, width: usize, height: usize) -> Self {
    let mut rng = rand::thread_rng();
    let range = Normal::new(mean, std).unwrap();

    let data = (0..height)
      .map(|_| {
        (0..width)
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
      for index in (0..self.columns) {
        let mut val = data[row][index];
        val = (val * 1000.0).round() / 1000.0;
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
        self.id,
        self.rows,
        self.columns,
        other.id,
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

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn element_add_inplace(&self, other: &Matrix) -> &Self {
    self.element_add_impl(other, true);
    return self;
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
        self.id,
        self.rows,
        self.columns,
        other.id,
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

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn element_subtract_inplace(&self, other: &Matrix) -> &Self {
    self.element_subtract_impl(other, true);
    return self;
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
        self.id,
        self.rows,
        self.columns,
        other.id,
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

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn element_multiply_inplace(&self, other: &Matrix) -> &Self {
    self.element_multiply_impl(other, true);
    return self;
  }

  fn scalar_multiply_impl(&self, scalar: f32, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_scalar_multiply(self.id, self.rows, self.columns, scalar, inplace) }

    return result_id;
  }

  pub fn scalar_multiply(&self, scalar: f32) -> Self {
    let result_id = self.scalar_multiply_impl(scalar, false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn scalar_multiply_inplace(&self, scalar: f32) -> &Self {
    self.scalar_multiply_impl(scalar, true);
    return self;
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
        self.id,
        self.rows,
        self.columns,
        other.id,
        other.rows,
        other.columns,
      )
    }

    let output_rows = self.rows;
    let output_columns = other.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
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
        self.id,
        self.rows,
        self.columns,
        other.id,
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

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn add_vector_inplace(&self, other: &Matrix) -> &Self {
    self.add_vector_impl(other, true);

    return self;
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
        self.id,
        self.rows,
        self.columns,
        other.id,
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

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn divide_by_vector_inplace(&self, other: &Matrix) -> &Self {
    self.divide_by_vector_impl(other, true);

    return self;
  }

  fn element_exp_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_element_exp(self.id, self.rows, self.columns, inplace) }
    return result_id;
  }

  pub fn element_exp(&self) -> Self {
    let result_id = self.element_exp_impl(false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn element_exp_inplace(&self) -> &Self {
    self.element_exp_impl(true);

    return self;
  }

  #[allow(non_snake_case)]
  fn element_ReLU_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_element_ReLU(self.id, self.rows, self.columns, inplace) }

    return result_id;
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU(&self) -> Self {
    let result_id = self.element_ReLU_impl(false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_inplace(&self) -> &Self {
    self.element_ReLU_impl(true);

    return self;
  }

  #[allow(non_snake_case)]
  fn element_ReLU_prime_impl(&self, inplace: bool) -> usize {
    let result_id: usize;
    unsafe { result_id = cuda_element_ReLU_prime(self.id, self.rows, self.columns, inplace) }

    return result_id;
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime(&self) -> Self {
    let result_id = self.element_ReLU_prime_impl(false);
    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime_inplace(&self) {
    self.element_ReLU_prime_impl(true);
  }

  pub fn sum_rows_matrix(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_sum_rows(self.id, self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = 1;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn sum_columns_matrix(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_sum_columns(self.id, self.rows, self.columns) }

    let output_rows = 1;
    let output_columns = self.columns;

    let result_matrix = Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };

    return result_matrix;
  }

  pub fn sum_columns(&self) -> Vec<f32> {
    let result_id: usize;
    unsafe { result_id = cuda_sum_columns(self.id, self.rows, self.columns) }

    let output_rows = 1;
    let output_columns = self.columns;

    let result_matrix = Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };

    let result = result_matrix.get_data()[0].to_vec();
    return result;
  }

  pub fn transpose(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_transpose(self.id, self.rows, self.columns) }

    let output_rows = self.columns;
    let output_columns = self.rows;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn max_pool(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_max_pool(self.id, self.rows, self.columns) }

    let output_rows = self.rows / 2 + self.rows % 2;
    let output_columns = self.columns / 2 + self.columns % 2;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn rotate_180(&self) -> Self {
    let result_id: usize;

    unsafe { result_id = cuda_rotate_180(self.id, self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
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
        self.id,
        self.rows,
        self.columns,
        kernel.id,
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

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
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

  let mat_ids = to_flatten.iter().map(|mat| mat.id).collect_vec();

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
  return Matrix {
    id: out_id,
    rows: out_rows,
    columns: out_cols,
  };
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
      to_unflatten.id,
      to_unflatten.get_data_length(),
      mat_rows,
      mat_cols,
      mat_ids.as_mut_ptr() as *mut c_ulonglong,
    )
  };

  return mat_ids
    .iter()
    .map(|mat_id| Matrix {
      id: *mat_id,
      rows: mat_rows,
      columns: mat_cols,
    })
    .collect_vec();
}

#[cfg(test)]
mod tests {
  use itertools::{izip, Itertools};
  use rand::prelude::Distribution;
  use rayon::vec;
  use statrs::distribution::Normal;

  use crate::{
    bindings::*, flatten_matrix_array, lib_cpu::MatrixCpu, unflatten_array_to_matrices,
    ConvolutionType, Matrix,
  };

  #[test]
  fn element_add() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let test_data_2 = Matrix::new_2d(&vec![vec![5.0, 1.0, 3.3], vec![0.0, -1.0, 1000.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![6.0, 3.0, 6.3], vec![4.0, 4.0, 1006.0]]);

    let observed_result = test_data.element_add(&test_data_2);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn element_sub() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let test_data_2 = Matrix::new_2d(&vec![vec![5.0, 1.0, 3.3], vec![0.0, -1.0, 1000.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![-4.0, 1.0, -0.3], vec![4.0, 6.0, -994.0]]);

    let observed_result = test_data.element_subtract(&test_data_2);

    assert!(matrix_are_equal(&observed_result, &expected_result, 5));
  }

  #[test]
  fn element_multiply() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let test_data_2 = Matrix::new_2d(&vec![vec![5.0, 1.0, 3.3], vec![0.0, -1.0, 1000.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![5.0, 2.0, 9.9], vec![0.0, -5.0, 6000.0]]);

    let observed_result = test_data.element_multiply(&test_data_2);
    observed_result.print();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_multiply() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = 5.0;

    let expected_result = Matrix::new_2d(&vec![vec![10.0, 20.0], vec![5.0, 15.0]]);

    let observed_result = test_data.scalar_multiply(scalar);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn matrix_multiply_gpu() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let test_data_2 = Matrix::new_2d(&vec![vec![3.0, 1.0, 5.0], vec![-2.0, 1.0, 3.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![-2.0, 6.0, 22.0], vec![-3.0, 4.0, 14.0]]);

    let observed_result = test_data.matrix_multiply(&test_data_2);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn large_matmult_cpu_gpu_agreement() {
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 1.0).unwrap();

    for common in 999..1024 {
      let rows = 1024;
      let cols = common;
      let data_1 = (0..rows)
        .map(|_| {
          (0..cols)
            .map(|_| range.sample(&mut rng) as f32)
            .collect_vec()
        })
        .collect_vec();
      let data_2 = (0..cols)
        .map(|_| {
          (0..rows)
            .map(|_| range.sample(&mut rng) as f32)
            .collect_vec()
        })
        .collect_vec();

      let mat_gpu_1 = Matrix::new_2d(&data_1);
      let mat_gpu_2 = Matrix::new_2d(&data_2);
      let mat_cpu_1 = MatrixCpu::new_2d(&data_1);
      let mat_cpu_2 = MatrixCpu::new_2d(&data_2);

      let result_gpu = mat_gpu_1.matrix_multiply(&mat_gpu_2);
      let result_cpu = mat_cpu_1.matrix_multiply(&mat_cpu_2);

      assert!(matrix_are_equal_gpu_cpu(&result_gpu, &result_cpu, 2));
    }
  }

  #[test]
  fn add_column_vector() {
    let matrix = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let vector = Matrix::new_2d(&vec![vec![1.0], vec![-1.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0]]);

    let observed_result = matrix.add_vector(&vector);
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn add_row_vector() {
    let matrix = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let vector = Matrix::new_2d(&vec![vec![1.0, 2.1, -1.5]]);

    let expected_result = Matrix::new_2d(&vec![vec![2.0, 4.1, 1.5], vec![5.0, 7.1, 4.5]]);

    let observed_result = matrix.add_vector(&vector);
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn divide_by_col_vector() {
    let matrix = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let vector = Matrix::new_2d(&vec![vec![3.0], vec![-1.0]]);

    let expected_result = Matrix::new_2d(&vec![
      vec![1.0 / 3.0, 2.0 / 3.0, 3.0 / 3.0],
      vec![-4.0, -5.0, -6.0],
    ]);

    let observed_result = matrix.divide_by_vector(&vector);
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn divide_by_row_vector() {
    let matrix = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let vector = Matrix::new_2d(&vec![vec![1.0, 2.1, -1.5]]);

    let expected_result = Matrix::new_2d(&vec![
      vec![1.0, 2.0 / 2.1, 3.0 / -1.5],
      vec![4.0, 5.0 / 2.1, 6.0 / -1.5],
    ]);

    let observed_result = matrix.divide_by_vector(&vector);
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  // TODO: Add tests for exp, relu, relu_prime, add_vector_rows, divide_vector_col_divide_vector_row
  #[test]
  fn element_exp() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let test_data_matrix = Matrix::new_2d(&test_data);

    let expected_result = test_data
      .iter()
      .map(|row| row.iter().map(|val| val.exp()).collect_vec())
      .collect_vec();

    let expected_result_matrix = Matrix::new_2d(&expected_result);

    let observed_result = test_data_matrix.element_exp();

    assert!(matrix_are_equal(
      &observed_result,
      &expected_result_matrix,
      8
    ));
  }

  #[test]
  fn element_relu() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let test_data_matrix = Matrix::new_2d(&test_data);

    let expected_result = test_data
      .iter()
      .map(|row| {
        row
          .iter()
          .map(|&val| if val > 0.0 { val } else { 0.0 })
          .collect_vec()
      })
      .collect_vec();

    let expected_result_matrix = Matrix::new_2d(&expected_result);

    let observed_result = test_data_matrix.element_ReLU();

    assert!(matrix_are_equal(
      &observed_result,
      &expected_result_matrix,
      8
    ));
  }

  #[test]
  fn element_relu_prime() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let test_data_matrix = Matrix::new_2d(&test_data);

    let expected_result = test_data
      .iter()
      .map(|row| {
        row
          .iter()
          .map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
          .collect_vec()
      })
      .collect_vec();

    let expected_result_matrix = Matrix::new_2d(&expected_result);

    let observed_result = test_data_matrix.element_ReLU_prime();

    assert!(matrix_are_equal(
      &observed_result,
      &expected_result_matrix,
      8
    ));
  }

  #[test]
  fn sum_row_matrix() {
    let matrix = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![6.0], vec![15.0]]);

    let observed_result = matrix.sum_rows_matrix();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn sum_column() {
    let matrix = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = vec![5.0, 7.0, 9.0];

    let observed_result = matrix.sum_columns();

    assert_eq!(observed_result, expected_result);
  }

  #[test]
  fn transpose() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);

    let observed_result = test_data.transpose();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn max_pool_v1() {
    let test_data = MatrixCpu::new_2d(&vec![vec![-5.0, 2.0], vec![4.0, 5.0]]);

    let expected_result = MatrixCpu::new_2d(&vec![vec![5.0]]);

    let observed_result = test_data.max_pool();

    assert!(matrix_are_equal_cpu(&observed_result, &expected_result, 8));
  }

  #[test]
  fn max_pool_v2() {
    let test_data = MatrixCpu::new_2d(&vec![vec![-5.0, 2.0, -100.0], vec![4.0, 5.0, 23.0]]);

    let expected_result = MatrixCpu::new_2d(&vec![vec![5.0, 23.0]]);

    let observed_result = test_data.max_pool();

    assert!(matrix_are_equal_cpu(&observed_result, &expected_result, 8));
  }

  #[test]
  fn max_pool_gpu_v1() {
    let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0], vec![4.0, 5.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![5.0]]);

    let observed_result = test_data.max_pool();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn max_pool_gpu_v2() {
    let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0, -100.0], vec![4.0, 5.0, 23.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![5.0, 23.0]]);

    let observed_result = test_data.max_pool();

    expected_result.print();
    observed_result.print();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn max_pool_cpu_gpu_agreement() {
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 1e8).unwrap();

    let rows = 5000;
    let cols = 784;
    let data = (0..rows)
      .map(|_| {
        (0..cols)
          .map(|_| range.sample(&mut rng) as f32)
          .collect_vec()
      })
      .collect_vec();

    let mat_gpu = Matrix::new_2d(&data).max_pool();
    let mat_cpu = MatrixCpu::new_2d(&data).max_pool();

    assert!(matrix_are_equal_gpu_cpu(&mat_gpu, &mat_cpu, 8));
  }

  #[test]
  fn rotate_180() {
    let test_data = MatrixCpu::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_result = MatrixCpu::new_2d(&vec![
      vec![9.0, 8.0, 7.0],
      vec![6.0, 5.0, 4.0],
      vec![3.0, 2.0, 1.0],
    ]);

    let observed_result = test_data.rotate_180();

    assert!(matrix_are_equal_cpu(&observed_result, &expected_result, 8));
  }

  #[test]
  fn rotate_180_2() {
    let test_data = MatrixCpu::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = MatrixCpu::new_2d(&vec![vec![6.0, 5.0, 4.0], vec![3.0, 2.0, 1.0]]);

    let observed_result = test_data.rotate_180();

    assert!(matrix_are_equal_cpu(&observed_result, &expected_result, 8));
  }

  #[test]
  fn rotate_180_gpu() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![9.0, 8.0, 7.0],
      vec![6.0, 5.0, 4.0],
      vec![3.0, 2.0, 1.0],
    ]);

    let observed_result = test_data.rotate_180();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn rotate_180_gpu_2() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![6.0, 5.0, 4.0], vec![3.0, 2.0, 1.0]]);

    let observed_result = test_data.rotate_180();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn rotate_180_cpu_gpu_agreement() {
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 1e8).unwrap();

    let rows = 5000;
    let cols = 784;
    let data = (0..rows)
      .map(|_| {
        (0..cols)
          .map(|_| range.sample(&mut rng) as f32)
          .collect_vec()
      })
      .collect_vec();

    let mat_gpu = Matrix::new_2d(&data).max_pool();
    let mat_cpu = MatrixCpu::new_2d(&data).max_pool();

    assert!(matrix_are_equal_gpu_cpu(&mat_gpu, &mat_cpu, 8));
  }

  #[test]
  fn convolution() {
    let test_data = MatrixCpu::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = MatrixCpu::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_result = MatrixCpu::new_2d(&vec![
      vec![94.0, 154.0, 106.0],
      vec![186.0, 285.0, 186.0],
      vec![106.0, 154.0, 94.0],
    ]);

    let observed_result = test_data.convolution(&kernel);

    assert!(matrix_are_equal_cpu(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_2() {
    let test_data = MatrixCpu::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = MatrixCpu::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_result = MatrixCpu::new_2d(&vec![
      vec![111.0, 178.0, 217.0, 145.0],
      vec![231.0, 348.0, 393.0, 252.0],
      vec![133.0, 190.0, 211.0, 127.0],
    ]);

    let observed_result = test_data.convolution(&kernel);

    assert!(matrix_are_equal_cpu(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_gpu_same() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![94.0, 154.0, 106.0],
      vec![186.0, 285.0, 186.0],
      vec![106.0, 154.0, 94.0],
    ]);

    let observed_result = test_data.convolution(&kernel, ConvolutionType::SAME);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_gpu_same_2() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![111.0, 178.0, 217.0, 145.0],
      vec![231.0, 348.0, 393.0, 252.0],
      vec![133.0, 190.0, 211.0, 127.0],
    ]);

    let observed_result = test_data.convolution(&kernel, ConvolutionType::SAME);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_gpu_valid_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![37.0, 47.0], vec![67.0, 77.0]]);

    let observed_result = test_data.convolution(&kernel, ConvolutionType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_gpu_valid_2() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![106.0, 127.0], vec![190.0, 211.0]]);

    let observed_result = test_data.convolution(&kernel, ConvolutionType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_gpu_valid_3() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![44.0, 54.0, 64.0], vec![84.0, 94.0, 104.0]]);

    let observed_result = test_data.convolution(&kernel, ConvolutionType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_gpu_full_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![
      vec![4.0, 11.0, 18.0, 9.0],
      vec![18.0, 37.0, 47.0, 21.0],
      vec![36.0, 67.0, 77.0, 33.0],
      vec![14.0, 23.0, 26.0, 9.0],
    ]);

    let observed_result = test_data.convolution(&kernel, ConvolutionType::FULL);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_gpu_full_2() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![9.0, 26.0, 50.0, 38.0, 21.0],
      vec![42.0, 94.0, 154.0, 106.0, 54.0],
      vec![90.0, 186.0, 285.0, 186.0, 90.0],
      vec![54.0, 106.0, 154.0, 94.0, 42.0],
      vec![21.0, 38.0, 50.0, 26.0, 9.0],
    ]);

    let observed_result = test_data.convolution(&kernel, ConvolutionType::FULL);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_cpu_gpu_agreement() {
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 1e2).unwrap();

    let rows = 5000;
    let cols = 784;
    let data = (0..rows)
      .map(|_| {
        (0..cols)
          .map(|_| range.sample(&mut rng) as f32)
          .collect_vec()
      })
      .collect_vec();

    let kernel = &vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ];

    let mat_gpu = Matrix::new_2d(&data).convolution(&Matrix::new_2d(kernel), ConvolutionType::SAME);
    let mat_cpu = MatrixCpu::new_2d(&data).convolution(&MatrixCpu::new_2d(kernel));

    assert!(matrix_are_equal_gpu_cpu(&mat_gpu, &mat_cpu, 2));
  }

  #[test]
  fn flatten_matrix_gpu() {
    let out_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let out_2 = Matrix::new_2d(&vec![
      vec![10.0, 11.0, 12.0],
      vec![13.0, 14.0, 15.0],
      vec![16.0, 17.0, 18.0],
    ]);

    let out_3 = Matrix::new_2d(&vec![
      vec![19.0, 20.0, 21.0],
      vec![22.0, 23.0, 24.0],
      vec![25.0, 26.0, 27.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![(1..28).map(|x| x as f32).collect_vec()]);

    let observed_result = flatten_matrix_array(&vec![out_1, out_2, out_3]);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn unflatten_matrix_gpu() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let mat_2 = Matrix::new_2d(&vec![
      vec![10.0, 11.0, 12.0],
      vec![13.0, 14.0, 15.0],
      vec![16.0, 17.0, 18.0],
    ]);

    let mat_3 = Matrix::new_2d(&vec![
      vec![19.0, 20.0, 21.0],
      vec![22.0, 23.0, 24.0],
      vec![25.0, 26.0, 27.0],
    ]);

    let flattened = Matrix::new_2d(&vec![(1..28).map(|x| x as f32).collect_vec()]);

    let expected_result = vec![mat_1, mat_2, mat_3];
    let observed_result = unflatten_array_to_matrices(&flattened, 3, 3);

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn stress_test_sum_rows_matrix() {
    let mut matrices = Vec::new();

    for _ in 0..1000 {
      let matrix = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
      let observed_result = matrix.sum_rows_matrix();
      matrices.push(observed_result);
    }

    let expected_result = Matrix::new_2d(&vec![vec![6.0], vec![15.0]]);
    matrices.iter().for_each(|observed_result| {
      assert!(matrix_are_equal(observed_result, &expected_result, 8));
    });
  }

  #[test]
  fn large_tranpose_cpu_gpu_agreement() {
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 1.0).unwrap();

    let rows = 5000;
    let cols = 784;
    let data = (0..rows)
      .map(|_| {
        (0..cols)
          .map(|_| range.sample(&mut rng) as f32)
          .collect_vec()
      })
      .collect_vec();

    let mat_gpu = Matrix::new_2d(&data).transpose();
    let mat_cpu = MatrixCpu::new_2d(&data).transpose();

    assert!(matrix_are_equal_gpu_cpu(&mat_gpu, &mat_cpu, 8));
  }

  fn matrix_are_equal(a: &Matrix, b: &Matrix, precision: usize) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
      println!("Matrices not the same shape!");
      return false;
    }

    a.print();
    b.print();

    let a_data = a.get_data();
    let b_data = b.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b_data[i][j], precision) {
          return false;
        }
      }
    }

    return true;
  }

  fn matrix_are_equal_cpu(a: &MatrixCpu, b: &MatrixCpu, precision: usize) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
      println!("Matrices not the same shape!");
      return false;
    }

    a.print();
    b.print();

    let a_data = a.get_data();
    let b_data = b.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b_data[i][j], precision) {
          return false;
        }
      }
    }

    return true;
  }

  fn matrix_are_equal_gpu_cpu(a: &Matrix, b: &MatrixCpu, precision: usize) -> bool {
    if a.get_data_length() < 100 && b.rows * b.columns < 100 {
      a.print();
      b.print();
    }

    if a.rows != b.rows || a.columns != b.columns {
      println!("Matrices do not even share dimensions");
      return false;
    }

    let a_data = a.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b[i][j], precision) {
          println!(
            "Matrices not equal at index: {} {} with value: {} {}",
            i, j, a_data[i][j], b[i][j]
          );
          return false;
        }
      }
    }

    return true;
  }

  fn approx_equal(a: f32, b: f32, precision: usize) -> bool {
    let tolerance = f32::powf(10.0, -1.0 * precision as f32);
    return (a - b).abs() < tolerance;
  }

  #[test]
  fn cuda_accessible() {
    unsafe {
      test();
    }
  }

  #[test]
  fn cuda_data_passing() {
    // Create vector to fill
    let len: usize = 1000;
    let mut out = Vec::<f32>::with_capacity(len);
    unsafe {
      test_array_fill(out.as_mut_ptr(), len);
      out.set_len(len);
    }

    // Ensure output is correct
    (0..len).for_each(|i| assert_eq!(out[i], i as f32));
  }
}
