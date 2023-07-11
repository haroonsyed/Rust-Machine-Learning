pub mod bindings;

use bindings::*;
use itertools::{concat, Itertools};
use rayon::prelude::*;
use std::{
  ffi::c_double,
  ops::{Index, IndexMut},
};

pub struct Matrix {
  pub data: Vec<f64>,
  pub rows: usize,
  pub columns: usize,
}

impl Index<usize> for Matrix {
  type Output = [f64];

  fn index(&self, row: usize) -> &Self::Output {
    let start = row * self.columns;
    let end = start + self.columns;
    &self.data[start..end]
  }
}

impl IndexMut<usize> for Matrix {
  fn index_mut(&mut self, row: usize) -> &mut Self::Output {
    let start = row * self.columns;
    let end = start + self.columns;
    &mut self.data[start..end]
  }
}

impl Matrix {
  pub fn zeros(rows: usize, columns: usize) -> Self {
    return Matrix {
      data: vec![0.0; rows * columns],
      rows,
      columns,
    };
  }

  pub fn new_2d(data: Vec<Vec<f64>>) -> Self {
    let rows = data.len();
    let columns = data[0].len();
    let flattened = concat(data);
    return Matrix {
      data: flattened,
      rows,
      columns,
    };
  }

  pub fn iter(&self) -> impl Iterator<Item = &[f64]> {
    self.data.chunks(self.columns)
  }

  pub fn print(&self) {
    println!("");
    println!("");
    for row in 0..self.rows {
      let formatted = (0..self.columns)
        .map(|col| format!("{:<5}", self[row][col]))
        .collect::<Vec<String>>()
        .join(" ");
      println!("{}", formatted);
    }
    println!("");
    println!("");
  }
  pub fn same_shape(&self, other: &Matrix) -> bool {
    return self.rows == other.rows && self.columns == other.columns;
  }
  pub fn element_add(&self, other: &Matrix) -> Self {
    if !self.same_shape(other) {
      panic!("Matrices not the same shape for addition!");
    }

    let mut result = Self::zeros(self.rows, self.columns);

    result
      .data
      .par_chunks_mut(result.columns)
      .enumerate()
      .for_each(|(result_row_index, result_row_slice)| {
        for j in 0..result.columns {
          result_row_slice[j] = self[result_row_index][j] + other[result_row_index][j];
        }
      });

    return result;
  }

  pub fn element_multiply(&self, other: &Matrix) -> Self {
    if !self.same_shape(other) {
      panic!("Matrices not the same shape for element_multiply!");
    }

    let mut result = Self::zeros(self.rows, self.columns);

    result
      .data
      .par_chunks_mut(result.columns)
      .enumerate()
      .for_each(|(result_row_index, result_row_slice)| {
        for j in 0..result.columns {
          result_row_slice[j] = self[result_row_index][j] * other[result_row_index][j];
        }
      });

    return result;
  }

  pub fn element_subtract(&self, other: &Matrix) -> Self {
    if !self.same_shape(other) {
      panic!("Matrices not the same shape for element_subtract!");
    }

    let mut result = Self::zeros(self.rows, self.columns);

    result
      .data
      .par_chunks_mut(result.columns)
      .enumerate()
      .for_each(|(result_row_index, result_row_slice)| {
        for j in 0..result.columns {
          result_row_slice[j] = self[result_row_index][j] - other[result_row_index][j];
        }
      });

    return result;
  }

  pub fn add_vector_to_columns(&self, other: &Matrix) -> Self {
    if self.rows != other.rows {
      panic!("Matrices not the correct shape for add_vector_to_columns!");
    }

    let mut result = Self::zeros(self.rows, self.columns);

    for i in 0..result.rows {
      for j in 0..result.columns {
        result[i][j] = self[i][j] + other[i][0];
      }
    }

    return result;
  }

  pub fn matrix_multiply(&self, other: &Matrix) -> Self {
    // Bound Check
    if self.columns != other.rows {
      panic!("Matrices not compatible shape for mat mult!");
    }

    // For larger matrices use gpu (idk i'm gonna say anything with > 16384 elements)
    let gpu_threshold = 16384;
    let total_output_elements = self.rows * other.columns;

    if total_output_elements < gpu_threshold {
      return self.matrix_multiply_cpu(other);
    } else {
      return self.matrix_multiply_gpu(other);
    }
  }

  fn matrix_multiply_gpu(&self, other: &Matrix) -> Self {
    // Create vector to fill
    let result_rows = self.rows;
    let result_columns = other.columns;
    let result_len: usize = result_rows * result_columns;
    let mut result = Vec::<c_double>::with_capacity(result_len);

    // Run cuda kernel
    unsafe {
      cuda_matrix_multiply(
        self.data.as_ptr(),
        self.rows,
        self.columns,
        other.data.as_ptr(),
        other.rows,
        other.columns,
        result.as_mut_ptr(),
        result_rows,
        result_columns,
      );
      result.set_len(result_len);
    }

    return Matrix {
      data: result,
      rows: result_rows,
      columns: result_columns,
    };
  }

  fn matrix_multiply_cpu(&self, other: &Matrix) -> Self {
    // Result dimensions will be rows self x columns other
    let result_rows = self.rows;
    let result_columns = other.columns;
    let mut result = Self::zeros(result_rows, result_columns);

    // Row of this * column of that
    result
      .data
      .par_chunks_mut(result_columns)
      .enumerate()
      .for_each(|(result_row_index, result_row_slice)| {
        for vector_offset in 0..self.columns {
          for result_col in 0..result_columns {
            result_row_slice[result_col] +=
              self[result_row_index][vector_offset] * other[vector_offset][result_col];
          }
        }
      });

    return result;
  }

  pub fn scalar_multiply(&self, scalar: f64) -> Self {
    let mut result = Self::zeros(self.rows, self.columns);

    result
      .data
      .par_chunks_mut(result.columns)
      .enumerate()
      .for_each(|(result_row_index, result_row_slice)| {
        for j in 0..result.columns {
          result_row_slice[j] = self[result_row_index][j] * scalar;
        }
      });

    return result;
  }

  pub fn transpose(&self) -> Self {
    // Create result 2d vec with #rows and #columns flipped
    let mut result = Self::zeros(self.columns, self.rows);

    // Every row becomes column
    result
      .data
      .par_chunks_mut(result.columns)
      .enumerate()
      .for_each(|(result_row_index, result_row_slice)| {
        for j in 0..result.columns {
          result_row_slice[j] = self[j][result_row_index];
        }
      });

    return result;
  }
  pub fn element_apply(&self, func: &dyn Fn(f64) -> f64) -> Self {
    let mut result = Self::zeros(self.rows, self.columns);

    for i in 0..self.rows {
      for j in 0..self.columns {
        result[i][j] += func(self[i][j]);
      }
    }

    return result;
  }

  pub fn sum_rows(&self) -> Vec<f64> {
    return self.iter().map(|row| row.iter().sum()).collect_vec();
  }

  pub fn sum_rows_matrix(&self) -> Matrix {
    let mut result = Self::zeros(self.rows, 1);

    result
      .data
      .par_chunks_mut(1)
      .enumerate()
      .for_each(|(result_row_index, result_row_slice)| {
        result_row_slice[0] = self[result_row_index].iter().sum();
      });

    return result;
  }

  pub fn sum_columns(&self) -> Vec<f64> {
    let mut result = vec![0.0; self.columns];

    result
      .par_iter_mut()
      .enumerate()
      .for_each(|(column_sum_index, val)| {
        for i in 0..self.rows {
          *val += self[i][column_sum_index];
        }
      });

    return result;
  }
}

#[cfg(test)]
mod tests {
  use crate::bindings::*;
  use crate::Matrix;
  use std::ffi::c_double;

  #[test]
  fn element_add() {
    let test_data = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let test_data_2 = Matrix::new_2d(vec![vec![5.0, 1.0, 3.3], vec![0.0, -1.0, 1000.0]]);

    let expected_result = Matrix::new_2d(vec![vec![6.0, 3.0, 6.3], vec![4.0, 4.0, 1006.0]]);

    let observed_result = test_data.element_add(&test_data_2);

    assert!(matrix_are_equal(observed_result, expected_result, 8));
  }

  #[test]
  fn element_sub() {
    let test_data = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let test_data_2 = Matrix::new_2d(vec![vec![5.0, 1.0, 3.3], vec![0.0, -1.0, 1000.0]]);

    let expected_result = Matrix::new_2d(vec![vec![-4.0, 1.0, -0.3], vec![4.0, 6.0, -994.0]]);

    let observed_result = test_data.element_subtract(&test_data_2);
    observed_result.print();

    assert!(matrix_are_equal(observed_result, expected_result, 8));
  }

  #[test]
  fn matrix_multiply_cpu() {
    let test_data = Matrix::new_2d(vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let test_data_2 = Matrix::new_2d(vec![vec![3.0, 1.0, 5.0], vec![-2.0, 1.0, 3.0]]);

    let expected_result = Matrix::new_2d(vec![vec![-2.0, 6.0, 22.0], vec![-3.0, 4.0, 14.0]]);

    let observed_result = test_data.matrix_multiply_cpu(&test_data_2);

    assert!(matrix_are_equal(observed_result, expected_result, 8));
  }

  #[test]
  fn matrix_multiply_gpu() {
    let test_data = Matrix::new_2d(vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let test_data_2 = Matrix::new_2d(vec![vec![3.0, 1.0, 5.0], vec![-2.0, 1.0, 3.0]]);

    let expected_result = Matrix::new_2d(vec![vec![-2.0, 6.0, 22.0], vec![-3.0, 4.0, 14.0]]);

    let observed_result = test_data.matrix_multiply_gpu(&test_data_2);

    assert!(matrix_are_equal(observed_result, expected_result, 8));
  }

  #[test]
  fn transpose() {
    let test_data = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);

    let observed_result = test_data.transpose();

    assert!(matrix_are_equal(observed_result, expected_result, 8));
  }

  #[test]
  fn element_apply() {
    let test_data = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]);

    let observed_result = test_data.element_apply(&|x| x - 1.0);

    assert!(matrix_are_equal(observed_result, expected_result, 8));
  }

  #[test]
  fn add_vector_to_columns() {
    let matrix = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let vector = Matrix::new_2d(vec![vec![1.0], vec![-1.0]]);

    let expected_result = Matrix::new_2d(vec![vec![2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0]]);

    let observed_result = matrix.add_vector_to_columns(&vector);

    assert!(matrix_are_equal(observed_result, expected_result, 8));
  }

  #[test]
  fn sum_row() {
    let matrix = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = vec![6.0, 15.0];

    let observed_result = matrix.sum_rows();

    assert_eq!(observed_result, expected_result);
  }

  #[test]
  fn sum_row_matrix() {
    let matrix = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(vec![vec![6.0], vec![15.0]]);

    let observed_result = matrix.sum_rows_matrix();

    assert!(matrix_are_equal(observed_result, expected_result, 8));
  }

  #[test]
  fn sum_column() {
    let matrix = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = vec![5.0, 7.0, 9.0];

    let observed_result = matrix.sum_columns();

    assert_eq!(observed_result, expected_result);
  }

  fn matrix_are_equal(a: Matrix, b: Matrix, precision: usize) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
      return false;
    }

    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a[i][j], b[i][j], precision) {
          return false;
        }
      }
    }

    return true;
  }

  fn approx_equal(a: f64, b: f64, precision: usize) -> bool {
    let tolerance = f64::powf(10.0, -1.0 * precision as f64);
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
    let mut out = Vec::<c_double>::with_capacity(len);
    unsafe {
      test_array_fill(out.as_mut_ptr(), len);
      out.set_len(len);
    }

    // Ensure output is correct
    (0..len).for_each(|i| assert_eq!(out[i], i as f64));
  }
}
