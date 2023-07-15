pub mod bindings;
pub mod lib_cpu;

use bindings::*;
use std::ffi::c_double;

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

  pub fn get_data(&self) -> Vec<Vec<f64>> {
    let mut data = Vec::<c_double>::with_capacity(self.get_data_length());
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
    let data = Vec::<c_double>::with_capacity(rows * columns);
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

  pub fn new_2d(data: &Vec<Vec<f64>>) -> Self {
    if data.len() == 0 {
      return Self::no_fill(0, 0);
    }

    let rows = data.len();
    let columns = data[0].len();
    let mut flattened = Vec::<f64>::with_capacity(rows * columns);
    data.iter().for_each(|row| flattened.extend(row));

    let id;
    unsafe {
      id = register_matrix(flattened.as_ptr(), rows, columns);
    }

    return Matrix { id, rows, columns };
  }

  pub fn print(&self) {
    println!("");
    println!("");

    // OPTIMIZE!
    let data = self.get_data();

    for row in 0..self.rows {
      let formatted = (0..self.columns)
        .map(|col| format!("{:<5}", data[row][col]))
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
      )
    }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn element_subtract(&self, other: &Matrix) -> Self {
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
      )
    }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn element_multiply(&self, other: &Matrix) -> Self {
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
      )
    }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn scalar_multiply(&self, scalar: f64) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_scalar_multiply(self.id, self.rows, self.columns, scalar) }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
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

  pub fn add_vector(&self, other: &Matrix) -> Self {
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
      )
    }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn divide_by_vector(&self, other: &Matrix) -> Self {
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
      )
    }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  pub fn element_exp(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_element_exp(self.id, self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_element_ReLU(self.id, self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }

  #[allow(non_snake_case)]
  pub fn element_ReLU_prime(&self) -> Self {
    let result_id: usize;
    unsafe { result_id = cuda_element_ReLU_prime(self.id, self.rows, self.columns) }

    let output_rows = self.rows;
    let output_columns = self.columns;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
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

  pub fn sum_columns(&self) -> Vec<f64> {
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

    // Fast transpose, no memory operations
    // let is_vector = self.columns == 1 || self.rows == 1;
    // if is_vector {
    //   return Matrix {
    //     id: self.id,
    //     rows: self.columns,
    //     columns: self.rows,
    //   };
    // }
    unsafe { result_id = cuda_transpose(self.id, self.rows, self.columns) }

    let output_rows = self.columns;
    let output_columns = self.rows;

    return Matrix {
      id: result_id,
      rows: output_rows,
      columns: output_columns,
    };
  }
}

#[cfg(test)]
mod tests {
  use itertools::Itertools;
  use rand::prelude::Distribution;
  use statrs::distribution::Normal;

  use crate::bindings::*;
  use crate::lib_cpu::MatrixCpu;
  use crate::Matrix;
  use std::ffi::c_double;

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
    observed_result.print();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
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
      .map(|_| (0..cols).map(|_| range.sample(&mut rng)).collect_vec())
      .collect_vec();

    let mat_gpu = Matrix::new_2d(&data).transpose();
    let mat_cpu = MatrixCpu::new_2d(&data).transpose();

    matrix_are_equal_gpu_cpu(&mat_gpu, &mat_cpu, 12);
  }

  fn matrix_are_equal(a: &Matrix, b: &Matrix, precision: usize) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
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
    a.print();
    b.print();

    if a.rows != b.rows || a.columns != b.columns {
      println!("Matrices do not even share dimensions");
      return false;
    }

    let a_data = a.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b[i][j], precision) {
          println!("Matrices not equal at {} {}", a_data[i][j], b[i][j]);
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
