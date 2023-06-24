use itertools::{izip, Itertools};

use crate::py_util::py_print;

pub struct Matrix {
  pub data: Vec<Vec<f64>>,
}

impl Matrix {
  pub fn print(&self) {
    println!();
    for row in self.data.iter() {
      let formatted = row
        .iter()
        .map(|num| format! {"{:<5}", num})
        .collect::<Vec<String>>()
        .join(" ");
      println!("{}", formatted);
      py_print(&formatted);
    }
    println!();
  }
  pub fn get_rows(&self) -> usize {
    return self.data.len();
  }
  pub fn get_columns(&self) -> usize {
    return self.data[0].len();
  }
  pub fn same_shape(&self, other: &Matrix) -> bool {
    return self.get_rows() == other.get_rows() && self.get_columns() == other.get_columns();
  }
  pub fn element_add(&self, other: &Matrix) -> Self {
    if !self.same_shape(other) {
      panic!("Matrices not the same shape for addition!");
    }

    let result_data = izip!(self.data.iter(), other.data.iter())
      .map(|(row_a, row_b)| {
        izip!(row_a.iter(), row_b.iter())
          .map(|(&a, &b)| a + b)
          .collect()
      })
      .collect();

    return Matrix { data: result_data };
  }

  pub fn add_vector_to_columns(&self, other: &Matrix) -> Self {
    if self.get_rows() != other.get_rows() {
      panic!("Matrices not the correct shape for add_vector_to_columns!");
    }

    let result_data = self
      .data
      .iter()
      .enumerate()
      .map(|(row_index, row_data)| {
        row_data
          .iter()
          .map(|val| val + other.data[row_index][0])
          .collect_vec()
      })
      .collect_vec();

    return Matrix { data: result_data };
  }

  pub fn matrix_multiply(&self, other: &Matrix) -> Self {
    // Bound Check
    if self.get_columns() != other.get_rows() {
      panic!("Matrices not compatible shape for mat mult!");
    }

    // Result dimensions will be rows self x columns other
    let result_rows = self.get_rows();
    let result_columns = other.get_columns();
    let mut result_data = vec![vec![0.0; result_columns]; result_rows];

    // Row of this * column of that
    for result_col in 0..result_columns {
      for result_row in 0..result_rows {
        for vector_offset in 0..self.get_columns() {
          result_data[result_row][result_col] +=
            self.data[result_row][vector_offset] * other.data[vector_offset][result_col];
        }
      }
    }

    return Matrix { data: result_data };
  }
  pub fn transpose(&self) -> Self {
    // Create result 2d vec with #rows and #columns flipped
    let mut result_data = vec![vec![0.0; self.get_rows()]; self.get_columns()];

    // Every row becomes column
    for i in 0..self.get_rows() {
      for j in 0..self.get_columns() {
        result_data[j][i] = self.data[i][j];
      }
    }

    return Matrix { data: result_data };
  }
  pub fn element_apply(&self, func: &dyn Fn(f64) -> f64) -> Self {
    let result_data = self
      .data
      .iter()
      .map(|row| row.iter().map(|&a| func(a)).collect())
      .collect();

    return Matrix { data: result_data };
  }

  pub fn sum_rows(&self) -> Vec<f64> {
    return self.data.iter().map(|row| row.iter().sum()).collect_vec();
  }

  pub fn sum_columns(&self) -> Vec<f64> {
    let mut result = vec![0.0; self.get_columns()];

    for i in 0..self.get_rows() {
      for j in 0..self.get_columns() {
        result[j] += self.data[i][j];
      }
    }

    return result;
  }
}
