use std::ops::{Index, IndexMut};

use itertools::{concat, izip, Itertools};

use crate::py_util::py_print;

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
    py_print(&"");
    py_print(&"");
    for row in 0..self.rows {
      let formatted = (0..self.columns)
        .map(|col| format!("{:<5}", self[row][col]))
        .collect::<Vec<String>>()
        .join(" ");
      py_print(&formatted);
    }
    py_print(&"");
    py_print(&"");
  }
  pub fn same_shape(&self, other: &Matrix) -> bool {
    return self.rows == other.rows && self.columns == other.columns;
  }
  pub fn element_add(&self, other: &Matrix) -> Self {
    if !self.same_shape(other) {
      panic!("Matrices not the same shape for addition!");
    }

    let mut result = Self::zeros(self.rows, self.columns);

    for i in 0..result.rows {
      for j in 0..result.columns {
        result[i][j] = self[i][j] + other[i][j];
      }
    }

    return result;
  }

  pub fn element_multiply(&self, other: &Matrix) -> Self {
    if !self.same_shape(other) {
      panic!("Matrices not the same shape for element_multiply!");
    }

    let mut result = Self::zeros(self.rows, self.columns);

    for i in 0..result.rows {
      for j in 0..result.columns {
        result[i][j] = self[i][j] * other[i][j];
      }
    }

    return result;
  }

  pub fn element_subtract(&self, other: &Matrix) -> Self {
    if !self.same_shape(other) {
      panic!("Matrices not the same shape for element_subtract!");
    }

    let mut result = Self::zeros(self.rows, self.columns);

    for i in 0..result.rows {
      for j in 0..result.columns {
        result[i][j] = self[i][j] - other[i][j];
      }
    }

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

    // Result dimensions will be rows self x columns other
    let result_rows = self.rows;
    let result_columns = other.columns;
    let mut result = Self::zeros(result_rows, result_columns);

    // Row of this * column of that
    for result_row in 0..result_rows {
      for vector_offset in 0..self.columns {
        for result_col in 0..result_columns {
          result[result_row][result_col] +=
            self[result_row][vector_offset] * other[vector_offset][result_col];
        }
      }
    }

    return result;
  }

  pub fn scalar_multiply(&self, scalar: f64) -> Self {
    let mut result = Self::zeros(self.rows, self.columns);

    for i in 0..result.rows {
      for j in 0..result.columns {
        result[i][j] = self[i][j] * scalar;
      }
    }

    return result;
  }

  pub fn transpose(&self) -> Self {
    // Create result 2d vec with #rows and #columns flipped
    let mut result = Self::zeros(self.columns, self.rows);

    // Every row becomes column
    for i in 0..self.rows {
      for j in 0..self.columns {
        result[j][i] = self[i][j];
      }
    }

    return result;
  }
  pub fn element_apply(&self, func: &dyn Fn(f64) -> f64) -> Self {
    let mut result = Self::zeros(self.rows, self.columns);

    for i in 0..result.rows {
      for j in 0..result.columns {
        result[i][j] = func(self[i][j]);
      }
    }

    return result;
  }

  pub fn sum_rows(&self) -> Vec<f64> {
    return self.iter().map(|row| row.iter().sum()).collect_vec();
  }

  pub fn sum_rows_matrix(&self) -> Matrix {
    let mut result = Self::zeros(self.rows, 1);

    for i in 0..self.rows {
      for j in 0..self.columns {
        result[i][0] += self[i][j];
      }
    }

    return result;
  }

  pub fn sum_columns(&self) -> Vec<f64> {
    let mut result = vec![0.0; self.columns];

    for i in 0..self.rows {
      for j in 0..self.columns {
        result[j] += self[i][j];
      }
    }

    return result;
  }
}
