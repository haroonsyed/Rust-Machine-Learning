use std::ops::{Index, IndexMut};

use itertools::Itertools;
use rayon::prelude::*;

pub struct MatrixCpu {
  pub data: Vec<f32>,
  pub rows: usize,
  pub columns: usize,
}

impl Index<usize> for MatrixCpu {
  type Output = [f32];

  fn index(&self, row: usize) -> &Self::Output {
    let start = row * self.columns;
    let end = start + self.columns;
    &self.data[start..end]
  }
}

impl IndexMut<usize> for MatrixCpu {
  fn index_mut(&mut self, row: usize) -> &mut Self::Output {
    let start = row * self.columns;
    let end = start + self.columns;
    &mut self.data[start..end]
  }
}

impl MatrixCpu {
  pub fn zeros(rows: usize, columns: usize) -> Self {
    return MatrixCpu {
      data: vec![0.0; rows * columns],
      rows,
      columns,
    };
  }

  pub fn get_rows(&self) -> usize {
    return self.rows;
  }

  pub fn get_columns(&self) -> usize {
    return self.columns;
  }

  pub fn get_data(&self) -> Vec<Vec<f32>> {
    return self.iter().map(|x| x.to_vec()).collect_vec();
  }

  pub fn new_2d(data: &Vec<Vec<f32>>) -> Self {
    let rows = data.len();
    let columns = data[0].len();
    let mut flattened = Vec::<f32>::with_capacity(rows * columns);
    data.iter().for_each(|row| flattened.extend(row));
    return MatrixCpu {
      data: flattened,
      rows,
      columns,
    };
  }

  pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
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
  pub fn same_shape(&self, other: &MatrixCpu) -> bool {
    return self.rows == other.rows && self.columns == other.columns;
  }
  pub fn element_add(&self, other: &MatrixCpu) -> Self {
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

  pub fn element_multiply(&self, other: &MatrixCpu) -> Self {
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

  pub fn element_subtract(&self, other: &MatrixCpu) -> Self {
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

  pub fn add_vector_to_columns(&self, other: &MatrixCpu) -> Self {
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

  pub fn matrix_multiply(&self, other: &MatrixCpu) -> Self {
    // Bound Check
    if self.columns != other.rows {
      panic!("Matrices not compatible shape for mat mult!");
    }

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

  pub fn scalar_multiply(&self, scalar: f32) -> Self {
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
  pub fn element_apply(&self, func: &dyn Fn(f32) -> f32) -> Self {
    let mut result = Self::zeros(self.rows, self.columns);

    for i in 0..result.rows {
      for j in 0..result.columns {
        result[i][j] = func(self[i][j]);
      }
    }

    return result;
  }

  pub fn sum_rows(&self) -> Vec<f32> {
    return self.iter().map(|row| row.iter().sum()).collect_vec();
  }

  pub fn sum_rows_matrix(&self) -> MatrixCpu {
    let mut result = Self::zeros(self.rows, 1);

    for i in 0..self.rows {
      for j in 0..self.columns {
        result[i][0] += self[i][j];
      }
    }

    return result;
  }

  pub fn sum_columns(&self) -> Vec<f32> {
    let mut result = vec![0.0; self.columns];

    for i in 0..self.rows {
      for j in 0..self.columns {
        result[j] += self[i][j];
      }
    }

    return result;
  }

  pub fn max_pool(&self) -> Self {
    let result_rows = self.rows / 2 + self.rows % 2;
    let result_cols = self.columns / 2 + self.columns % 2;
    let mut result = Self::zeros(result_rows, result_cols);

    for i in 0..result_rows {
      for j in 0..result_cols {
        let block_start_row = i * 2;
        let block_start_col = j * 2;

        // let block_00_oob = false;
        let block_01_oob = (block_start_col + 1) >= self.columns;
        let block_10_oob = (block_start_row + 1) >= self.rows;
        let block_11_oob = block_01_oob || block_10_oob;

        // grab the 4x4 surrounding items
        let block_00 = self[block_start_row][block_start_col];
        let block_01 = if block_01_oob {
          -1e30
        } else {
          self[block_start_row][block_start_col + 1]
        };
        let block_10 = if block_10_oob {
          -1e30
        } else {
          self[block_start_row + 1][block_start_col]
        };
        let block_11 = if block_11_oob {
          -1e30
        } else {
          self[block_start_row + 1][block_start_col + 1]
        };

        result[i][j] += f32::max(f32::max(block_00, block_01), f32::max(block_10, block_11));
      }
    }

    return result;
  }

  pub fn rotate_180(&self) -> Self {
    let result_rows = self.rows;
    let result_cols = self.columns;
    let mut result = Self::zeros(result_rows, result_cols);

    for i in 0..result_rows {
      for j in 0..result_cols {
        // Rotating an array 180 means
        // x_output = length - x_current
        // y_output = height - y_current
        let x_out = self.columns - 1 - j;
        let y_out = self.rows - 1 - i;
        result[y_out][x_out] = self[i][j];
      }
    }

    return result;
  }

  pub fn correlate(&self, kernel: &MatrixCpu) -> Self {
    let result_rows = self.rows;
    let result_cols = self.columns;
    let mut result = Self::zeros(result_rows, result_cols);

    for i in 0..result_rows {
      for j in 0..result_cols {
        let mut result_val = 0.0;
        let apothem = kernel.rows / 2;
        for m in 0..kernel.rows {
          for n in 0..kernel.columns {
            let input_y = i as isize - apothem as isize + m as isize;
            let input_x = j as isize - apothem as isize + n as isize;
            if input_y >= 0
              && input_y < self.rows as isize
              && input_x >= 0
              && input_x < self.columns as isize
            {
              result_val += self[input_y as usize][input_x as usize] * kernel[m][n];
            }
          }
        }

        result[i][j] = result_val;
      }
    }

    return result;
  }
}
