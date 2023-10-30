use crate::cuda_bindings::*;
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
