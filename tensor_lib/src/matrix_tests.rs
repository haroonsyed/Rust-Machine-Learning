#[cfg(test)]
mod tests {
  use crate::{
    convolution_packed, cuda_bindings::*, flatten_matrix_array, img2col, matrix::*,
    matrix_cpu::MatrixCpu, unflatten_array_strided_to_matrices, unflatten_array_to_matrices,
    ConvolutionType,
  };
  use itertools::{izip, Itertools};
  use rand::prelude::Distribution;
  use statrs::distribution::Normal;

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
  fn scalar_multiply_inplace() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = 5.0;

    let expected_result = Matrix::new_2d(&vec![vec![10.0, 20.0], vec![5.0, 15.0]]);

    let observed_result = test_data.scalar_multiply_inplace(scalar);

    assert_eq!(test_data.get_id(), observed_result.get_id());
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_divide() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = 5.0;

    let expected_result = Matrix::new_2d(&vec![vec![0.4, 0.8], vec![0.2, 0.6]]);

    let observed_result = test_data.scalar_divide(scalar);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_divide_inplace() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = 5.0;

    let expected_result = Matrix::new_2d(&vec![vec![0.4, 0.8], vec![0.2, 0.6]]);

    let observed_result = test_data.scalar_divide_inplace(scalar);

    assert_eq!(test_data.get_id(), observed_result.get_id());
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_add() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = 5.0;

    let expected_result = Matrix::new_2d(&vec![vec![7.0, 9.0], vec![6.0, 8.0]]);

    let observed_result = test_data.scalar_add(scalar);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_add_inplace() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = 5.0;

    let expected_result = Matrix::new_2d(&vec![vec![7.0, 9.0], vec![6.0, 8.0]]);

    let observed_result = test_data.scalar_add_inplace(scalar);

    assert_eq!(test_data.get_id(), observed_result.get_id());
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_subtract() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = 5.0;

    let expected_result = Matrix::new_2d(&vec![vec![-3.0, -1.0], vec![-4.0, -2.0]]);

    let observed_result = test_data.scalar_subtract(scalar);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_subtract_inplace() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = 5.0;

    let expected_result = Matrix::new_2d(&vec![vec![-3.0, -1.0], vec![-4.0, -2.0]]);

    let observed_result = test_data.scalar_subtract_inplace(scalar);

    assert_eq!(test_data.get_id(), observed_result.get_id());
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_multiply_matrix() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = Matrix::new_1d(&vec![5.0], 1, 1);

    let expected_result = Matrix::new_2d(&vec![vec![10.0, 20.0], vec![5.0, 15.0]]);

    let observed_result = test_data.scalar_multiply_matrix(&scalar);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_multiply_matrix_inplace() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = Matrix::new_1d(&vec![5.0], 1, 1);

    let expected_result = Matrix::new_2d(&vec![vec![10.0, 20.0], vec![5.0, 15.0]]);

    let observed_result = test_data.scalar_multiply_matrix_inplace(&scalar);

    assert_eq!(test_data.get_id(), observed_result.get_id());
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_divide_matrix() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = Matrix::new_1d(&vec![5.0], 1, 1);

    let expected_result = Matrix::new_2d(&vec![vec![0.4, 0.8], vec![0.2, 0.6]]);

    let observed_result = test_data.scalar_divide_matrix(&scalar);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_divide_matrix_inplace() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = Matrix::new_1d(&vec![5.0], 1, 1);

    let expected_result = Matrix::new_2d(&vec![vec![0.4, 0.8], vec![0.2, 0.6]]);

    let observed_result = test_data.scalar_divide_matrix_inplace(&scalar);

    assert_eq!(test_data.get_id(), observed_result.get_id());
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_add_matrix() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = Matrix::new_1d(&vec![5.0], 1, 1);

    let expected_result = Matrix::new_2d(&vec![vec![7.0, 9.0], vec![6.0, 8.0]]);

    let observed_result = test_data.scalar_add_matrix(&scalar);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_add_matrix_inplace() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = Matrix::new_1d(&vec![5.0], 1, 1);

    let expected_result = Matrix::new_2d(&vec![vec![7.0, 9.0], vec![6.0, 8.0]]);

    let observed_result = test_data.scalar_add_matrix_inplace(&scalar);

    assert_eq!(test_data.get_id(), observed_result.get_id());
    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_subtract_matrix() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = Matrix::new_1d(&vec![5.0], 1, 1);

    let expected_result = Matrix::new_2d(&vec![vec![-3.0, -1.0], vec![-4.0, -2.0]]);

    let observed_result = test_data.scalar_subtract_matrix(&scalar);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn scalar_subtract_matrix_inplace() {
    let test_data = Matrix::new_2d(&vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let scalar = Matrix::new_1d(&vec![5.0], 1, 1);

    let expected_result = Matrix::new_2d(&vec![vec![-3.0, -1.0], vec![-4.0, -2.0]]);

    let observed_result = test_data.scalar_subtract_matrix_inplace(&scalar);

    assert_eq!(test_data.get_id(), observed_result.get_id());
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

    let (observed_result, _) = test_data.max_pool();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn max_pool_gpu_v2() {
    let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0, -100.0], vec![4.0, 5.0, 23.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![5.0, 23.0]]);

    let (observed_result, _) = test_data.max_pool();

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

    let (mat_gpu, _) = Matrix::new_2d(&data).max_pool();
    let mat_cpu = MatrixCpu::new_2d(&data).max_pool();

    assert!(matrix_are_equal_gpu_cpu(&mat_gpu, &mat_cpu, 8));
  }

  #[test]
  fn max_pool_bitmask_gpu() {
    let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0, -100.0], vec![4.0, 5.0, 23.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 1.0]]);

    let (_, observed_result) = test_data.max_pool();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn max_pool_bitmask_gpu_2() {
    let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0], vec![4.0, 5.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![0.0, 0.0], vec![0.0, 1.0]]);

    let (_, observed_result) = test_data.max_pool();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
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
      vec![7.7, 8.0, 9.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![9.0, 8.0, 7.7],
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

    let mat_gpu = Matrix::new_2d(&data).rotate_180();
    let mat_cpu = MatrixCpu::new_2d(&data).rotate_180();

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
  fn packed_convolution_same_1() {
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
    let kernel_2 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![94.0, 154.0, 106.0],
        vec![186.0, 285.0, 186.0],
        vec![106.0, 154.0, 94.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![94.0, 154.0, 106.0],
        vec![186.0, 286.0, 188.0],
        vec![106.0, 158.0, 99.0],
      ],);
      16
    ]);

    let mut observed_result = convolution_packed(
      &vec![&test_data; 16],
      &vec![&kernel; 16],
      ConvolutionType::SAME,
    );
    observed_result.extend(convolution_packed(
      &vec![&test_data; 16],
      &vec![&kernel_2; 16],
      ConvolutionType::SAME,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
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
  fn packed_convolution_valid_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let kernel_2 = Matrix::new_2d(&vec![vec![1.0, 1.0], vec![1.0, 1.0]]);

    let mut expected_result = vec![Matrix::new_2d(&vec![vec![37.0, 47.0], vec![67.0, 77.0]]); 16];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![12.0, 16.0],
        vec![24.0, 28.0]
      ]);
      16
    ]);

    let mut observed_result = convolution_packed(
      &vec![&test_data; 16],
      &vec![&kernel; 16],
      ConvolutionType::VALID,
    );
    observed_result.extend(convolution_packed(
      &vec![&test_data; 16],
      &vec![&kernel_2; 16],
      ConvolutionType::VALID,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
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
  fn packed_convolution_full_1() {
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
    let kernel_2 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![9.0, 26.0, 50.0, 38.0, 21.0],
        vec![42.0, 94.0, 154.0, 106.0, 54.0],
        vec![90.0, 186.0, 285.0, 186.0, 90.0],
        vec![54.0, 106.0, 154.0, 94.0, 42.0],
        vec![21.0, 38.0, 50.0, 26.0, 9.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![9.0, 26.0, 50.0, 38.0, 21.0],
        vec![42.0, 94.0, 154.0, 106.0, 54.0],
        vec![90.0, 186.0, 286.0, 188.0, 93.0],
        vec![54.0, 106.0, 158.0, 99.0, 48.0],
        vec![21.0, 38.0, 57.0, 34.0, 18.0],
      ],);
      16
    ]);

    let mut observed_result = convolution_packed(
      &vec![&test_data; 16],
      &vec![&kernel; 16],
      ConvolutionType::FULL,
    );
    observed_result.extend(convolution_packed(
      &vec![&test_data; 16],
      &vec![&kernel_2; 16],
      ConvolutionType::FULL,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
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

  #[test]
  fn img2col_depth_1() {
    let input = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_output = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 4.0, 5.0],
      vec![2.0, 3.0, 5.0, 6.0],
      vec![4.0, 5.0, 7.0, 8.0],
      vec![5.0, 6.0, 8.0, 9.0],
    ]);

    let observed_output = img2col(&vec![input], 2, 2);

    assert!(matrix_are_equal(&observed_output, &expected_output, 8));
  }

  #[test]
  fn img2col_depth_2() {
    let input = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let input2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_output = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 4.0, 5.0],
      vec![2.0, 3.0, 5.0, 6.0],
      vec![4.0, 5.0, 7.0, 8.0],
      vec![5.0, 6.0, 8.0, 9.0],
      vec![1.0, 2.0, 4.0, 5.0],
      vec![2.0, 3.0, 5.0, 6.0],
      vec![4.0, 5.0, 7.0, 8.0],
      vec![5.0, 6.0, 8.0, 9.0],
    ]);

    let observed_output = img2col(&vec![input, input2], 2, 2);

    assert!(matrix_are_equal(&observed_output, &expected_output, 8));
  }

  #[test]
  fn img2col_kernel_size_1() {
    let input = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_output = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]);

    let observed_output = img2col(&vec![input], 1, 1);

    assert!(matrix_are_equal(&observed_output, &expected_output, 8));
  }

  #[test]
  fn convolution_v2_gpu_valid_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![37.0, 47.0], vec![67.0, 77.0]]);

    let observed_result = test_data.convolution_v2(&kernel, ConvolutionType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_v2_gpu_valid_2() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![106.0, 127.0], vec![190.0, 211.0]]);

    let observed_result = test_data.convolution_v2(&kernel, ConvolutionType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_v2_gpu_valid_3() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![44.0, 54.0, 64.0], vec![84.0, 94.0, 104.0]]);

    let observed_result = test_data.convolution_v2(&kernel, ConvolutionType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn convolution_v2_gpu_valid_4() {
    let test_data = vec![1.0; 64 * 64];
    let test_data = Matrix::new_1d(&test_data, 64, 64);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0]; 3]);

    let observed_result = test_data.convolution_v2(&kernel, ConvolutionType::VALID);

    unsafe { cuda_synchronize() }

    assert!(1 == 1);
  }

  #[test]
  fn nearest_neighbor_gpu_2x_upsample_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
      vec![13.0, 14.0, 15.0, 16.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
      vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
      vec![5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
      vec![5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
      vec![9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0],
      vec![9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0],
      vec![13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0],
      vec![13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0],
    ]);

    let observed_result = test_data.nearest_neighbor_2x_upsample(false);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn nearest_neighbor_gpu_2x_upsample_odd() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
      vec![13.0, 14.0, 15.0, 16.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
      vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
      vec![5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
      vec![5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
      vec![9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0],
      vec![9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0],
      vec![13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0],
    ]);

    let observed_result = test_data.nearest_neighbor_2x_upsample(true);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn pad_matrix_gpu() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![
      vec![0.0, 0.0, 0.0, 0.0, 0.0],
      vec![0.0, 1.0, 2.0, 3.0, 0.0],
      vec![0.0, 4.0, 5.0, 6.0, 0.0],
      vec![0.0, 0.0, 0.0, 0.0, 0.0],
    ]);

    let observed_result = test_data.center_pad(1, 1);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn pad_matrix_gpu_2() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    // Pad by 2
    let expected_result = Matrix::new_2d(&vec![
      vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
      vec![0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
      vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]);

    let observed_result = test_data.center_pad(2, 2);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn pad_matrix_gpu_3() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![
      vec![0.0, 0.0, 0.0, 0.0, 0.0],
      vec![0.0, 0.0, 0.0, 0.0, 0.0],
      vec![0.0, 1.0, 2.0, 3.0, 0.0],
      vec![0.0, 4.0, 5.0, 6.0, 0.0],
      vec![0.0, 0.0, 0.0, 0.0, 0.0],
      vec![0.0, 0.0, 0.0, 0.0, 0.0],
    ]);

    let observed_result = test_data.center_pad(2, 1);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn softmax_gpu() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![0.00235563, 0.00235563, 0.00235563],
      vec![0.04731416, 0.04731416, 0.04731416],
      vec![0.95033021, 0.95033021, 0.95033021],
    ]);

    let observed_result = test_data.softmax();

    assert!(matrix_are_equal(&observed_result, &expected_result, 6));
  }

  #[test]
  fn softmax_gpu_2() {
    // Example that would normally overflow
    let test_data = Matrix::new_2d(&vec![
      vec![600.0, 300.0, 170.0],
      vec![699.0, 360.0, 200.0],
      vec![700.0, 400.0, 100.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![2.71959346e-44, 3.72007598e-44, 9.35762297e-14],
      vec![2.68941421e-01, 4.24835426e-18, 1.00000000e+00],
      vec![7.31058579e-01, 1.00000000e+00, 3.72007598e-44],
    ]);

    let observed_result = test_data.softmax();

    assert!(matrix_are_equal(&observed_result, &expected_result, 6));
  }

  #[test]
  fn crop_gpu() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0, 5.0],
      vec![6.0, 7.0, 8.0, 9.0, 10.0],
      vec![11.0, 12.0, 13.0, 14.0, 15.0],
      vec![16.0, 17.0, 18.0, 19.0, 20.0],
      vec![21.0, 22.0, 23.0, 24.0, 25.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![7.0, 8.0, 9.0],
      vec![12.0, 13.0, 14.0],
      vec![17.0, 18.0, 19.0],
    ]);

    let observed_result = test_data.crop(1, 1, 3, 3);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn crop_gpu_2() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0, 5.0],
      vec![6.0, 7.0, 8.0, 9.0, 10.0],
      vec![11.0, 12.0, 13.0, 14.0, 15.0],
      vec![16.0, 17.0, 18.0, 19.0, 20.0],
      vec![21.0, 22.0, 23.0, 24.0, 25.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![
      vec![7.0, 8.0, 9.0, 10.0],
      vec![12.0, 13.0, 14.0, 15.0],
      vec![17.0, 18.0, 19.0, 20.0],
    ]);

    let observed_result = test_data.crop(1, 1, 3, 4);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn deep_copy_gpu_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0, 5.0],
      vec![6.0, 7.0, 8.0, 9.0, 10.0],
      vec![11.0, 12.0, 13.0, 14.0, 15.0],
      vec![16.0, 17.0, 18.0, 19.0, 20.0],
      vec![21.0, 22.0, 23.0, 24.0, 25.0],
    ]);

    let observed_result = test_data.deep_copy();

    assert_ne!(observed_result.get_id(), test_data.get_id());
    assert!(matrix_are_equal(&observed_result, &test_data, 8));
  }

  #[test]
  fn deep_copy_gpu_2() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0],
      vec![6.0],
      vec![11.0],
      vec![16.0],
      vec![21.0],
    ]);

    let observed_result = test_data.deep_copy();

    assert_ne!(observed_result.get_id(), test_data.get_id());
    assert!(matrix_are_equal(&observed_result, &test_data, 8));
  }

  #[test]
  fn deep_copy_gpu_3() {
    let rows = 257;
    let cols = 1;
    let data = (0..rows * cols).map(|x| x as f32).collect_vec();
    let test_data = Matrix::new_1d(&data, rows, cols);

    let observed_result = test_data.deep_copy();

    assert_ne!(observed_result.get_id(), test_data.get_id());
    assert!(matrix_are_equal(&observed_result, &test_data, 8));
  }

  #[test]
  fn sum_all_matrix_elements_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0, 5.0],
      vec![6.0, 7.0, 8.0, 9.0, 10.0],
      vec![11.0, 12.0, 13.0, 14.0, 15.0],
      vec![16.0, 17.0, 18.0, 19.0, 20.0],
      vec![21.0, 22.0, 23.0, 24.0, 25.0],
    ]);

    let expected_result = Matrix::new_2d(&vec![vec![325.0]]);

    let observed_result = test_data.sum_all_matrix_elements();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn sum_all_matrix_elements_2() {
    let rows = 1;
    let cols = 257;
    let mut sum = 0.0;
    let data = (0..rows)
      .map(|row| {
        (0..cols)
          .map(|col| {
            let val = (row * cols + col) as f32;
            sum += val;
            return val;
          })
          .collect_vec()
      })
      .collect_vec();

    let test_data = Matrix::new_2d(&data);

    let expected_result = Matrix::new_2d(&vec![vec![sum]]);
    let observed_result = test_data.sum_all_matrix_elements();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn sum_all_matrix_elements_3() {
    let rows = 256;
    let cols = 256;
    let data = (0..rows * cols).map(|x| x as f32).collect_vec();
    let mut sum = 0.0;

    let block_size = 256;
    data.chunks(block_size).for_each(|block| {
      let block_sum: f32 = block.iter().sum();
      sum += block_sum;
    });

    let test_data = Matrix::new_1d(&data, rows, cols);

    let expected_result = Matrix::new_2d(&vec![vec![sum]]);
    let observed_result = test_data.sum_all_matrix_elements();

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  fn matrix_are_equal(a: &Matrix, b: &Matrix, precision: usize) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
      println!("Matrices not the same shape!");
      return false;
    }

    let a_data = a.get_data();
    let b_data = b.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b_data[i][j], precision) {
          a.print();
          b.print();
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

    let a_data = a.get_data();
    let b_data = b.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b_data[i][j], precision) {
          a.print();
          b.print();
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
