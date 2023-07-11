#[cfg(test)]
mod matrix_tests {
  use matrix_lib::bindings::*;
  use matrix_lib::Matrix;
  use std::ffi::c_double;

  #[test]
  fn element_add() {
    let test_data = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let test_data_2 = Matrix::new_2d(vec![vec![5.0, 1.0, 3.3], vec![0.0, -1.0, 1000.0]]);

    let expected_result = Matrix::new_2d(vec![vec![6.0, 3.0, 6.3], vec![4.0, 4.0, 1006.0]]);

    let observed_result = test_data.element_add(&test_data_2);

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  #[test]
  fn matrix_multiply() {
    let test_data = Matrix::new_2d(vec![vec![2.0, 4.0], vec![1.0, 3.0]]);
    let test_data_2 = Matrix::new_2d(vec![vec![3.0, 1.0, 5.0], vec![-2.0, 1.0, 3.0]]);

    let expected_result = Matrix::new_2d(vec![vec![-2.0, 6.0, 22.0], vec![-3.0, 4.0, 14.0]]);

    let observed_result = test_data.matrix_multiply(&test_data_2);

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  #[test]
  fn transpose() {
    let test_data = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);

    let observed_result = test_data.transpose();

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  #[test]
  fn element_apply() {
    let test_data = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]);

    let observed_result = test_data.element_apply(&|x| x - 1.0);

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  #[test]
  fn add_vector_to_columns() {
    let matrix = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let vector = Matrix::new_2d(vec![vec![1.0], vec![-1.0]]);

    let expected_result = Matrix::new_2d(vec![vec![2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0]]);

    let observed_result = matrix.add_vector_to_columns(&vector);

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  #[test]
  fn sum_row() {
    let matrix = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = vec![6.0, 15.0];

    let observed_result = matrix.sum_rows();

    assert_eq!(observed_result, expected_result);
  }

  #[test]
  fn sum_column() {
    let matrix = Matrix::new_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = vec![5.0, 7.0, 9.0];

    let observed_result = matrix.sum_columns();

    assert_eq!(observed_result, expected_result);
  }

  fn matrix_are_equal(a: Matrix, b: Matrix) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
      return false;
    }

    for i in 0..a.rows {
      for j in 0..a.columns {
        if a[i][j] != b[i][j] {
          return false;
        }
      }
    }

    return true;
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
