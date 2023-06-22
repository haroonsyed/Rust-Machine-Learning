#[cfg(test)]
mod matrix_tests {

  use Rust_Machine_Learning::matrix_lib::Matrix;

  #[test]
  fn element_add() {
    let test_data = Matrix {
      data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    };
    let test_data_2 = Matrix {
      data: vec![vec![5.0, 1.0, 3.3], vec![0.0, -1.0, 1000.0]],
    };

    let expected_result = Matrix {
      data: vec![vec![6.0, 3.0, 6.3], vec![4.0, 4.0, 1006.0]],
    };

    let observed_result = test_data.element_add(&test_data_2);

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  #[test]
  fn matrix_multiply() {
    let test_data = Matrix {
      data: vec![vec![2.0, 4.0], vec![1.0, 3.0]],
    };
    let test_data_2 = Matrix {
      data: vec![vec![3.0, 1.0, 5.0], vec![-2.0, 1.0, 3.0]],
    };

    let expected_result = Matrix {
      data: vec![vec![-2.0, 6.0, 22.0], vec![-3.0, 4.0, 14.0]],
    };

    let observed_result = test_data.matrix_multiply(&test_data_2);

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  #[test]
  fn transpose() {
    let test_data = Matrix {
      data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    };
    let expected_result = Matrix {
      data: vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]],
    };

    let observed_result = test_data.transpose();

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  #[test]
  fn element_apply() {
    let test_data = Matrix {
      data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    };
    let expected_result = Matrix {
      data: vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]],
    };

    let observed_result = test_data.element_apply(&|x| x - 1.0);

    assert!(matrix_are_equal(observed_result, expected_result));
  }

  fn matrix_are_equal(a: Matrix, b: Matrix) -> bool {
    if a.get_rows() != b.get_rows() || a.get_columns() != b.get_columns() {
      return false;
    }

    for i in 0..a.get_rows() {
      for j in 0..a.get_columns() {
        if a.data[i][j] != b.data[i][j] {
          return false;
        }
      }
    }

    return true;
  }
}
