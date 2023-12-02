#[cfg(test)]
mod tests {
  use crate::{
    correlate_packed, cuda_bindings::*, element_add_packed, flatten_matrix_array, img2col,
    matrix::*, matrix_cpu::MatrixCpu, unflatten_array_to_matrices, PaddingType,
  };
  use itertools::{izip, Itertools};
  use rand::{prelude::Distribution, random};
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
  fn element_add_packed_1() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![2.0, 4.0, 6.0],
        vec![8.0, 10.0, 12.0],
        vec![14.0, 16.0, 18.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![3.0, 4.0, 6.0],
        vec![8.0, 10.0, 12.0],
        vec![14.0, 16.0, 18.5],
      ],);
      16
    ]);

    let mat_1s = (0..32).map(|_| mat_1.deep_copy()).collect_vec();
    let mut observed_result = element_add_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], false);
    observed_result.extend(element_add_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      false,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_add_packed_inplace() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![2.0, 4.0, 6.0],
        vec![8.0, 10.0, 12.0],
        vec![14.0, 16.0, 18.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![3.0, 4.0, 6.0],
        vec![8.0, 10.0, 12.0],
        vec![14.0, 16.0, 18.5],
      ],);
      16
    ]);

    let mat_1s = (0..32).map(|_| mat_1.deep_copy()).collect_vec();
    let mut observed_result = element_add_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], true);
    observed_result.extend(element_add_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      true,
    ));

    izip!(mat_1s, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_add_packed_inplace_repeated_origin() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let expected_result = vec![
      mat_1.element_add(
        &mat_2
          .scalar_multiply(16.0)
          .element_add(&mat_3.scalar_multiply(16.0))
      );
      32
    ];

    let mat_1s = (0..32).map(|_| mat_1.clone()).collect_vec();
    let mut observed_result = element_add_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], true);
    observed_result.extend(element_add_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      true,
    ));

    izip!(mat_1s, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
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
  fn element_subtract_packed_1() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![-1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, -0.5],
      ],);
      16
    ]);

    let mat_1s = (0..32).map(|_| mat_1.deep_copy()).collect_vec();
    let mut observed_result =
      element_subtract_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], false);
    observed_result.extend(element_subtract_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      false,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_subtract_packed_inplace() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![-1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, -0.5],
      ],);
      16
    ]);

    let mat_1s = (0..32).map(|_| mat_1.deep_copy()).collect_vec();
    let mut observed_result =
      element_subtract_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], true);
    observed_result.extend(element_subtract_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      true,
    ));

    izip!(mat_1s, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_subtract_packed_inplace_repeated_origin() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let expected_result = vec![
      mat_1.element_subtract(
        &mat_2
          .scalar_multiply(16.0)
          .element_add(&mat_3.scalar_multiply(16.0))
      );
      32
    ];

    let mat_1s = (0..32).map(|_| mat_1.clone()).collect_vec();
    let mut observed_result =
      element_subtract_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], true);
    observed_result.extend(element_subtract_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      true,
    ));

    izip!(mat_1s, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
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
  fn element_multiply_packed_1() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![1.0, 4.0, 9.0],
        vec![16.0, 25.0, 36.0],
        vec![49.0, 64.0, 81.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![2.0, 4.0, 9.0],
        vec![16.0, 25.0, 36.0],
        vec![49.0, 64.0, 85.5],
      ],);
      16
    ]);

    let mat_1s = (0..32).map(|_| mat_1.deep_copy()).collect_vec();
    let mut observed_result =
      element_multiply_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], false);
    observed_result.extend(element_multiply_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      false,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_multiply_packed_inplace() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![1.0, 4.0, 9.0],
        vec![16.0, 25.0, 36.0],
        vec![49.0, 64.0, 81.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![2.0, 4.0, 9.0],
        vec![16.0, 25.0, 36.0],
        vec![49.0, 64.0, 85.5],
      ],);
      16
    ]);

    let mat_1s = (0..32).map(|_| mat_1.deep_copy()).collect_vec();
    let mut observed_result =
      element_multiply_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], true);
    observed_result.extend(element_multiply_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      true,
    ));

    izip!(mat_1s, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_multiply_packed_inplace_repeated_origin() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let expected_result_single = mat_1.deep_copy();
    for _ in 0..16 {
      expected_result_single.element_multiply_inplace(&mat_2);
    }
    for _ in 16..32 {
      expected_result_single.element_multiply_inplace(&mat_3);
    }

    let expected_result = vec![expected_result_single; 32];

    let mat_1s = (0..32).map(|_| mat_1.clone()).collect_vec();
    let mut observed_result =
      element_multiply_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], true);
    observed_result.extend(element_multiply_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      true,
    ));

    izip!(mat_1s, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_divide() {
    let test_data = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let test_data_2 = Matrix::new_2d(&vec![vec![5.0, 1.0, 3.3], vec![2.0, -1.0, 1000.0]]);

    let expected_result = Matrix::new_2d(&vec![
      vec![0.2, 2.0, 0.9090909090909091],
      vec![2.0, -5.0, 0.006],
    ]);

    let observed_result = test_data.element_divide(&test_data_2);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn element_divide_packed_1() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.947368421],
      ],);
      16
    ]);

    let mat_1s = (0..32).map(|_| mat_1.deep_copy()).collect_vec();
    let mut observed_result =
      element_divide_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], false);
    observed_result.extend(element_divide_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      false,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_divide_packed_inplace() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let mut expected_result = vec![
      Matrix::new_2d(&vec![
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
      ],);
      16
    ];
    expected_result.extend(vec![
      Matrix::new_2d(&vec![
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.947368421],
      ],);
      16
    ]);

    let mat_1s = (0..32).map(|_| mat_1.deep_copy()).collect_vec();
    let mut observed_result =
      element_divide_packed(&mat_1s[0..16].to_vec(), &vec![mat_2; 16], true);
    observed_result.extend(element_divide_packed(
      &mat_1s[16..32].to_vec(),
      &vec![mat_3; 16],
      true,
    ));

    izip!(mat_1s, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn element_divide_packed_inplace_repeated_origin() {
    let mat_1 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_2 = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let mat_3 = Matrix::new_2d(&vec![
      vec![2.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.5],
    ]);

    let expected_result_single = mat_1.deep_copy();
    for _ in 0..2 {
      expected_result_single.element_divide_inplace(&mat_2);
    }
    for _ in 2..4 {
      expected_result_single.element_divide_inplace(&mat_3);
    }

    let expected_result = vec![expected_result_single; 10];

    let mat_1s = (0..10).map(|_| mat_1.clone()).collect_vec();
    let mut observed_result = element_divide_packed(&mat_1s[0..2].to_vec(), &vec![mat_2; 2], true);
    observed_result.extend(element_divide_packed(
      &mat_1s[2..4].to_vec(),
      &vec![mat_3; 2],
      true,
    ));

    izip!(mat_1s, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
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
  fn scalar_multiply_packed_out_of_place() {
    let random_matrices = (0..10)
      .map(|_| Matrix::new_random(0.0, 10.0, 256, 256))
      .collect_vec();

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, random_matrices.len()).get_data()[0];

    let expected_results = izip!(random_matrices.iter(), random_scalars.iter())
      .map(|(mat, &scalar)| mat.scalar_multiply(scalar))
      .collect_vec();

    let observed_result = scalar_multiply_packed(&random_matrices, &random_scalars, false);

    izip!(observed_result, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn scalar_multiply_packed_inplace() {
    let random_matrices = (0..10)
      .map(|_| Matrix::new_random(0.0, 10.0, 256, 256))
      .collect_vec();

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, random_matrices.len()).get_data()[0];

    let expected_results = izip!(random_matrices.iter(), random_scalars.iter())
      .map(|(mat, &scalar)| mat.scalar_multiply(scalar))
      .collect_vec();

    let _ = scalar_multiply_packed(&random_matrices, &random_scalars, true);

    izip!(random_matrices, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn scalar_multiply_packed_inplace_to_origin() {
    let random_matrix = Matrix::new_random(0.0, 10.0, 3, 3);

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, 5).get_data()[0];

    let expected_result = random_matrix.scalar_multiply(random_scalars[0]);
    random_scalars.iter().skip(1).for_each(|&scalar| {
      expected_result.scalar_multiply_inplace(scalar);
    });

    random_matrix.print();

    let _ = scalar_multiply_packed(
      &vec![random_matrix.clone(); random_scalars.len()],
      &random_scalars,
      true,
    );

    assert!(matrix_are_equal(&random_matrix, &expected_result, 1));
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
  fn scalar_divide_packed_out_of_place() {
    let random_matrices = (0..10)
      .map(|_| Matrix::new_random(0.0, 10.0, 256, 256))
      .collect_vec();

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, random_matrices.len()).get_data()[0];

    let expected_results = izip!(random_matrices.iter(), random_scalars.iter())
      .map(|(mat, &scalar)| mat.scalar_divide(scalar))
      .collect_vec();

    let observed_result = scalar_divide_packed(&random_matrices, &random_scalars, false);

    izip!(observed_result, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn scalar_divide_packed_inplace() {
    let random_matrices = (0..10)
      .map(|_| Matrix::new_random(0.0, 10.0, 256, 256))
      .collect_vec();

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, random_matrices.len()).get_data()[0];

    let expected_results = izip!(random_matrices.iter(), random_scalars.iter())
      .map(|(mat, &scalar)| mat.scalar_divide(scalar))
      .collect_vec();

    let _ = scalar_divide_packed(&random_matrices, &random_scalars, true);

    izip!(random_matrices, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn scalar_divide_packed_inplace_to_origin() {
    let random_matrix = Matrix::new_random(0.0, 10.0, 256, 256);

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, 5).get_data()[0];

    let expected_result = random_matrix.scalar_divide(random_scalars[0]);
    random_scalars.iter().skip(1).for_each(|&scalar| {
      expected_result.scalar_divide_inplace(scalar);
    });

    let _ = scalar_divide_packed(
      &vec![random_matrix.clone(); random_scalars.len()],
      &random_scalars,
      true,
    );

    assert!(matrix_are_equal(&random_matrix, &expected_result, 3));
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
  fn scalar_add_packed_out_of_place() {
    let random_matrices = (0..10)
      .map(|_| Matrix::new_random(0.0, 10.0, 256, 256))
      .collect_vec();

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, random_matrices.len()).get_data()[0];

    let expected_results = izip!(random_matrices.iter(), random_scalars.iter())
      .map(|(mat, &scalar)| mat.scalar_add(scalar))
      .collect_vec();

    let observed_result = scalar_add_packed(&random_matrices, &random_scalars, false);

    izip!(observed_result, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn scalar_add_packed_inplace() {
    let random_matrices = (0..10)
      .map(|_| Matrix::new_random(0.0, 10.0, 256, 256))
      .collect_vec();

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, random_matrices.len()).get_data()[0];

    let expected_results = izip!(random_matrices.iter(), random_scalars.iter())
      .map(|(mat, &scalar)| mat.scalar_add(scalar))
      .collect_vec();

    let _ = scalar_add_packed(&random_matrices, &random_scalars, true);

    izip!(random_matrices, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn scalar_add_packed_inplace_to_origin() {
    let random_matrix = Matrix::new_random(0.0, 10.0, 256, 256);

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, 5).get_data()[0];

    let expected_result = random_matrix.scalar_add(random_scalars[0]);
    random_scalars.iter().skip(1).for_each(|&scalar| {
      expected_result.scalar_add_inplace(scalar);
    });

    let _ = scalar_add_packed(
      &vec![random_matrix.clone(); random_scalars.len()],
      &random_scalars,
      true,
    );

    assert!(matrix_are_equal(&random_matrix, &expected_result, 3));
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
  fn scalar_subtract_packed_out_of_place() {
    let random_matrices = (0..10)
      .map(|_| Matrix::new_random(0.0, 10.0, 256, 256))
      .collect_vec();

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, random_matrices.len()).get_data()[0];

    let expected_results = izip!(random_matrices.iter(), random_scalars.iter())
      .map(|(mat, &scalar)| mat.scalar_subtract(scalar))
      .collect_vec();

    let observed_result = scalar_subtract_packed(&random_matrices, &random_scalars, false);

    izip!(observed_result, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn scalar_subtract_packed_inplace() {
    let random_matrices = (0..10)
      .map(|_| Matrix::new_random(0.0, 10.0, 256, 256))
      .collect_vec();

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, random_matrices.len()).get_data()[0];

    let expected_results = izip!(random_matrices.iter(), random_scalars.iter())
      .map(|(mat, &scalar)| mat.scalar_subtract(scalar))
      .collect_vec();

    let _ = scalar_subtract_packed(&random_matrices, &random_scalars, true);

    izip!(random_matrices, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn scalar_subtract_packed_inplace_to_origin() {
    let random_matrix = Matrix::new_random(0.0, 10.0, 256, 256);

    let random_scalars = &Matrix::new_random(0.0, 10.0, 1, 5).get_data()[0];

    let expected_result = random_matrix.scalar_subtract(random_scalars[0]);
    random_scalars.iter().skip(1).for_each(|&scalar| {
      expected_result.scalar_subtract_inplace(scalar);
    });

    let _ = scalar_subtract_packed(
      &vec![random_matrix.clone(); random_scalars.len()],
      &random_scalars,
      true,
    );

    assert!(matrix_are_equal(&random_matrix, &expected_result, 3));
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
  fn element_sqrt() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let test_data_matrix = Matrix::new_2d(&test_data);

    let expected_result = test_data
      .iter()
      .map(|row| row.iter().map(|val| val.sqrt()).collect_vec())
      .collect_vec();

    let expected_result_matrix = Matrix::new_2d(&expected_result);

    let observed_result = test_data_matrix.element_sqrt();

    assert!(matrix_are_equal(
      &observed_result,
      &expected_result_matrix,
      6
    ));
  }

  #[test]
  fn element_sqrt_packed_out_of_place() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let test_data_matrix = Matrix::new_2d(&test_data);

    let expected_result = test_data
      .iter()
      .map(|row| row.iter().map(|val| val.sqrt()).collect_vec())
      .collect_vec();

    let test_data_packed = (0..32).map(|_| test_data_matrix.deep_copy()).collect_vec();
    let expected_result_matrix = vec![Matrix::new_2d(&expected_result); 32];

    let observed_result = element_sqrt_packed(&test_data_packed, false);

    izip!(observed_result, expected_result_matrix)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 6)));
  }

  #[test]
  fn element_sqrt_packed_in_place() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let test_data_matrix = Matrix::new_2d(&test_data);

    let expected_result = test_data
      .iter()
      .map(|row| row.iter().map(|val| val.sqrt()).collect_vec())
      .collect_vec();

    let test_data_packed = (0..32).map(|_| test_data_matrix.deep_copy()).collect_vec();
    let expected_result_matrix = vec![Matrix::new_2d(&expected_result); 32];

    let _ = element_sqrt_packed(&test_data_packed, true);

    izip!(test_data_packed, expected_result_matrix)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 6)));
  }

  #[test]
  fn element_exp() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.0, 6.0]];
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
      4
    ));
  }

  #[test]
  fn element_exp_packed_out_of_place() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.0, 6.0]];
    let test_data_matrix = Matrix::new_2d(&test_data);

    let expected_result = test_data
      .iter()
      .map(|row| row.iter().map(|val| val.exp()).collect_vec())
      .collect_vec();

    let test_data_packed = (0..32).map(|_| test_data_matrix.deep_copy()).collect_vec();
    let expected_result_matrix = vec![Matrix::new_2d(&expected_result); 32];

    let observed_result = element_exp_packed(&test_data_packed, false);

    izip!(observed_result, expected_result_matrix)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 4)));
  }

  #[test]
  fn element_exp_packed_in_place() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.0, 6.0]];
    let test_data_matrix = Matrix::new_2d(&test_data);

    let expected_result = test_data
      .iter()
      .map(|row| row.iter().map(|val| val.exp()).collect_vec())
      .collect_vec();

    let test_data_packed = (0..32).map(|_| test_data_matrix.deep_copy()).collect_vec();
    let expected_result_matrix = vec![Matrix::new_2d(&expected_result); 32];

    let _ = element_exp_packed(&test_data_packed, true);

    izip!(test_data_packed, expected_result_matrix)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 4)));
  }

  #[test]
  fn element_relu() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.0, 6.0]];
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
  fn element_relu_packed_out_of_place() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.0, 6.0]];
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

    let test_data_packed = (0..32).map(|_| test_data_matrix.deep_copy()).collect_vec();
    let expected_result_matrix = vec![Matrix::new_2d(&expected_result); 32];

    let observed_result = element_ReLU_packed(&test_data_packed, false);

    izip!(observed_result, expected_result_matrix)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 6)));
  }

  #[test]
  fn element_relu_packed_in_place() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.0, 6.0]];
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

    let test_data_packed = (0..32).map(|_| test_data_matrix.deep_copy()).collect_vec();
    let expected_result_matrix = vec![Matrix::new_2d(&expected_result); 32];

    let _ = element_ReLU_packed(&test_data_packed, true);

    izip!(test_data_packed, expected_result_matrix)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 6)));
  }

  #[test]
  fn element_relu_prime() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.0, 6.0]];
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
  fn element_relu_prime_packed_out_of_place() {
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.0, 6.0]];
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

    let test_data_packed = (0..32).map(|_| test_data_matrix.deep_copy()).collect_vec();
    let expected_result_matrix = vec![Matrix::new_2d(&expected_result); 32];

    let observed_result = element_ReLU_prime_packed(&test_data_packed, false);

    izip!(observed_result, expected_result_matrix)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 6)));
  }

  #[test]
  fn element_relu_prime_packed_in_place() {
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

    let test_data_packed = (0..32).map(|_| test_data_matrix.deep_copy()).collect_vec();
    let expected_result_matrix = vec![Matrix::new_2d(&expected_result); 32];

    let _ = element_ReLU_prime_packed(&test_data_packed, true);

    izip!(test_data_packed, expected_result_matrix)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 6)));
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
    for _ in 0..100 {
      let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0], vec![4.0, 5.0]]);

      let expected_result = Matrix::new_2d(&vec![vec![5.0]]);

      let (observed_result, _) = test_data.max_pool();

      assert!(matrix_are_equal(&observed_result, &expected_result, 8));
    }
  }

  #[test]
  fn max_pool_gpu_v2() {
    for _ in 0..100 {
      let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0, -100.0], vec![4.0, 5.0, 23.0]]);

      let expected_result = Matrix::new_2d(&vec![vec![5.0, 23.0]]);

      let (observed_result, _) = test_data.max_pool();

      assert!(matrix_are_equal(&observed_result, &expected_result, 8));
    }
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
    for _ in 0..100 {
      let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0, -100.0], vec![4.0, 5.0, 23.0]]);

      let expected_result = Matrix::new_2d(&vec![vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 1.0]]);

      let (_, observed_result) = test_data.max_pool();

      assert!(matrix_are_equal(&observed_result, &expected_result, 8));
    }
  }

  #[test]
  fn max_pool_bitmask_gpu_2() {
    for _ in 0..100 {
      let test_data = Matrix::new_2d(&vec![vec![-5.0, 2.0], vec![4.0, 5.0]]);

      let expected_result = Matrix::new_2d(&vec![vec![0.0, 0.0], vec![0.0, 1.0]]);

      let (_, observed_result) = test_data.max_pool();

      assert!(matrix_are_equal(&observed_result, &expected_result, 8));
    }
  }

  #[test]
  fn max_pool_packed_1() {
    let random_matrices = (0..100)
      .map(|_| Matrix::new_random(0.0, 100.0, 256, 256))
      .collect_vec();

    let expected_results = random_matrices
      .iter()
      .map(|mat| mat.max_pool())
      .collect_vec();

    let expected_results_pooled = expected_results.iter().map(|(mat, _)| mat).collect_vec();
    let expected_results_bitmask = expected_results.iter().map(|(_, mat)| mat).collect_vec();

    let (observed_results_pooled, observed_results_bitmask) = max_pool_packed(&random_matrices);

    izip!(observed_results_pooled, expected_results_pooled)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));

    izip!(observed_results_bitmask, expected_results_bitmask)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn max_pool_packed_2() {
    let random_matrices = (0..100)
      .map(|_| Matrix::new_random(0.0, 100.0, 255, 255))
      .collect_vec();

    let expected_results = random_matrices
      .iter()
      .map(|mat| mat.max_pool())
      .collect_vec();

    let expected_results_pooled = expected_results.iter().map(|(mat, _)| mat).collect_vec();
    let expected_results_bitmask = expected_results.iter().map(|(_, mat)| mat).collect_vec();

    let (observed_results_pooled, observed_results_bitmask) = max_pool_packed(&random_matrices);

    izip!(observed_results_pooled, expected_results_pooled)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));

    izip!(observed_results_bitmask, expected_results_bitmask)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
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
  fn rotate_180_packed_1() {
    let random_matrices = (0..100)
      .map(|_| Matrix::new_random(0.0, 100.0, 3, 3))
      .collect_vec();

    let expected_results = random_matrices
      .iter()
      .map(|mat| mat.rotate_180())
      .collect_vec();

    let observed_results = rotate_180_packed(&random_matrices);

    izip!(observed_results, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn rotate_180_packed_2() {
    let random_matrices = (0..100)
      .map(|_| Matrix::new_random(0.0, 100.0, 2, 3))
      .collect_vec();

    let expected_results = random_matrices
      .iter()
      .map(|mat| mat.rotate_180())
      .collect_vec();

    let observed_results = rotate_180_packed(&random_matrices);

    izip!(observed_results, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
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
  fn correlation() {
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

    let observed_result = test_data.correlate(&kernel);

    assert!(matrix_are_equal_cpu(&observed_result, &expected_result, 8));
  }

  #[test]
  fn correlation_2() {
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

    let observed_result = test_data.correlate(&kernel);

    assert!(matrix_are_equal_cpu(&observed_result, &expected_result, 8));
  }

  #[test]
  fn correlation_gpu_same() {
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

    let observed_result = test_data.correlate(&kernel, PaddingType::SAME);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn correlation_gpu_same_2() {
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

    let observed_result = test_data.correlate(&kernel, PaddingType::SAME);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn packed_correlation_same_1() {
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

    let mut observed_result = correlate_packed(
      &vec![test_data.clone(); 16],
      &vec![kernel; 16],
      PaddingType::SAME,
    );
    observed_result.extend(correlate_packed(
      &vec![test_data.clone(); 16],
      &vec![kernel_2; 16],
      PaddingType::SAME,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn correlation_gpu_valid_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![37.0, 47.0], vec![67.0, 77.0]]);

    let observed_result = test_data.correlate(&kernel, PaddingType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn correlation_gpu_valid_2() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![106.0, 127.0], vec![190.0, 211.0]]);

    let observed_result = test_data.correlate(&kernel, PaddingType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn correlation_gpu_valid_3() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![44.0, 54.0, 64.0], vec![84.0, 94.0, 104.0]]);

    let observed_result = test_data.correlate(&kernel, PaddingType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn packed_correlation_valid_1() {
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

    let mut observed_result = correlate_packed(
      &vec![test_data.clone(); 16],
      &vec![kernel; 16],
      PaddingType::VALID,
    );
    observed_result.extend(correlate_packed(
      &vec![test_data.clone(); 16],
      &vec![kernel_2; 16],
      PaddingType::VALID,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn correlation_gpu_full_1() {
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

    let observed_result = test_data.correlate(&kernel, PaddingType::FULL);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn correlation_gpu_full_2() {
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

    let observed_result = test_data.correlate(&kernel, PaddingType::FULL);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn packed_correlation_full_1() {
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

    let mut observed_result = correlate_packed(
      &vec![test_data.clone(); 16],
      &vec![kernel; 16],
      PaddingType::FULL,
    );
    observed_result.extend(correlate_packed(
      &vec![test_data.clone(); 16],
      &vec![kernel_2; 16],
      PaddingType::FULL,
    ));

    izip!(observed_result, expected_result).for_each(|(observed, expected)| {
      assert!(matrix_are_equal(&observed, &expected, 8));
    });
  }

  #[test]
  fn correlation_cpu_gpu_agreement() {
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

    let mat_gpu = Matrix::new_2d(&data).correlate(&Matrix::new_2d(kernel), PaddingType::SAME);
    let mat_cpu = MatrixCpu::new_2d(&data).correlate(&MatrixCpu::new_2d(kernel));

    assert!(matrix_are_equal_gpu_cpu(&mat_gpu, &mat_cpu, 2));
  }

  #[test]
  fn convolution_gpu_same() {
    // Generate random data of different sizes
    for i in 1..25 {
      for j in 1..25 {
        let random_input = Matrix::new_random(0.0, 100.0, i, j);

        // Grab closes odd number to j
        let kernel_size = j + 1 - (j % 2);

        let random_kernel = Matrix::new_random(0.0, 100.0, kernel_size, kernel_size);

        let expected_result =
          random_input.correlate(&random_kernel.rotate_180(), PaddingType::SAME);
        let observed_result = random_input.convolve(&random_kernel, PaddingType::SAME);

        assert!(matrix_are_equal(&observed_result, &expected_result, 8));
      }
    }
  }

  #[test]
  fn packed_convolution_gpu_same() {
    // Generate random data of different sizes
    for i in 2..25 {
      for j in 2..25 {
        let random_inputs = (0..i * j)
          .map(|_| Matrix::new_random(0.0, 100.0, i, j))
          .collect_vec();

        let kernel_size = j + 1 - (j % 2);
        let random_kernels = (0..i * j)
          .map(|_| Matrix::new_random(0.0, 100.0, kernel_size, kernel_size))
          .collect_vec();

        let rotated_kernels = rotate_180_packed(&random_kernels);
        let expected_results =
          correlate_packed(&random_inputs, &rotated_kernels, PaddingType::SAME);
        let observed_results = convolve_packed(&random_inputs, &random_kernels, PaddingType::SAME);

        izip!(observed_results, expected_results).for_each(|(observed, expected)| {
          assert!(matrix_are_equal(&observed, &expected, 8));
        });
      }
    }
  }

  #[test]
  fn convolution_gpu_valid() {
    // Generate random data of different sizes
    for i in 2..25 {
      for j in 2..25 {
        let random_input = Matrix::new_random(0.0, 100.0, i, j);

        let random_kernel = Matrix::new_random(0.0, 100.0, i / 2, j / 2);

        let expected_result =
          random_input.correlate(&random_kernel.rotate_180(), PaddingType::VALID);
        let observed_result = random_input.convolve(&random_kernel, PaddingType::VALID);

        assert!(matrix_are_equal(&observed_result, &expected_result, 8));
      }
    }
  }

  #[test]
  fn packed_convolution_gpu_valid() {
    // Generate random data of different sizes
    for i in 2..25 {
      for j in 2..25 {
        let random_inputs = (0..i * j)
          .map(|_| Matrix::new_random(0.0, 100.0, i, j))
          .collect_vec();

        let random_kernels = (0..i * j)
          .map(|_| Matrix::new_random(0.0, 100.0, i / 2, j / 2))
          .collect_vec();

        let rotated_kernels = rotate_180_packed(&random_kernels);
        let expected_results =
          correlate_packed(&random_inputs, &rotated_kernels, PaddingType::VALID);
        let observed_results = convolve_packed(&random_inputs, &random_kernels, PaddingType::VALID);

        izip!(observed_results, expected_results).for_each(|(observed, expected)| {
          assert!(matrix_are_equal(&observed, &expected, 8));
        });
      }
    }
  }

  #[test]
  fn convolution_gpu_full() {
    // Generate random data of different sizes
    for i in 2..25 {
      for j in 2..25 {
        let random_input = Matrix::new_random(0.0, 100.0, i, j);

        let random_kernel = Matrix::new_random(0.0, 100.0, i / 2, j / 2);

        let expected_result =
          random_input.correlate(&random_kernel.rotate_180(), PaddingType::FULL);
        let observed_result = random_input.convolve(&random_kernel, PaddingType::FULL);

        assert!(matrix_are_equal(&observed_result, &expected_result, 8));
      }
    }
  }

  #[test]
  fn packed_convolution_gpu_full() {
    // Generate random data of different sizes
    for i in 2..25 {
      for j in 2..25 {
        let random_inputs = (0..i * j)
          .map(|_| Matrix::new_random(0.0, 100.0, i, j))
          .collect_vec();

        let random_kernels = (0..i * j)
          .map(|_| Matrix::new_random(0.0, 100.0, i / 2, j / 2))
          .collect_vec();

        let rotated_kernels = rotate_180_packed(&random_kernels);
        let expected_results =
          correlate_packed(&random_inputs, &rotated_kernels, PaddingType::FULL);
        let observed_results = convolve_packed(&random_inputs, &random_kernels, PaddingType::FULL);

        izip!(observed_results, expected_results).for_each(|(observed, expected)| {
          assert!(matrix_are_equal(&observed, &expected, 8));
        });
      }
    }
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
  fn correlation_v2_gpu_valid_1() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![37.0, 47.0], vec![67.0, 77.0]]);

    let observed_result = test_data.correlate_v2(&kernel, PaddingType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn correlation_v2_gpu_valid_2() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![106.0, 127.0], vec![190.0, 211.0]]);

    let observed_result = test_data.correlate_v2(&kernel, PaddingType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
  }

  #[test]
  fn correlation_v2_gpu_valid_3() {
    let test_data = Matrix::new_2d(&vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 10.0, 11.0, 12.0],
    ]);

    let kernel = Matrix::new_2d(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let expected_result = Matrix::new_2d(&vec![vec![44.0, 54.0, 64.0], vec![84.0, 94.0, 104.0]]);

    let observed_result = test_data.correlate_v2(&kernel, PaddingType::VALID);

    assert!(matrix_are_equal(&observed_result, &expected_result, 8));
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
  fn nearest_neighbor_2x_upsample_packed_1() {
    let random_matrices = (0..100)
      .map(|_| Matrix::new_random(0.0, 100.0, 256, 256))
      .collect_vec();

    let expected_results = random_matrices
      .iter()
      .map(|mat| mat.nearest_neighbor_2x_upsample(false))
      .collect_vec();

    let observed_results = nearest_neighbor_2x_upsample_packed(&random_matrices, false);

    izip!(observed_results, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
  }

  #[test]
  fn nearest_neighbor_2x_upsample_packed_2() {
    let random_matrices = (0..100)
      .map(|_| Matrix::new_random(0.0, 100.0, 256, 256))
      .collect_vec();

    let expected_results = random_matrices
      .iter()
      .map(|mat| mat.nearest_neighbor_2x_upsample(true))
      .collect_vec();

    let observed_results = nearest_neighbor_2x_upsample_packed(&random_matrices, true);

    izip!(observed_results, expected_results)
      .for_each(|(observed, expected)| assert!(matrix_are_equal(&observed, &expected, 8)));
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
