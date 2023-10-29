use matrix_lib::Matrix;

pub fn get_expected_softmax_output() -> Matrix {
  let data = vec![
    0.14862543, 0.11053791, 0.09936849, 0.10802596, 0.12074736, 0.11478395, 0.14319846, 0.01910333,
    0.09117025, 0.04443886,
  ];

  return Matrix::new_1d(&data, 10, 1);
}
