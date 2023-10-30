use tensor_lib::Matrix;

pub fn get_expected_softmax_gradient() -> Matrix {
  let data = vec![
    0.0964437003840343,
    0.10098823562640104,
    0.09971604128615202,
    0.0990740267393079,
    0.09933910242993932,
    0.09800496874152793,
    0.10417807913962442,
    -0.899331777332868,
    0.10071505742233165,
    0.10087256556354943,
  ];

  return Matrix::new_1d(&data, 10, 1);
}
