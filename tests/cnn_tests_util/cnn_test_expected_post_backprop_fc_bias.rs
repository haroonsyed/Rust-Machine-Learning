use matrix_lib::Matrix;
use std::{
  fs::File,
  io::{prelude::*, BufReader},
  path::Path,
};

pub fn get_expected_post_backprop_fc_bias() -> Matrix {
  let data = vec![
    -9.64437003840343e-05,
    -0.00010098823562640104,
    -9.971604128615202e-05,
    -9.90740267393079e-05,
    -9.933910242993932e-05,
    -9.800496874152793e-05,
    -0.00010417807913962442,
    0.000899331777332868,
    -0.00010071505742233166,
    -0.00010087256556354942,
  ];

  return Matrix::new_1d(&data, 10, 1);
}

pub fn get_expected_post_backprop_fc_bias_i(i: usize) -> Matrix {
  // Change to load from a file, where the line number specifies the index
  // tests/cnn_tests_util/cnn_test_post_backprop_fc_bias_i.txt
  let path = Path::new("tests/cnn_tests_util/cnn_test_post_backprop_fc_bias_i.txt");
  let file = File::open(path).expect("Failed to open file");
  let reader = BufReader::new(file);

  let mut data = Vec::<f32>::new();

  // Read the line at index i
  let mut line_count = 0;
  for line in reader.lines() {
    let line = line.expect("Failed to read line");
    if line_count == i {
      for num_str in line.trim().split(',') {
        if let Ok(num) = num_str.parse::<f64>() {
          data.push(num as f32); // Convert f64 to f32
        } else {
          eprintln!("Failed to parse a number: {}", num_str);
        }
      }
      break;
    }
    line_count += 1;
  }

  return Matrix::new_1d(&data, 10, 1);
}
