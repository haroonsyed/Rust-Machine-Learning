use matrix_lib::Matrix;
use std::{
  fs::File,
  io::{prelude::*, BufReader},
  path::Path,
};

pub fn get_expected_post_backprop_fc_weight() -> Vec<Matrix> {
  // Load comma-separated f32 (with e-n) data from the cnn_test_initial_fc_weights.txt file
  let path = Path::new("tests/cnn_tests_util/cnn_test_post_backprop_fc_weight.txt");
  let file = File::open(path).expect("Failed to open file");
  let reader = BufReader::new(file);

  let mut data = Vec::<f32>::new();

  for line in reader.lines() {
    let line = line.expect("Failed to read line");
    for num_str in line.trim().split(',') {
      if let Ok(num) = num_str.parse::<f64>() {
        data.push(num as f32); // Convert f64 to f32
      } else {
        eprintln!("Failed to parse a number: {}", num_str);
      }
    }
  }

  // Assuming your Matrix::new_1d and transpose functions exist
  return vec![Matrix::new_1d(&data, 5408, 10).transpose()];
}

pub fn get_expected_post_backprop_fc_weight_i(i: usize) -> Vec<Matrix> {
  // Load comma-separated f32 (with e-n) data from the cnn_test_initial_fc_weights.txt file
  let path = Path::new("tests/cnn_tests_util/cnn_test_post_backprop_fc_weight_i.txt");
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

  // Assuming your Matrix::new_1d and transpose functions exist
  return vec![Matrix::new_1d(&data, 5408, 10).transpose()];
}
