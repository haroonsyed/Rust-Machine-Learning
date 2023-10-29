use matrix_lib::Matrix;
use std::{
  fs::File,
  io::{prelude::*, BufReader},
  path::Path,
};

pub fn get_initial_fc_weights() -> Vec<Matrix> {
  // Load comma-separated f32 (with e-n) data from the cnn_test_initial_fc_weights.txt file
  let path = Path::new("tests/cnn_tests_util/cnn_test_initial_fc_weights.txt");
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

  // PRINT THE FIRST 20 ELEMENTS OF THE DATA
  println!("data: {:?}", &data[0..20]);

  // Assuming your Matrix::new_1d and transpose functions exist
  return vec![Matrix::new_1d(&data, 5408, 10).transpose()];
}
