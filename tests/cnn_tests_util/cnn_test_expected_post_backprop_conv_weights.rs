use matrix_lib::Matrix;
use std::{
  fs::File,
  io::{prelude::*, BufReader},
  path::Path,
};

pub fn get_expected_post_backprop_conv_weights() -> Vec<Matrix> {
  let weights_1 = vec![
    0.5488370120567405,
    0.715218361029481,
    0.6027914334472829,
    0.5449125609501411,
    0.4236847100375578,
    0.6459233429367287,
    0.43761568761127206,
    0.8917997917122611,
    0.9636896190804105,
  ];

  let weights_2 = vec![
    0.4170480042575679,
    0.7203563911445954,
    0.00014483986272093876,
    0.30236425618949103,
    0.14678808271421948,
    0.09237011152281534,
    0.1862903480817416,
    0.34559091652002355,
    0.39679514062506893,
  ];
  let weights_3 = vec![
    0.436023547378719,
    0.025958200914829505,
    0.5496949152092868,
    0.4353571083412841,
    0.4204032974429932,
    0.33037043600164234,
    0.20468069739512432,
    0.6193048363481451,
    0.29968721953068395,
  ];
  let weights_4 = vec![
    0.5508277358697947,
    0.708181414851173,
    0.290939821238957,
    0.5108629413015036,
    0.8929848902857769,
    0.8963300030225684,
    0.12561578975398852,
    0.20727803668980105,
    0.05149818369219025,
  ];
  let weights_5 = vec![
    0.9670588535938285,
    0.5472629071612369,
    0.9727172657841743,
    0.7148485190237385,
    0.6977635106197763,
    0.21612493459411714,
    0.9763029140387963,
    0.006263333270331322,
    0.25301183594415916,
  ];
  let weights_6 = vec![
    0.2220221527524667,
    0.8707598193304105,
    0.2067498822820559,
    0.9186393659620669,
    0.4884412070816479,
    0.611774699901777,
    0.7659330314743011,
    0.5184460558017565,
    0.296828750409128,
  ];
  let weights_7 = vec![
    0.8928882309708199,
    0.3320047083210896,
    0.8212602750279735,
    0.04172355699741085,
    0.10768499846072323,
    0.5950844231079768,
    0.5298432938498058,
    0.41883567814653666,
    0.3354383265570678,
  ];
  let weights_8 = vec![
    0.07633457220776264,
    0.7799431258722008,
    0.43844060640800925,
    0.723490675201814,
    0.9780170699697597,
    0.5385252540813221,
    0.5011467022304533,
    0.07207825448627715,
    0.2684662884780098,
  ];

  let mut weights = Vec::new();
  weights.push(Matrix::new_1d(&weights_1, 3, 3));
  weights.push(Matrix::new_1d(&weights_2, 3, 3));
  weights.push(Matrix::new_1d(&weights_3, 3, 3));
  weights.push(Matrix::new_1d(&weights_4, 3, 3));
  weights.push(Matrix::new_1d(&weights_5, 3, 3));
  weights.push(Matrix::new_1d(&weights_6, 3, 3));
  weights.push(Matrix::new_1d(&weights_7, 3, 3));
  weights.push(Matrix::new_1d(&weights_8, 3, 3));

  return weights;
}

pub fn get_expected_post_backprop_conv_weights_i(i: usize) -> Vec<Matrix> {
  let path = Path::new("tests/cnn_tests_util/cnn_test_post_backprop_conv_weights_i.txt");
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

  let mut weights = Vec::new();

  // Add data to weights
  for i in 0..8 {
    weights.push(Matrix::new_1d(&data[i * 9..i * 9 + 9].to_vec(), 3, 3));
  }

  return weights;
}