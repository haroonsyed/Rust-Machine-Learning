mod cnn_tests_util;
use cnn_tests_util::*;

#[cfg(test)]
mod simplified_cnn_tests {
  use crate::cnn_tests_util::*;
  use itertools::{izip, Itertools};
  use matrix_lib::{flatten_matrix_array, Matrix};
  use rust_machine_learning::{
    basic_neural_network::BasicNeuralNetworkRust,
    simplified_convolutional_neural_network::SimplifiedConvolutionalNeuralNetworkRust,
  };
  // ALL EXPECTED RESULTS ARE FROM A SMALL PYTHON CNN WITH A SIMILAR ARCHITECTURE I MADE THAT IK WORKS.

  #[test]
  fn test_feed_forward() {
    // We will use 28x28 input images
    let observations_matrices = get_mnist_test_matrix();
    let num_classifications = 10;
    let input_width = 28;
    let input_height = 28;
    let input_depth = 1;
    let filters_per_conv_layer = vec![8];
    let filter_dimension = 3;

    let mut cnn = SimplifiedConvolutionalNeuralNetworkRust::new(
      num_classifications,
      input_width,
      input_height,
      input_depth,
      filters_per_conv_layer,
      filter_dimension,
    );

    // Override the weights with known values, remember it is in format layer->filter->channel
    cnn.conv_layers = get_conv_layer();

    let filter_outputs = cnn.feed_forward(&observations_matrices);

    let expected_outputs_matrices = get_expected_feed_forward_outputs();

    for (observed, expected) in izip!(
      filter_outputs[0][0].iter(),
      expected_outputs_matrices.iter()
    ) {
      println!("Testing output matrix:");
      assert!(matrix_are_equal(&observed, expected, 6));
    }
  }

  #[test]
  fn test_flatten() {
    let filter_outputs = get_expected_feed_forward_outputs();

    // Linearize the final output
    let flattened_sample_outputs = flatten_matrix_array(&filter_outputs);

    let expected_flattened_sample_outputs = get_expected_flattened_outputs();

    assert!(matrix_are_equal(
      &flattened_sample_outputs,
      &expected_flattened_sample_outputs,
      6
    ));
  }

  #[test]
  fn test_fc_feed_forward() {
    // We will use 28x28 input images
    let num_classifications = 10;
    let input_width = 28;
    let input_height = 28;
    let input_depth = 1;
    let filters_per_conv_layer = vec![8];
    let filter_dimension = 3;

    let mut cnn = SimplifiedConvolutionalNeuralNetworkRust::new(
      num_classifications,
      input_width,
      input_height,
      input_depth,
      filters_per_conv_layer,
      filter_dimension,
    );

    // Make the CNN FC have predictable init random weights
    cnn.fully_connected_layer.weights = get_initial_fc_weights();

    // Set the CNN neuron outputs
    cnn.fully_connected_layer.neuron_outputs = cnn
      .fully_connected_layer
      .non_input_layer_sizes
      .iter()
      .map(|&layer_size| Matrix::no_fill(layer_size, 1))
      .collect();

    let flattened_outputs = get_expected_flattened_outputs().transpose(); // Change to have observations as columns

    // Feed forward through FC
    cnn.fully_connected_layer.feed_forward(&flattened_outputs);

    let fc_output = &cnn.fully_connected_layer.neuron_outputs[0];
    let expected_fc_output = get_expected_fc_output();

    assert!(matrix_are_equal(&expected_fc_output, &fc_output, 2));
  }

  #[test]
  fn test_softmax() {
    // We will use 28x28 input images
    let num_classifications = 10;
    let input_width = 28;
    let input_height = 28;
    let input_depth = 1;
    let filters_per_conv_layer = vec![8];
    let filter_dimension = 3;

    let mut cnn = SimplifiedConvolutionalNeuralNetworkRust::new(
      num_classifications,
      input_width,
      input_height,
      input_depth,
      filters_per_conv_layer,
      filter_dimension,
    );

    // Make the CNN FC have predictable init random weights
    cnn.fully_connected_layer.weights = get_initial_fc_weights();

    // Set the CNN neuron outputs
    cnn.fully_connected_layer.neuron_outputs = cnn
      .fully_connected_layer
      .non_input_layer_sizes
      .iter()
      .map(|&layer_size| Matrix::no_fill(layer_size, 1))
      .collect();

    let flattened_outputs = get_expected_flattened_outputs().transpose(); // Change to have observations as columns

    // Feed forward through FC
    cnn.fully_connected_layer.feed_forward(&flattened_outputs);

    // Get the predicted probabilities
    let predicted_probabilities =
      BasicNeuralNetworkRust::softmax(&cnn.fully_connected_layer.neuron_outputs);

    let expected_prediceted_probabilities = get_expected_softmax_output();

    assert!(matrix_are_equal(
      &expected_prediceted_probabilities,
      &predicted_probabilities,
      2
    ));
  }

  fn matrix_are_equal(a: &Matrix, b: &Matrix, precision: usize) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
      println!("Matrices are not the same shape!");
      a.print_shape();
      b.print_shape();
      return false;
    }

    a.print();
    b.print();

    let a_data = a.get_data();
    let b_data = b.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b_data[i][j], precision) {
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

  fn get_learning_rate() -> f32 {
    return 1e-4;
  }

  fn get_mnist_test_label() -> f32 {
    return 6.0;
  }
}
