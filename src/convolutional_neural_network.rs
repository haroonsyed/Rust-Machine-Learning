// use crate::py_util::py_print;
use itertools::{izip, Itertools};
use matrix_lib::Matrix;
use pyo3::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution};
use statrs::distribution::Normal;

use crate::basic_neural_network::BasicNeuralNetworkRust;

#[pyclass]
pub struct ConvolutionalNeuralNetwork {
  pub network: ConvolutionalNeuralNetworkRust,
}

#[pymethods]
impl ConvolutionalNeuralNetwork {
  #[new]
  fn new(
    features_train: Vec<Vec<f32>>,
    input_labels: Vec<f32>,
    num_classifications: usize,
    filters_per_conv_layer: Vec<usize>,
    filter_dimension: usize, // Must be odd
    learning_rate: f32,
    batch_size: usize, // Set to 0 to use stochastic GD
    num_iterations: usize,
  ) -> Self {
    let network = ConvolutionalNeuralNetworkRust::new(
      features_train,
      input_labels,
      num_classifications,
      filters_per_conv_layer,
      filter_dimension,
      learning_rate,
      batch_size,
      num_iterations,
    );

    // Cleanup and return
    return Self { network };
  }

  fn classify(&self, features_test: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let classifications = self.network.classify(&features_test);

    return Ok(classifications);
  }
}

pub struct ConvolutionalNeuralNetworkRust {
  pub conv_layers: Vec<Vec<Matrix>>,
  pub biases: Vec<Matrix>,
  pub fully_connected_layer: BasicNeuralNetworkRust,
}

impl ConvolutionalNeuralNetworkRust {
  pub fn new(
    features_train: Vec<Vec<f32>>,
    input_labels: Vec<f32>,
    num_classifications: usize,
    filters_per_conv_layer: Vec<usize>,
    filter_dimension: usize, // Must be odd
    learning_rate: f32,
    batch_size: usize, // Set to 0 to use stochastic GD
    num_iterations: usize,
  ) -> Self {
    // Gather data on dimensions for matrices
    let num_observations = features_train.len();

    // Init the filters and biases
    // Random seed for weights
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 0.68).unwrap();

    // Create the fully connected layer

    // Create the CNN object

    // Train the network

    // Cleanup and return
    return network;
  }

  pub fn classify(&self, features_test: &Vec<Vec<f32>>) -> Vec<f32> {
    let num_observations = features_test.len();

    // Feed forward through network
    let observations = Matrix::new_2d(&features_test).transpose();

    // Classify

    return classifications;
  }

  pub fn mini_batch(
    row_based_observations: &Vec<Vec<f32>>,
    labels: &Vec<f32>,
    batch_size: usize,
  ) -> (Matrix, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let row_dist = Uniform::new(0, row_based_observations.len());

    let mut mini_batch_data = Vec::new();
    let mut mini_batch_labels = vec![0.0; batch_size];
    for i in 0..batch_size {
      let row_index = row_dist.sample(&mut rng);
      let sampled_row = row_based_observations[row_index].to_vec();
      mini_batch_data.push(sampled_row);
      mini_batch_labels[i] = labels[row_index];
    }

    let mini_batch = Matrix::new_2d(&mini_batch_data).transpose();
    return (mini_batch, mini_batch_labels);
  }

  pub fn train(
    &mut self,
    observations: Vec<Vec<f32>>,
    conv_layer_outputs: &mut Vec<Vec<Matrix>>,
    labels: Vec<f32>,
    learning_rate: f32,
    num_iterations: usize,
    batch_size: usize,
  ) {
    let observations_matrix = Matrix::new_2d(&observations).transpose();

    // For now we will make the number of iterations a constant
    for i in 0..num_iterations {
      // Batch
      let batch_data: (Matrix, Vec<f32>) =
        BasicNeuralNetworkRust::mini_batch(&observations, &labels, batch_size);

      let batch = if batch_size == 0 {
        &observations_matrix
      } else {
        &batch_data.0
      };
      let batch_labels = if batch_size == 0 {
        &labels
      } else {
        &batch_data.1
      };

      // Feed forward
      self.feed_forward(batch, conv_layer_outputs);

      // Calculate error from feed forward step

      // Backpropogate hidden
    }
  }

  pub fn feed_forward(&self, observations: &Matrix, conv_layer_outputs: &mut Vec<Vec<Matrix>>) {}

  pub fn backpropogation_hidden_layer(
    &mut self,
    observations: &Matrix,
    conv_layer_outputs: &Vec<Vec<Matrix>>,
    next_layer_error: &Vec<Matrix>,
    learning_rate: f32,
    layer: usize,
  ) {
    // Used for yin
    let prev_layer_outputs = if layer == 0 {
      observations
    } else {
      &neuron_outputs[layer - 1]
    };

    // Possibly need a normalization factor?
    //let normalization_factor = 1.0 / prev_layer_outputs.columns as f32;

    // RelU
    let activation_prime_x = neuron_outputs[layer].element_ReLU_prime();

    // Calculate Error to backprop

    // Calculate error in bias

    // Calculate error in erights
  }
}
