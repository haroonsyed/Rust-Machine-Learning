use itertools::Itertools;
use matrix_lib::Matrix;
use pyo3::prelude::*;
use rand::prelude::Distribution;
use statrs::distribution::Normal;

use crate::{basic_neural_network::BasicNeuralNetworkRust, image_util::ImageBatchLoaderRust};

#[pyclass]
pub struct ConvolutionalNeuralNetwork {
  pub network: ConvolutionalNeuralNetworkRust,
  batch_loader: Option<ImageBatchLoaderRust>,
}

#[pymethods]
impl ConvolutionalNeuralNetwork {
  #[new]
  fn new(
    num_classifications: usize,
    max_pool_stride: usize,
    input_width: usize,
    input_height: usize,
    filters_per_conv_layer: Vec<usize>,
    filter_dimension: usize, // Must be odd
  ) -> Self {
    let network = ConvolutionalNeuralNetworkRust::new(
      num_classifications,
      max_pool_stride,
      input_width,
      input_height,
      filters_per_conv_layer,
      filter_dimension,
    );

    // Cleanup and return
    return Self {
      network,
      batch_loader: Option::None,
    };
  }

  fn set_image_loader(&mut self, parent_folder: String, sample_width: usize, sample_height: usize) {
    self.batch_loader = Option::Some(ImageBatchLoaderRust::new(
      parent_folder,
      sample_width,
      sample_height,
    ));
  }

  fn train_using_image_loader(
    &mut self,
    learning_rate: f32,
    batch_size: usize,
    num_iterations: usize,
  ) {
    if let Some(batch_loader) = self.batch_loader.as_ref() {
      let (observations, labels) = batch_loader.batch_sample(batch_size);
      self
        .network
        .train(&observations, &labels, learning_rate, num_iterations);
    }
  }

  fn train_raw_data(
    &mut self,
    observations: Vec<Vec<Vec<f32>>>,
    labels: Vec<f32>,
    learning_rate: f32,
    num_iterations: usize,
  ) {
    self
      .network
      .train(&observations, &labels, learning_rate, num_iterations);
  }

  fn classify(&self, features_test: Vec<Vec<Vec<f32>>>) -> PyResult<Vec<f32>> {
    let classifications = self.network.classify(&features_test);

    return Ok(classifications);
  }
}

pub struct ConvolutionalNeuralNetworkRust {
  pub max_pool_stride: usize,
  pub conv_layers: Vec<Vec<Matrix>>,
  pub biases: Vec<Matrix>,
  pub fully_connected_layer: BasicNeuralNetworkRust,
}

impl ConvolutionalNeuralNetworkRust {
  pub fn new(
    num_classifications: usize,
    max_pool_stride: usize,
    input_width: usize,
    input_height: usize,
    filters_per_conv_layer: Vec<usize>,
    filter_dimension: usize, // Must be odd
  ) -> Self {
    // Init the filters and biases
    // Random seed for weights
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 0.68).unwrap();

    let num_layers = filters_per_conv_layer.len();

    let biases = (0..num_layers)
      .map(|_| Matrix::zeros(filter_dimension, filter_dimension))
      .collect_vec();

    let conv_layers = filters_per_conv_layer
      .iter()
      .map(|layer_size| {
        (0..*layer_size)
          .map(|_| {
            Matrix::new_2d(
              &(0..filter_dimension)
                .map(|_| {
                  (0..filter_dimension)
                    .map(|_| range.sample(&mut rng) as f32)
                    .collect_vec()
                })
                .collect_vec(),
            )
          })
          .collect_vec()
      })
      .collect_vec();

    // Create the fully connected layer
    // hidden layer size is 0, it is just a translation layer of linearized max pool to classification
    // num_features will be final conv layer linearized
    let num_max_pool = num_layers / max_pool_stride;
    let output_width = input_width / (2 << num_max_pool);
    let output_height = input_height / (2 << num_max_pool);
    let num_features = output_width * output_height;
    let fully_connected_layer =
      BasicNeuralNetworkRust::new(Vec::new(), num_features, num_classifications);

    // Create the CNN object
    let network = ConvolutionalNeuralNetworkRust {
      max_pool_stride,
      conv_layers,
      biases,
      fully_connected_layer,
    };
    return network;
  }

  pub fn classify(&self, features_test: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    let num_observations = features_test.len();

    // Feed forward through network
    // let observations = Matrix::new_2d(&features_test).transpose();

    // Classify

    let classifications = Vec::new();
    return classifications;
  }

  pub fn train(
    &mut self,
    observations: &Vec<Vec<Vec<f32>>>,
    labels: &Vec<f32>,
    learning_rate: f32,
    num_iterations: usize,
  ) {
    // let observations_matrix = Matrix::new_2d(&observations).transpose();
    let conv_layer_outputs: &mut Vec<Vec<Matrix>>;

    for i in 0..num_iterations {
      // Feed forward
      // self.feed_forward(observations, conv_layer_outputs);

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
    // let prev_layer_outputs = if layer == 0 {
    //   observations
    // } else {
    //   &neuron_outputs[layer - 1]
    // };

    // Possibly need a normalization factor?
    //let normalization_factor = 1.0 / prev_layer_outputs.columns as f32;

    // RelU
    // let activation_prime_x = neuron_outputs[layer].element_ReLU_prime();

    // Calculate Error to backprop

    // Calculate error in bias

    // Calculate error in erights
  }
}
