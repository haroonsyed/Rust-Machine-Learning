use itertools::{izip, Itertools};
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
    input_depth: usize,
    filters_per_conv_layer: Vec<usize>,
    filter_dimension: usize, // Must be odd
  ) -> Self {
    let network = ConvolutionalNeuralNetworkRust::new(
      num_classifications,
      max_pool_stride,
      input_width,
      input_height,
      input_depth,
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
      for _ in 0..num_iterations {
        let (observations, labels) = batch_loader.batch_sample(batch_size);
        self.network.train(&observations, &labels, learning_rate);
      }
    }
  }

  fn train_raw_data(
    &mut self,
    observations: Vec<Vec<Vec<f32>>>,
    labels: Vec<f32>,
    learning_rate: f32,
  ) {
    self.network.train(&observations, &labels, learning_rate);
  }

  fn classify(&self, features_test: Vec<Vec<Vec<f32>>>) -> PyResult<Vec<f32>> {
    let classifications = self.network.classify(&features_test);

    return Ok(classifications);
  }
}

pub struct ConvolutionalNeuralNetworkRust {
  pub num_classifications: usize,
  pub max_pool_stride: usize,
  pub input_width: usize,
  pub input_height: usize,
  pub input_depth: usize,
  pub conv_layers: Vec<Vec<Vec<Matrix>>>, // Layer -> Filter -> Depth
  pub biases: Vec<Vec<Matrix>>,           // Layer -> Filter -> Bias
  pub fully_connected_layer: BasicNeuralNetworkRust,
}

impl ConvolutionalNeuralNetworkRust {
  pub fn new(
    num_classifications: usize,
    max_pool_stride: usize,
    input_width: usize,
    input_height: usize,
    input_depth: usize,
    filters_per_conv_layer: Vec<usize>,
    filter_dimension: usize, // Must be odd
  ) -> Self {
    // Init the filters and biases
    // Random seed for weights
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 0.68).unwrap();

    let num_layers = filters_per_conv_layer.len();
    let filter_depth_per_conv_layer = (0..filters_per_conv_layer.len())
      .map(|index| {
        if index == 0 {
          return input_depth;
        } else {
          return filters_per_conv_layer[index - 1];
        }
      })
      .collect_vec();

    let biases = filters_per_conv_layer
      .iter()
      .map(|filter_count| {
        (0..*filter_count)
          .map(|_| Matrix::zeros(filter_dimension, filter_dimension))
          .collect_vec()
      })
      .collect_vec();

    let conv_layers = izip!(
      filters_per_conv_layer.iter(),
      filter_depth_per_conv_layer.iter()
    )
    .map(|(&filter_count, &filter_depth)| {
      (0..filter_count)
        .map(|_| {
          (0..filter_depth)
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
        .collect_vec()
    })
    .collect_vec();

    // Create the fully connected layer
    // hidden layer size is 0, it is just a translation layer of linearized max pool to classification
    // num_features will be final conv layer linearized
    let num_max_pool = num_layers / max_pool_stride;
    let final_filtered_width = input_width / (2 << num_max_pool);
    let final_filtered_height = input_height / (2 << num_max_pool);
    let num_features = (final_filtered_width * final_filtered_height)
      * filter_depth_per_conv_layer.last().unwrap_or(&1);
    let fully_connected_layer =
      BasicNeuralNetworkRust::new(Vec::new(), num_features, num_classifications);

    // Create the CNN object
    let network = ConvolutionalNeuralNetworkRust {
      num_classifications,
      max_pool_stride,
      input_width,
      input_height,
      input_depth,
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
  ) {
    // Obervations are of shape sample -> depth -> data
    // Observations matrices is sample -> depth -> data (as matrix with image width and height)
    let observations_matrices = observations
      .iter()
      .map(|sample| {
        sample
          .iter()
          .map(|channel_data| Matrix::new_1d(channel_data, self.input_height, self.input_width))
          .collect_vec()
      })
      .collect_vec();

    // Feed forward
    let filter_outputs = self.feed_forward(&observations_matrices);

    // Calculate error from feed forward step

    // Backpropogate hidden

    // Update the weights
  }

  //return Vec<Vec<Matrix>> layer -> filters -> Matrix
  pub fn feed_forward(&self, observations: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>> {
    return Vec::new();
  }

  pub fn backpropogation_hidden_layer(
    &mut self,
    observations: &Vec<Vec<Matrix>>,
    filter_outputs: &Vec<Vec<Matrix>>,
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
