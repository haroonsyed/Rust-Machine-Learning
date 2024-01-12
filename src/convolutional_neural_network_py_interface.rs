use std::collections::HashMap;

use itertools::Itertools;
use pyo3::prelude::*;
use tensor_lib::{create_matrix_group, cuda_bindings::cuda_synchronize, Matrix};

use crate::{
  convolutional_neural_network::ConvolutionalNeuralNetworkRust,
  image_util::ImageBatchLoaderRust,
  packed_optimizers::{
    PackedAdamOptimizer, PackedMomentumOptimizer, PackedStochasticGradientDescentOptimizer,
  },
};

#[pyclass(unsendable)]
pub struct ConvolutionalNeuralNetwork {
  pub network: ConvolutionalNeuralNetworkRust,
  batch_loader: Option<ImageBatchLoaderRust>,
}

#[pymethods]
impl ConvolutionalNeuralNetwork {
  #[new]
  fn new(
    num_classifications: usize,
    input_height: usize,
    input_width: usize,
    input_depth: usize,
  ) -> Self {
    let network = ConvolutionalNeuralNetworkRust::new(
      num_classifications,
      input_height,
      input_width,
      input_depth,
    );

    // Cleanup and return
    return Self {
      network,
      batch_loader: Option::None,
    };
  }

  fn set_optimizer_stochastic_gradient_descent(&mut self, learning_rate: f32) {
    let optimizer = Box::new(PackedStochasticGradientDescentOptimizer::new(learning_rate));
    self.network.set_optimizer(optimizer);
  }

  fn set_optimizer_momentum(&mut self, learning_rate: f32, beta: f32) {
    let optimizer = Box::new(PackedMomentumOptimizer::new(learning_rate, beta));
    self.network.set_optimizer(optimizer);
  }

  // fn set_optimizer_adagrad(&mut self, learning_rate: f32) {
  //   let optimizer = Box::new(PackedAdagradOptimizer::new(learning_rate));
  //   self.network.set_optimizer(optimizer);
  // }

  // fn set_optimizer_RMSProp(&mut self, learning_rate: f32, beta: f32) {
  //   let optimizer = Box::new(PackedRMSPropOptimizer::new(learning_rate, beta));
  //   self.network.set_optimizer(optimizer);
  // }

  fn set_optimizer_adam(&mut self, learning_rate: f32, beta1: f32, beta2: f32) {
    let optimizer = Box::new(PackedAdamOptimizer::new(learning_rate, beta1, beta2));
    self.network.set_optimizer(optimizer);
  }

  fn add_convolutional_layer(
    &mut self,
    filter_height: usize,
    filter_width: usize,
    filter_count: usize,
  ) {
    self
      .network
      .add_convolutional_layer(filter_height, filter_width, filter_count);
  }

  fn add_max_pool_layer(&mut self) {
    self.network.add_max_pool_layer();
  }

  fn add_fully_connected_layer(&mut self) {
    self.network.add_fully_connected_layer();
  }

  fn set_image_loader(&mut self, parent_folder: String, sample_height: usize, sample_width: usize) {
    self.batch_loader = Option::Some(ImageBatchLoaderRust::new(
      parent_folder,
      sample_width,
      sample_height,
    ));
  }

  fn train_using_image_loader(&mut self, batch_size: usize, num_iterations: usize) {
    if let Some(batch_loader) = self.batch_loader.as_ref() {
      for _ in 0..num_iterations {
        let (observations, labels) = batch_loader.batch_sample_as_matrix(batch_size);
        self.network.train(observations, labels);
      }
      unsafe { cuda_synchronize() }
    }
  }

  fn train_raw_data(&mut self, observations: Vec<Vec<Vec<f32>>>, labels: Vec<f32>) {
    let num_classifications = self.network.num_classifications;
    let observations_matrices = self.convert_features_to_matrices(&observations);
    let encoded_labels = Self::convert_labels_one_hot_encoded(&labels, num_classifications);
    self.network.train(observations_matrices, encoded_labels);
    unsafe { cuda_synchronize() }
  }

  fn classify(&mut self, features_test: Vec<Vec<Vec<f32>>>) -> PyResult<Vec<f32>> {
    let features_matrix = self.convert_features_to_matrices(&features_test);
    let classifications = self.network.classify(features_matrix);

    return Ok(classifications);
  }

  fn get_image_loader_classification_map(&self) -> PyResult<HashMap<String, f32>> {
    if let Some(batch_loader) = &self.batch_loader {
      let map = &batch_loader.classifications_map;
      return Ok(map.clone());
    }
    return Ok(HashMap::new());
  }

  fn get_performance_info(&self) -> PyResult<Vec<(f32, f32)>> {
    let performance_info = self.network.get_performance_info();
    return Ok(performance_info);
  }
}

impl ConvolutionalNeuralNetwork {
  fn convert_labels_one_hot_encoded(labels: &Vec<f32>, num_classifications: usize) -> Matrix {
    return Matrix::new_one_hot_encoded(labels, num_classifications).transpose();
  }

  fn convert_features_to_matrices(&self, features: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Matrix>> {
    let (input_height, input_width, _) = self.network.input_dimensions;
    let sample_count = features.len();
    let channel_count = features[0].len();
    let matrix_count = sample_count * channel_count;

    let matrices = create_matrix_group(input_height, input_width, matrix_count);

    let mut current_matrix_index = 0;
    for sample in features {
      for channel in sample {
        matrices[current_matrix_index].set_data_1d(channel);
        current_matrix_index += 1;
      }
    }

    // Now group into sample -> depth
    let grouped_matrices = matrices
      .into_iter()
      .chunks(channel_count)
      .into_iter()
      .map(|sample| sample.into_iter().collect_vec())
      .collect_vec();

    return grouped_matrices;
  }
}
