use std::collections::HashMap;

use pyo3::prelude::*;

use crate::{
  convolutional_neural_network::ConvolutionalNeuralNetworkRust, image_util::ImageBatchLoaderRust,
};

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
    self
      .network
      .set_optimizer_stochastic_gradient_descent(learning_rate);
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
        let (observations, labels) = batch_loader.batch_sample(batch_size);
        self.network.train(&observations, &labels);
      }
    }
  }

  fn train_raw_data(&mut self, observations: Vec<Vec<Vec<f32>>>, labels: Vec<f32>) {
    self.network.train(&observations, &labels);
  }

  fn classify(&mut self, features_test: Vec<Vec<Vec<f32>>>) -> PyResult<Vec<f32>> {
    let classifications = self.network.classify(&features_test);

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
