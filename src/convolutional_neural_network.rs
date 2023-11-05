use itertools::{izip, Itertools};
use tensor_lib::*;

use crate::basic_neural_network::BasicNeuralNetworkRust;

pub trait ConvolutionalLayer: Send {
  fn feed_forward(&mut self, input: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>>;
  fn backpropogation(&mut self, error: &Vec<Matrix>, learning_rate: f32) -> Vec<Matrix>;
}

struct ConvolutionalLayerRust {
  pub filters: Vec<Vec<Matrix>>, // Filter -> Depth
  pub biases: Vec<Matrix>,       // Filter
  pub prev_input: Vec<Vec<Matrix>>,
}

struct FlattenLayerRust {
  pub prev_output: Vec<Vec<Matrix>>,
}

struct MaxPoolLayerRust {
  pub prev_input: Vec<Vec<Matrix>>,
}

pub struct ConvolutionalNeuralNetworkRust {
  pub num_classifications: usize,
  pub input_width: usize,
  pub input_height: usize,
  pub input_depth: usize,
  pub layers: Vec<Box<dyn ConvolutionalLayer>>,
  pub fully_connected_layer: BasicNeuralNetworkRust,
}

impl ConvolutionalNeuralNetworkRust {
  pub fn new(
    num_classifications: usize,
    input_width: usize,
    input_height: usize,
    input_depth: usize,
    filters: Vec<Box<dyn ConvolutionalLayer>>,
  ) -> Self {
    let conv_layer_output_height = 10;
    let conv_layer_output_width = 10;
    let conv_layer_output_depth = 10;
    return Self {
      num_classifications,
      input_width,
      input_height,
      input_depth,
      layers: filters,
      fully_connected_layer: BasicNeuralNetworkRust::new(
        Vec::new(), // No hidden layers
        conv_layer_output_height * conv_layer_output_width * conv_layer_output_depth,
        num_classifications,
      ),
    };
  }

  fn convert_observations_to_matrices(
    &self,
    observations: &Vec<Vec<Vec<f32>>>,
  ) -> Vec<Vec<Matrix>> {
    return observations
      .iter()
      .map(|sample| {
        sample
          .iter()
          .map(|channel_data| Matrix::new_1d(channel_data, self.input_height, self.input_width))
          .collect_vec()
      })
      .collect_vec();
  }

  pub fn classify(&mut self, features_test: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    // Obervations are of shape sample -> depth -> pixels
    let observations_matrices = self.convert_observations_to_matrices(features_test);

    // Feed forward
    let filter_outputs = self.feed_forward(&observations_matrices);

    // Linearize the final output

    // Classify through FC

    return Vec::new();
  }

  pub fn train(
    &mut self,
    observations: &Vec<Vec<Vec<f32>>>, // sample -> depth -> pixels
    labels: &Vec<f32>,
    learning_rate: f32,
  ) {
    // Convert observations to matrices
    let observations_matrices = self.convert_observations_to_matrices(observations);

    // Feed forward
    let flattened_forward_output = self.feed_forward(&observations_matrices);

    // TODO: Flatten all of them into one matrix

    // Train through the fully connected layer

    // Unflatten the error

    // Backpropogate
    // self.backpropogation(&fc_error, learning_rate);
  }

  //return Vec<Vec<Vec<Matrix>>> sample -> layer -> filters -> Matrix
  pub fn feed_forward(
    &mut self,
    observations: &Vec<Vec<Matrix>>, // sample -> depth -> data
  ) -> Vec<Vec<Matrix>> {
    let mut prev_layer_output = observations.clone();
    for layer in self.layers.iter_mut() {
      prev_layer_output = layer.feed_forward(&prev_layer_output);
    }
    return prev_layer_output;
  }

  pub fn backpropogation(
    &mut self,
    fc_error: Vec<Matrix>, // sample -> depth -> data
    learning_rate: f32,
  ) {
    let mut prev_layer_error = fc_error;
    for layer in self.layers.iter_mut().rev() {
      prev_layer_error = layer.backpropogation(&prev_layer_error, learning_rate);
    }
  }
}
