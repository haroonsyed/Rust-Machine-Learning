use itertools::{izip, Itertools};
use tensor_lib::*;

use crate::basic_neural_network::BasicNeuralNetworkRust;

pub struct ConvolutionalNeuralNetworkRust {
  num_classifications: usize,
  input_width: usize,
  input_height: usize,
  input_depth: usize,
  layers: Vec<Box<dyn CNN_Layer>>,
  fully_connected_layer: FullyConnectedLayer,
}

impl ConvolutionalNeuralNetworkRust {
  pub fn new(
    num_classifications: usize,
    input_width: usize,
    input_height: usize,
    input_depth: usize,
    filters: Vec<Box<dyn CNN_Layer>>,
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
      fully_connected_layer: FullyConnectedLayer {
        // Move to new() function
        input_width: conv_layer_output_width,
        input_height: conv_layer_output_height,
        fully_connected_layer: BasicNeuralNetworkRust::new(
          Vec::new(),
          conv_layer_output_width * conv_layer_output_height * conv_layer_output_depth,
          num_classifications,
        ),
      },
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

    return self.fully_connected_layer.classify(&filter_outputs);
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
    let filter_outputs = self.feed_forward(&observations_matrices);

    // Train through FC
    let fc_error = self
      .fully_connected_layer
      .train(&filter_outputs, labels, learning_rate);

    // Backpropogate
    self.backpropogation(fc_error, learning_rate);
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
    unflattened_fc_error: Vec<Vec<Matrix>>, // sample -> depth -> data
    learning_rate: f32,
  ) {
    let mut prev_layer_error = unflattened_fc_error;
    for layer in self.layers.iter_mut().rev() {
      prev_layer_error = layer.backpropogation(&prev_layer_error, learning_rate);
    }
  }
}

pub trait CNN_Layer: Send {
  fn feed_forward(&mut self, input: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>>;
  fn backpropogation(&mut self, error: &Vec<Vec<Matrix>>, learning_rate: f32) -> Vec<Vec<Matrix>>;
}

struct ConvolutionalLayerRust {
  pub filters: Vec<Vec<Matrix>>,    // Filter -> Depth
  pub biases: Vec<Matrix>,          // Filter
  pub prev_input: Vec<Vec<Matrix>>, // Sample -> Depth
}

impl CNN_Layer for ConvolutionalLayerRust {
  fn feed_forward(&mut self, input: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>> {
    let mut sample_filter_outputs = Vec::new();

    // Valid Convolution
    let input_height = input[0][0].rows;
    let input_width = input[0][0].columns;
    let output_height = input_height - self.filters[0][0].rows + 1;
    let output_width = input_width - self.filters[0][0].columns + 1;

    // Sample
    for sample in input {
      let mut filter_outputs = Vec::new();

      // Filter
      for (filter, bias) in izip!(self.filters.iter(), self.biases.iter()) {
        let filter_output = Matrix::zeros(output_height, output_width);

        // Channel
        for (channel, kernel) in izip!(sample, filter) {
          let channel_output = channel.convolution(kernel, ConvolutionType::VALID);
          filter_output.element_add_inplace(&channel_output);
        }

        // Add the bias
        filter_output.element_add_inplace(bias);

        // relu
        filter_output.element_ReLU_inplace();

        filter_outputs.push(filter_output);
      }

      sample_filter_outputs.push(filter_outputs);
    }

    self.prev_input = input.clone();
    return sample_filter_outputs;
  }

  fn backpropogation(
    &mut self,
    sample_output_errors: &Vec<Vec<Matrix>>,
    learning_rate: f32,
  ) -> Vec<Vec<Matrix>> {
    let mut sample_input_errors = Vec::new();

    // n is the filter
    // m is the channel

    // Apply relu prime to error
    for sample_error in sample_output_errors {
      for filter_error in sample_error {
        filter_error.element_ReLU_prime_inplace();
      }
    }

    // PER SAMPLE
    for (sample_output_error, sample_prev_input) in
      izip!(sample_output_errors, self.prev_input.iter())
    {
      // Calculate the input error
      // PER FILTER
      let mut sample_input_error = Vec::new();
      for (filter_output_error, filter) in izip!(sample_output_error.iter(), self.filters.iter()) {
        // Xm' = Xm - sum(de/dy * conv_full * Knm)
        let delta_xm =
          filter_output_error.convolution(&filter[0].rotate_180(), ConvolutionType::FULL);

        // PER CHANNEL
        for channel in filter[1..].iter() {
          delta_xm.element_add_inplace(
            &filter_output_error.convolution(&channel.rotate_180(), ConvolutionType::FULL),
          );
        }

        sample_input_error.push(delta_xm);
      }

      // Update the biases
      // PER FILTER
      for (filter_output_error, bias) in izip!(sample_output_error.iter(), self.biases.iter_mut()) {
        // b' = b - de/dy * learning_rate
        bias.element_subtract_inplace(&filter_output_error.scalar_multiply(learning_rate));
      }

      // Update the filters
      // PER FILTER
      for (filter_output_error, filter) in izip!(sample_output_error, self.filters.iter()) {
        // PER CHANNEL
        for (prev_channel_input, channel) in izip!(sample_prev_input, filter) {
          // Knm' = Knm - learning_rate * Xm * conv_valid * de/dy
          let delta_channel =
            prev_channel_input.convolution(filter_output_error, ConvolutionType::VALID);
          channel.element_subtract_inplace(&delta_channel.scalar_multiply(learning_rate));
        }
      }

      sample_input_errors.push(sample_input_error);
    }

    return sample_input_errors;
  }
}

struct MaxPoolLayerRust {
  pub prev_input: Vec<Vec<Matrix>>, // Necessary for bitmask on backprop
}

impl CNN_Layer for MaxPoolLayerRust {
  fn feed_forward(&mut self, input: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>> {
    let mut pooled_samples = Vec::new();
    for sample in input {
      let mut pooled_channels = Vec::new();
      for channel in sample {
        pooled_channels.push(channel.max_pool());
      }
      pooled_samples.push(pooled_channels);
    }

    self.prev_input = input.clone();
    return pooled_samples;
  }

  fn backpropogation(&mut self, error: &Vec<Vec<Matrix>>, learning_rate: f32) -> Vec<Vec<Matrix>> {
    return Vec::new();
  }
}

struct FullyConnectedLayer {
  pub input_width: usize,
  pub input_height: usize,
  pub fully_connected_layer: BasicNeuralNetworkRust,
}

impl FullyConnectedLayer {
  fn classify(&mut self, input: &Vec<Vec<Matrix>>) -> Vec<f32> {
    let flattened_input = self.flatten(input);
    return self.fully_connected_layer.classify_matrix(&flattened_input);
  }

  fn train(
    &mut self,
    input: &Vec<Vec<Matrix>>,
    labels: &Vec<f32>,
    learning_rate: f32,
  ) -> Vec<Vec<Matrix>> {
    let flattened_input = self.flatten(input);
    let fc_error = self
      .fully_connected_layer
      .train_classification_observation_matrix(&flattened_input, labels, learning_rate);
    return self.unflatten(fc_error);
  }

  fn flatten(&mut self, input: &Vec<Vec<Matrix>>) -> Matrix {
    let sample_count = input.len();

    let to_flatten = input.to_owned().into_iter().flatten().collect_vec();
    let mut flattened = flatten_matrix_array(&to_flatten);

    // Take tranpose to ensure each column is a sample
    let before_transpose_flattened_rows = sample_count;
    let before_transpose_flattened_columns = flattened.get_data_length() / sample_count;
    flattened.reshape(
      before_transpose_flattened_rows,
      before_transpose_flattened_columns,
    );

    // Now take tranpose
    flattened = flattened.transpose();

    return flattened;
  }

  fn unflatten(&mut self, fc_error: Matrix) -> Vec<Vec<Matrix>> {
    // The fc_error is with respect to classification neurons, not input neurons to fc layer
    // So we will quickly do that here (adding it to Basic Neural Network was clunky)
    let mut fc_error = self.fully_connected_layer.weights[0]
      .transpose()
      .matrix_multiply(&fc_error);

    // Now let's take the transpose of that to make observations the rows
    fc_error = fc_error.transpose();

    // Now unflatten the output error row by row back to input shape
    // Because I decided not to treat each sample separately to FC I need to double unflatten (sample -> depth)
    let sample_errors = unflatten_array_strided_to_matrices(&fc_error, 1, fc_error.columns);
    let unflattened_errors = sample_errors
      .iter()
      .map(|sample_error| {
        unflatten_array_strided_to_matrices(&sample_error, self.input_height, self.input_width)
      })
      .collect_vec();

    return unflattened_errors;
  }
}
