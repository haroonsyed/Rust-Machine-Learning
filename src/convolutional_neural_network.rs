use itertools::{izip, Itertools};
use tensor_lib::*;

use crate::basic_neural_network::BasicNeuralNetworkRust;

pub struct ConvolutionalNeuralNetworkRust {
  pub input_dimensions: (usize, usize, usize), // Height, Width, Depth
  pub num_classifications: usize,
  pub layers: Vec<Box<dyn CNN_Layer>>,
  fully_connected_layer: Option<FullyConnectedLayer>,
}

impl ConvolutionalNeuralNetworkRust {
  pub fn new(
    num_classifications: usize,
    input_height: usize,
    input_width: usize,
    input_depth: usize,
  ) -> Self {
    return Self {
      input_dimensions: (input_height, input_width, input_depth),
      num_classifications,
      layers: Vec::new(),
      fully_connected_layer: None,
    };
  }

  pub fn add_convolutional_layer(
    &mut self,
    filter_height: usize,
    filter_width: usize,
    filter_count: usize,
  ) {
    let (input_height, input_width, input_depth) = self
      .layers
      .last()
      .map_or(self.input_dimensions, |layer| layer.get_output_dimensions());

    self.layers.push(Box::new(ConvolutionalLayerRust::new(
      input_height,
      input_width,
      input_depth,
      filter_height,
      filter_width,
      filter_count,
    )));
  }

  pub fn add_max_pool_layer(&mut self) {
    let (input_height, input_width, input_depth) = self
      .layers
      .last()
      .map_or(self.input_dimensions, |layer| layer.get_output_dimensions());

    self.layers.push(Box::new(MaxPoolLayerRust::new(
      input_height,
      input_width,
      input_depth,
    )));
  }

  pub fn add_fully_connected_layer(&mut self) {
    // Check if network is finalized
    if self.fully_connected_layer.is_some() {
      panic!("Network is already finalized and has a fully connected layer!");
    }

    let (input_height, input_width, input_depth) = self
      .layers
      .last()
      .map_or(self.input_dimensions, |layer| layer.get_output_dimensions());

    self.fully_connected_layer = Some(FullyConnectedLayer::new(
      input_height,
      input_width,
      input_depth,
      self.num_classifications,
    ));
  }

  fn convert_observations_to_matrices(
    &self,
    observations: &Vec<Vec<Vec<f32>>>,
  ) -> Vec<Vec<Matrix>> {
    let (input_height, input_width, _) = self.input_dimensions;
    return observations
      .iter()
      .map(|sample| {
        sample
          .iter()
          .map(|channel_data| Matrix::new_1d(channel_data, input_height, input_width))
          .collect_vec()
      })
      .collect_vec();
  }

  pub fn classify(&mut self, features_test: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    // Obervations are of shape sample -> depth -> pixels
    let observations_matrices = self.convert_observations_to_matrices(features_test);

    // Feed forward
    let filter_outputs = self.feed_forward(&observations_matrices);

    // Check none
    if self.fully_connected_layer.is_none() {
      panic!("Fully Connected Layer Is Missing!");
    }

    // Classify
    return self
      .fully_connected_layer
      .as_mut()
      .unwrap()
      .classify(&filter_outputs);
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

    // Check none
    if self.fully_connected_layer.is_none() {
      panic!("Fully Connected Layer Is Missing!");
    }
    let fc_error =
      self
        .fully_connected_layer
        .as_mut()
        .unwrap()
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
  fn get_input_dimensions(&self) -> (usize, usize, usize);
  fn get_output_dimensions(&self) -> (usize, usize, usize);
}

pub struct ConvolutionalLayerRust {
  pub filters: Vec<Vec<Matrix>>,                // Filter -> Depth
  pub biases: Vec<Matrix>,                      // Filter
  pub prev_input: Vec<Vec<Matrix>>,             // Sample -> Depth
  pub prev_output: Vec<Vec<Matrix>>,            // Sample -> Depth
  pub input_dimensions: (usize, usize, usize),  // Height, Width, Depth
  pub output_dimensions: (usize, usize, usize), // Height, Width, Depth
}

impl ConvolutionalLayerRust {
  pub fn new(
    input_height: usize,
    input_width: usize,
    input_depth: usize,
    filter_height: usize,
    filter_width: usize,
    filter_count: usize,
  ) -> Self {
    let input_dimensions = (input_height, input_width, input_depth);
    let output_dimensions = (
      input_height - filter_height + 1,
      input_width - filter_width + 1,
      filter_count,
    );

    let mut filters = Vec::new();
    let mut biases = Vec::new();
    let prev_input = Vec::new();
    let prev_output = Vec::new();

    // Create the filters
    for _ in 0..filter_count {
      let mut filter = Vec::new();
      for _ in 0..input_depth {
        filter.push(Matrix::new_random(
          0.0,
          f64::sqrt(2.0 / (filter_height * filter_width) as f64), // He initialization
          filter_height,
          filter_width,
        ));
      }
      filters.push(filter);
    }

    // Create the biases
    for _ in 0..filter_count {
      biases.push(Matrix::zeros(output_dimensions.0, output_dimensions.1));
    }

    return Self {
      filters,
      biases,
      prev_input,
      prev_output,
      input_dimensions,
      output_dimensions,
    };
  }
}

impl CNN_Layer for ConvolutionalLayerRust {
  fn get_input_dimensions(&self) -> (usize, usize, usize) {
    return self.input_dimensions;
  }

  fn get_output_dimensions(&self) -> (usize, usize, usize) {
    return self.output_dimensions;
  }

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
    self.prev_output = sample_filter_outputs.clone();
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

    // Apply relu prime
    for (sample_error, sample_output) in izip!(sample_output_errors.iter(), self.prev_output.iter())
    {
      for (filter_error, filter_output) in izip!(sample_error, sample_output) {
        filter_error.element_multiply_inplace(&filter_output.element_ReLU_prime());
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

pub struct MaxPoolLayerRust {
  pub prev_input_bm: Vec<Vec<Matrix>>,
  pub input_dimensions: (usize, usize, usize), // Height, Width, Depth
  pub output_dimensions: (usize, usize, usize), // Height, Width, Depth
}

impl MaxPoolLayerRust {
  pub fn new(input_height: usize, input_width: usize, input_depth: usize) -> Self {
    let input_dimensions = (input_height, input_width, input_depth);
    let output_dimensions = (input_height / 2, input_width / 2, input_depth);

    return Self {
      prev_input_bm: Vec::new(),
      input_dimensions,
      output_dimensions,
    };
  }
}

impl CNN_Layer for MaxPoolLayerRust {
  fn get_input_dimensions(&self) -> (usize, usize, usize) {
    return self.input_dimensions;
  }

  fn get_output_dimensions(&self) -> (usize, usize, usize) {
    return self.output_dimensions;
  }

  fn feed_forward(&mut self, input: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>> {
    let mut pooled_samples = Vec::new();
    let mut input_bm = Vec::new();
    for sample in input {
      let mut pooled_channels = Vec::new();
      let mut input_bm_channels = Vec::new();
      for channel in sample {
        let (pooled_channel, bitmask) = channel.max_pool();
        pooled_channels.push(pooled_channel);
        input_bm_channels.push(bitmask);
      }
      pooled_samples.push(pooled_channels);
      input_bm.push(input_bm_channels);
    }

    self.prev_input_bm = input_bm;
    return pooled_samples;
  }

  fn backpropogation(&mut self, error: &Vec<Vec<Matrix>>, _learning_rate: f32) -> Vec<Vec<Matrix>> {
    // Max pool is not differentiable, so we will just pass the error back to the max value
    // Element wise multiply max mask by nearest_neighbor upscaled error

    let mut sample_input_errors = Vec::new();

    for (sample_error, sample_bitmasks) in izip!(error, self.prev_input_bm.iter()) {
      let mut channel_input_errors = Vec::new();
      for (channel_error, bitmask) in izip!(sample_error, sample_bitmasks) {
        // Upsample the error
        let upsampled_error = channel_error.nearest_neighbor_2x_upsample();

        // Element wise multiply
        upsampled_error.element_multiply_inplace(bitmask);
        channel_input_errors.push(upsampled_error);
      }
      sample_input_errors.push(channel_input_errors);
    }

    return sample_input_errors;
  }
}

struct FullyConnectedLayer {
  pub input_dimensions: (usize, usize, usize), // Height, Width, Depth
  pub fully_connected_layer: BasicNeuralNetworkRust,
}

impl FullyConnectedLayer {
  pub fn new(
    input_height: usize,
    input_width: usize,
    input_depth: usize,
    num_classifications: usize,
  ) -> Self {
    return Self {
      input_dimensions: (input_height, input_width, input_depth),
      fully_connected_layer: BasicNeuralNetworkRust::new(
        Vec::new(),
        input_height * input_width * input_depth,
        num_classifications,
      ),
    };
  }

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
    let (input_height, input_width, _) = self.input_dimensions;
    let unflattened_errors = sample_errors
      .iter()
      .map(|sample_error| {
        unflatten_array_strided_to_matrices(&sample_error, input_height, input_width)
      })
      .collect_vec();

    return unflattened_errors;
  }
}
