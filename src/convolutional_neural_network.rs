use itertools::{izip, Itertools};
use tensor_lib::*;

use crate::{
  basic_neural_network::BasicNeuralNetworkRust,
  optimizers::{AdamOptimizer, Optimizer},
};

pub struct ConvolutionalNeuralNetworkRust {
  pub input_dimensions: (usize, usize, usize), // Height, Width, Depth
  pub num_classifications: usize,
  pub layers: Vec<Box<dyn CNN_Layer>>,
  fully_connected_layer: Option<FullyConnectedLayer>,
  optimizer: Box<dyn Optimizer>,
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
      optimizer: Box::new(AdamOptimizer::new(1e-2, 0.9, 0.999)), // Default optimizer
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
      &self.optimizer,
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

    self
      .fully_connected_layer
      .as_mut()
      .unwrap()
      .fully_connected_layer
      .set_optimizer(self.optimizer.clone());
  }

  pub fn set_optimizer(&mut self, optimizer: Box<dyn Optimizer>) {
    self.optimizer = optimizer;

    // TODO: Work out nice way to set optimizer after adding layers (although this is not really needed, but it would be nice)
    if self.layers.len() > 0 {
      panic!("Please set optimizer before adding layers!");
    }
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
    if features_test.len() == 0 {
      return Vec::new();
    }

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
  ) {
    if observations.len() == 0 {
      return;
    }

    // Convert observations to matrices
    let observations_matrices = self.convert_observations_to_matrices(observations);

    // Feed forward
    let filter_outputs = self.feed_forward(&observations_matrices);

    // Train through FC

    // Check none
    if self.fully_connected_layer.is_none() {
      panic!("Fully Connected Layer Is Missing!");
    }
    let fc_error = self
      .fully_connected_layer
      .as_mut()
      .unwrap()
      .train(&filter_outputs, labels);

    // Backpropogate
    self.backpropogation(fc_error);
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
  ) {
    let mut prev_layer_error = unflattened_fc_error;
    for layer in self.layers.iter_mut().rev() {
      prev_layer_error = layer.backpropogation(&prev_layer_error);
    }
  }

  pub fn get_performance_info(&self) -> Vec<(f32, f32)> {
    return self
      .fully_connected_layer
      .as_ref()
      .unwrap()
      .fully_connected_layer
      .get_performance_info();
  }
}

pub trait CNN_Layer: Send {
  fn feed_forward(&mut self, input: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>>;
  fn backpropogation(&mut self, error: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>>;
  fn get_input_dimensions(&self) -> (usize, usize, usize);
  fn get_output_dimensions(&self) -> (usize, usize, usize);
}

pub struct ConvolutionalLayerRust {
  pub filters: Vec<Vec<Matrix>>,                       // Filter -> Depth
  pub biases: Vec<Matrix>,                             // Filter
  pub filter_optimizers: Vec<Vec<Box<dyn Optimizer>>>, // Filter -> Depth
  pub bias_optimizers: Vec<Box<dyn Optimizer>>,        // Bias
  pub prev_input: Vec<Vec<Matrix>>,                    // Sample -> Depth
  pub prev_output: Vec<Vec<Matrix>>,                   // Sample -> Depth
  pub input_dimensions: (usize, usize, usize),         // Height, Width, Depth
  pub output_dimensions: (usize, usize, usize),        // Height, Width, Depth
}

impl ConvolutionalLayerRust {
  pub fn new(
    input_height: usize,
    input_width: usize,
    input_depth: usize,
    filter_height: usize,
    filter_width: usize,
    filter_count: usize,
    optimizer: &Box<dyn Optimizer>,
  ) -> Self {
    let input_dimensions = (input_height, input_width, input_depth);
    let output_dimensions = (
      input_height - filter_height + 1,
      input_width - filter_width + 1,
      filter_count,
    );

    let mut filters = Vec::new();
    let mut filter_optimizers = Vec::new();
    let mut biases = Vec::new();
    let mut bias_optimizers = Vec::new();
    let prev_input = Vec::new();
    let prev_output = Vec::new();

    // Create the filters
    for _ in 0..filter_count {
      let mut filter = Vec::new();
      let mut channel_optimizers = Vec::new();
      for _ in 0..input_depth {
        filter.push(Matrix::new_random(
          0.0,
          f64::sqrt(2.0 / (filter_height * filter_width * input_depth) as f64), // He initialization
          filter_height,
          filter_width,
        ));
        channel_optimizers.push(optimizer.clone());
      }
      filters.push(filter);
      filter_optimizers.push(channel_optimizers);
    }

    // Create the biases
    for _ in 0..filter_count {
      biases.push(Matrix::zeros(output_dimensions.0, output_dimensions.1));
      bias_optimizers.push(optimizer.clone());
    }

    return Self {
      filters,
      biases,
      filter_optimizers,
      bias_optimizers,
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
    let mut channels_to_correlate = Vec::new();
    let mut kernels_to_correlate = Vec::new();
    let mut filter_result_index_to_sum_filter_result_to = Vec::new();
    let mut filter_result_index_to_sum = Vec::new();
    let mut filter_result_index_to_sum_bias_to = Vec::new();
    let mut biases_to_sum = Vec::new();

    let mut raw_filter_output_indices = Vec::new();

    // The below logic is used to setup the packed operation.
    // It follows from doing the operations non-packed (commented out below)
    // I understand the packed version is more complex, but it is much faster
    // Sample
    for sample in input {
      // Filter
      for (filter, bias) in izip!(self.filters.iter(), self.biases.iter()) {
        // let filter_output = sample[0].correlate(&filter[0], PaddingType::VALID);
        channels_to_correlate.push(sample[0].clone());
        kernels_to_correlate.push(filter[0].clone());
        let index_to_sum_to = channels_to_correlate.len() - 1;

        // Channel
        for (channel, kernel) in izip!(sample[1..].iter(), filter[1..].iter()) {
          // let channel_output = channel.correlate(kernel, PaddingType::VALID);
          channels_to_correlate.push(channel.clone());
          kernels_to_correlate.push(kernel.clone());

          // filter_output.element_add_inplace(&channel_output);
          filter_result_index_to_sum_filter_result_to.push(index_to_sum_to);
          filter_result_index_to_sum.push(channels_to_correlate.len() - 1);
        }

        // Add the bias
        // filter_output.element_add_inplace(bias);
        filter_result_index_to_sum_bias_to.push(index_to_sum_to);
        biases_to_sum.push(bias.clone());

        raw_filter_output_indices.push(index_to_sum_to);

        // relu
        // filter_output.element_ReLU_inplace();
      }
    }

    // correlate
    let mut correlated_channels = correlate_packed(
      &channels_to_correlate,
      &kernels_to_correlate,
      PaddingType::VALID,
    );

    // Sum
    let mut sum_to = Vec::new();
    let mut to_add = Vec::new();
    let mut raw_filter_outputs = Vec::new();

    for (index_to_sum_to, index_to_sum) in izip!(
      filter_result_index_to_sum_filter_result_to,
      filter_result_index_to_sum
    ) {
      sum_to.push(correlated_channels[index_to_sum_to].clone());
      to_add.push(correlated_channels[index_to_sum].clone());
    }

    for index in raw_filter_output_indices {
      raw_filter_outputs.push(correlated_channels[index].clone());
    }

    for (index_to_sum_to, bias) in izip!(filter_result_index_to_sum_bias_to, biases_to_sum) {
      sum_to.push(correlated_channels[index_to_sum_to].clone());
      to_add.push(bias);
    }

    element_add_packed(&sum_to, &to_add, true);

    // ReLU
    let filter_outputs = element_ReLU_packed(&raw_filter_outputs, true);

    // Group into sample outputs of chunk size filter count
    let sample_filter_outputs = filter_outputs
      .into_iter()
      .chunks(self.filters.len())
      .into_iter()
      .map(|sample| sample.into_iter().collect_vec())
      .collect_vec();

    self.prev_input = input.clone();
    self.prev_output = sample_filter_outputs.clone();
    return sample_filter_outputs;
  }

  fn backpropogation(&mut self, sample_output_errors: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>> {
    let mut sample_input_errors = Vec::new();
    let normalization_factor = 1.0 / sample_output_errors.len() as f32;

    // n is the filter
    // m is the channel

    // Apply relu prime

    let filter_error_flattened = sample_output_errors
      .to_owned()
      .into_iter()
      .flatten()
      .collect_vec();

    let prev_output_flattened = self
      .prev_output
      .to_owned()
      .into_iter()
      .flatten()
      .collect_vec();

    element_ReLU_prime_packed(&prev_output_flattened, true);
    element_multiply_packed(&filter_error_flattened, &prev_output_flattened, true);

    // PER SAMPLE
    for (sample_output_error, sample_prev_input) in
      izip!(sample_output_errors, self.prev_input.iter())
    {
      // Calculate the input error
      // PER FILTER
      let mut sample_input_error = Vec::new();
      for (filter_error, filter) in izip!(sample_output_error.iter(), self.filters.iter()) {
        // deltaXm = sum(de/dy * conv_full * Knm)
        let delta_xm = filter_error.correlate(&filter[0].rotate_180(), PaddingType::FULL);

        // PER CHANNEL
        for channel in filter[1..].iter() {
          delta_xm
            .element_add_inplace(&filter_error.correlate(&channel.rotate_180(), PaddingType::FULL));
        }

        sample_input_error.push(delta_xm);
      }
      sample_input_errors.push(sample_input_error);

      // Update the biases
      // PER FILTER
      for (filter_output_error, bias, optimizer) in izip!(
        sample_output_error.iter(),
        self.biases.iter_mut(),
        self.bias_optimizers.iter_mut()
      ) {
        // b' = b - de/dy * learning_rate
        let bias_gradient = filter_output_error;
        let bias_step = optimizer.calculate_step(&bias_gradient);
        bias.element_subtract_inplace(&bias_step.scalar_multiply_inplace(normalization_factor));
      }

      // Update the filters
      // PER FILTER
      for (filter_output_error, filter, filter_optimizer) in izip!(
        sample_output_error,
        self.filters.iter(),
        self.filter_optimizers.iter_mut()
      ) {
        // PER CHANNEL
        for (prev_channel_input, channel, optimizer) in
          izip!(sample_prev_input, filter, filter_optimizer)
        {
          // Knm' = Knm - learning_rate * Xm * conv_valid * de/dy
          let delta_channel = prev_channel_input.correlate(filter_output_error, PaddingType::VALID);
          let channel_step = optimizer.calculate_step(&delta_channel);
          channel
            .element_subtract_inplace(&channel_step.scalar_multiply_inplace(normalization_factor));
        }
      }
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
    let output_dimensions = (
      input_height / 2 + input_height % 2,
      input_width / 2 + input_height % 2,
      input_depth,
    );

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
    // Pack the input and pool
    let to_pool = input.to_owned().into_iter().flatten().collect_vec();
    let (pooled_samples, bitmasks) = max_pool_packed(&to_pool);

    // Unpack the pooled results and bitmasks
    let input_depth = self.input_dimensions.2;
    let grouped_pooled_samples = pooled_samples
      .into_iter()
      .chunks(input_depth)
      .into_iter()
      .map(|sample| sample.into_iter().collect_vec())
      .collect_vec();

    let grouped_bitmasks = bitmasks
      .into_iter()
      .chunks(input_depth)
      .into_iter()
      .map(|sample| sample.into_iter().collect_vec())
      .collect_vec();

    self.prev_input_bm = grouped_bitmasks;
    return grouped_pooled_samples;
  }

  fn backpropogation(&mut self, error: &Vec<Vec<Matrix>>) -> Vec<Vec<Matrix>> {
    // Max pool is not differentiable, so we will just pass the error back to the max value
    // Element wise multiply max mask by nearest_neighbor upscaled error

    let odd_input_columns = self.input_dimensions.1 % 2 == 1;

    // Pack the error and bitmasks and upsample
    let to_upsample = error.to_owned().into_iter().flatten().collect_vec();
    let bitmasks = self
      .prev_input_bm
      .to_owned()
      .into_iter()
      .flatten()
      .collect_vec();
    let upsampled = nearest_neighbor_2x_upsample_packed(&to_upsample, odd_input_columns);
    let upsampled_errors = element_multiply_packed(&upsampled, &bitmasks, true);

    // Unpack the upsampled errors
    let input_depth = self.input_dimensions.2;
    let grouped_upsampled_errors = upsampled_errors
      .into_iter()
      .chunks(input_depth)
      .into_iter()
      .map(|sample| sample.into_iter().collect_vec())
      .collect_vec();

    return grouped_upsampled_errors;
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

  fn train(&mut self, input: &Vec<Vec<Matrix>>, labels: &Vec<f32>) -> Vec<Vec<Matrix>> {
    let flattened_input = self.flatten(input);
    let fc_error = self
      .fully_connected_layer
      .train_classification_observation_matrix(&flattened_input, labels);
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
