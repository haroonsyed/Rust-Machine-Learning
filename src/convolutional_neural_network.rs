use std::time::Instant;

use itertools::{izip, Itertools};
use tensor_lib::*;

use crate::{
  basic_neural_network::BasicNeuralNetworkRust,
  packed_optimizers::{PackedAdamOptimizer, PackedOptimizer},
};

pub struct ConvolutionalNeuralNetworkRust {
  pub input_dimensions: (usize, usize, usize), // Height, Width, Depth
  pub num_classifications: usize,
  layers: Vec<Box<dyn CnnLayer>>,
  fully_connected_layer: Option<FullyConnectedLayer>,
  optimizer: Box<dyn PackedOptimizer>,
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
      optimizer: Box::new(PackedAdamOptimizer::new(1e-2, 0.9, 0.999)), // Default optimizer
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
      .set_optimizer(self.optimizer.get_single_optimizer());
  }

  pub fn set_optimizer(&mut self, optimizer: Box<dyn PackedOptimizer>) {
    self.optimizer = optimizer;

    // TODO: Work out nice way to set optimizer after adding layers (although this is not really needed, but it would be nice)
    if self.layers.len() > 0 {
      panic!("Please set optimizer before adding layers!");
    }
  }

  pub fn classify(&mut self, observations_matrices: Vec<Vec<Matrix>>) -> Vec<f32> {
    if observations_matrices.len() == 0 {
      return Vec::new();
    }
    let observations_matrices = BatchData::new(observations_matrices);

    // Feed forward
    let filter_outputs = self.feed_forward(observations_matrices);

    // Check none
    if self.fully_connected_layer.is_none() {
      panic!("Fully Connected Layer Is Missing!");
    }

    // Classify
    return self
      .fully_connected_layer
      .as_mut()
      .unwrap()
      .classify(filter_outputs);
  }

  pub fn train(&mut self, observations_matrices: Vec<Vec<Matrix>>, encoded_labels: Matrix) {
    if observations_matrices.len() == 0 {
      return;
    }
    let observations_matrices = BatchData::new(observations_matrices);

    let start = Instant::now();

    // Feed forward
    let filter_outputs = self.feed_forward(observations_matrices);

    // Train through FC

    // Check none
    if self.fully_connected_layer.is_none() {
      panic!("Fully Connected Layer Is Missing!");
    }
    let fc_error = self
      .fully_connected_layer
      .as_mut()
      .unwrap()
      .train(filter_outputs, &encoded_labels);

    // Backpropogate
    self.backpropogation(fc_error);

    // println!("Total iteration time {:?}\n", Instant::now() - start)
  }

  //return Vec<Vec<Vec<Matrix>>> sample -> layer -> filters -> Matrix
  fn feed_forward(
    &mut self,
    observations: BatchData, // sample -> depth -> data
  ) -> BatchData {
    let mut prev_layer_output = observations;
    for layer in self.layers.iter_mut() {
      prev_layer_output = layer.feed_forward(prev_layer_output);
    }
    return prev_layer_output;
  }

  fn backpropogation(
    &mut self,
    unflattened_fc_error: BatchData, // sample -> depth -> data
  ) {
    let mut prev_layer_error = unflattened_fc_error;
    for layer in self.layers.iter_mut().rev() {
      prev_layer_error = layer.backpropogation(prev_layer_error);
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

struct BatchData {
  pub data: Vec<Matrix>,
  pub channel_count: usize,
  pub sample_count: usize,
}

impl BatchData {
  pub fn new(data: Vec<Vec<Matrix>>) -> Self {
    let sample_count = data.len();
    let channel_count = data[0].len();
    let data = data.into_iter().flatten().collect_vec();
    return Self {
      data,
      sample_count,
      channel_count,
    };
  }

  pub fn iter(&self) -> impl Iterator<Item = &[Matrix]> {
    self.data.chunks(self.channel_count)
  }
}

trait CnnLayer: Send {
  fn feed_forward(&mut self, input: BatchData) -> BatchData;
  fn backpropogation(&mut self, error: BatchData) -> BatchData;
  fn get_input_dimensions(&self) -> (usize, usize, usize);
  fn get_output_dimensions(&self) -> (usize, usize, usize);
}

struct ConvolutionalLayerRust {
  pub filters: Vec<Vec<Matrix>>, // Filter -> Depth
  pub biases: Vec<Matrix>,       // Filter
  pub packed_filter_optimizer: Box<dyn PackedOptimizer>, // Filter -> Depth
  pub packed_bias_optimizer: Box<dyn PackedOptimizer>, // Bias
  pub prev_input: BatchData,
  pub prev_output: BatchData,
  pub input_dimensions: (usize, usize, usize), // Height, Width, Depth
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
    optimizer: &Box<dyn PackedOptimizer>,
  ) -> Self {
    let input_dimensions = (input_height, input_width, input_depth);
    let output_dimensions = (
      input_height - filter_height + 1,
      input_width - filter_width + 1,
      filter_count,
    );

    let mut filters = Vec::new();
    let mut biases = Vec::new();
    let prev_input = BatchData {
      data: Vec::new(),
      sample_count: 0,
      channel_count: 0,
    };
    let prev_output = BatchData {
      data: Vec::new(),
      sample_count: 0,
      channel_count: 0,
    };

    // Create the filters
    for _ in 0..filter_count {
      let mut filter = Vec::new();
      for _ in 0..input_depth {
        filter.push(Matrix::new_random(
          0.0,
          f64::sqrt(2.0 / (filter_height * filter_width * input_depth) as f64), // He initialization
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
      packed_filter_optimizer: optimizer.clone(),
      packed_bias_optimizer: optimizer.clone(),
      prev_input,
      prev_output,
      input_dimensions,
      output_dimensions,
    };
  }
}

impl CnnLayer for ConvolutionalLayerRust {
  fn get_input_dimensions(&self) -> (usize, usize, usize) {
    return self.input_dimensions;
  }

  fn get_output_dimensions(&self) -> (usize, usize, usize) {
    return self.output_dimensions;
  }

  fn feed_forward(&mut self, input: BatchData) -> BatchData {
    // Record time here
    let start = Instant::now();

    let sample_count = input.sample_count;

    let vec_capacity = sample_count * self.filters.len() * self.filters[0].len();
    let mut channels_to_correlate = Vec::with_capacity(vec_capacity);
    let mut kernels_to_correlate = Vec::with_capacity(vec_capacity);
    let mut filter_result_index_to_sum_filter_result_to = Vec::with_capacity(vec_capacity);
    let mut filter_result_index_to_sum = Vec::with_capacity(vec_capacity);
    let mut filter_result_index_to_sum_bias_to =
      Vec::with_capacity(sample_count * self.filters.len());
    let mut biases_to_sum = Vec::with_capacity(sample_count * self.filters.len());

    let mut raw_filter_output_indices = Vec::with_capacity(sample_count * self.filters.len());

    // The below logic is used to setup the packed operation.
    // It follows from doing the operations non-packed (commented out below)
    // I understand the packed version is more complex, but it is much faster
    // Sample
    for sample in input.iter() {
      // Filter
      for (filter, bias) in izip!(self.filters.iter(), self.biases.iter()) {
        channels_to_correlate.push(sample[0].clone());
        kernels_to_correlate.push(filter[0].clone());
        let index_to_sum_to = channels_to_correlate.len() - 1;

        // Channel
        for (channel, kernel) in izip!(sample[1..].iter(), filter[1..].iter()) {
          channels_to_correlate.push(channel.clone());
          kernels_to_correlate.push(kernel.clone());

          filter_result_index_to_sum_filter_result_to.push(index_to_sum_to);
          filter_result_index_to_sum.push(channels_to_correlate.len() - 1);
        }

        // Add the bias
        filter_result_index_to_sum_bias_to.push(index_to_sum_to);
        biases_to_sum.push(bias.clone());

        raw_filter_output_indices.push(index_to_sum_to);
      }
    }

    let setup_overhead = start.elapsed();

    // correlate
    let correlated_channels = correlate_packed(
      &channels_to_correlate,
      &kernels_to_correlate,
      PaddingType::VALID,
    );

    let correlation_time = start.elapsed() - setup_overhead;

    // Sum
    let sum_to = filter_result_index_to_sum_filter_result_to
      .iter()
      .chain(filter_result_index_to_sum_bias_to.iter())
      .map(|index| correlated_channels[*index].clone())
      .collect_vec();

    let to_add = filter_result_index_to_sum
      .iter()
      .map(|index| correlated_channels[*index].clone())
      .chain(biases_to_sum.into_iter())
      .collect_vec();

    let raw_filter_outputs = raw_filter_output_indices
      .iter()
      .map(|index| correlated_channels[*index].clone())
      .collect_vec();

    let pure_element_add_packed = Instant::now();
    element_add_packed_inplace(&sum_to, &to_add);
    let pure_element_packed_add_time = pure_element_add_packed.elapsed();
    let sum_time = start.elapsed() - correlation_time - setup_overhead;

    // ReLU
    element_ReLU_packed_inplace(&raw_filter_outputs);

    let relu_time = start.elapsed() - sum_time - correlation_time - setup_overhead;

    self.prev_input = input;
    self.prev_output = BatchData {
      data: raw_filter_outputs.clone(),
      sample_count,
      channel_count: self.filters.len(),
    };

    let total_time = start.elapsed();
    // println!("Time to setup packed operations: {:?}", setup_overhead);
    // println!("Time to correlate: {:?}", correlation_time);
    // println!(
    //   "Pure element add packed: {:?}",
    //   pure_element_packed_add_time
    // );
    // println!("Time to sum: {:?}", sum_time);
    // println!("Time to relu: {:?}", relu_time);
    // println!("Total feed forward time: {:?}", total_time);

    return BatchData {
      data: raw_filter_outputs,
      sample_count,
      channel_count: self.filters.len(),
    };
  }

  fn backpropogation(&mut self, sample_output_errors: BatchData) -> BatchData {
    let sample_count = sample_output_errors.sample_count;

    // n is the filter
    // m is the channel

    // Apply relu prime

    let start = Instant::now();

    element_ReLU_prime_packed_inplace(&self.prev_output.data);
    element_multiply_packed_inplace(&sample_output_errors.data, &self.prev_output.data);

    let relu_prime_time = start.elapsed();

    let vec_capacity = sample_count * self.filters.len() * self.filters[0].len();

    // Delta Input Packed
    let mut delta_input_output_errors = Vec::with_capacity(vec_capacity);
    let mut delta_input_filters = Vec::with_capacity(vec_capacity);
    let mut delta_input_to_sum_to = Vec::with_capacity(vec_capacity);
    let mut delta_input_to_sum = Vec::with_capacity(vec_capacity);

    // Bias Update Packed
    let bias_gradients = &sample_output_errors.data;

    // Filter Update Packed
    let mut delta_filter_prev_input = Vec::with_capacity(vec_capacity);
    let mut delta_filter_output_error = Vec::with_capacity(vec_capacity);

    // PER SAMPLE
    let mut current_sample_index = 0;
    for (sample_output_error, sample_prev_input) in
      izip!(sample_output_errors.iter(), self.prev_input.iter())
    {
      // Calculate the input error
      // PER FILTER
      for (filter_error, filter) in izip!(sample_output_error.iter(), self.filters.iter()) {
        // deltaXm = sum(de/dy * conv_full * Knm)
        let sum_to_offset = current_sample_index * self.input_dimensions.2;

        // PER CHANNEL
        for (channel_index, channel) in filter.iter().enumerate() {
          delta_input_output_errors.push(filter_error.clone());
          delta_input_filters.push(channel.clone());
          delta_input_to_sum_to.push(sum_to_offset + channel_index);
          delta_input_to_sum.push(delta_input_output_errors.len() - 1);
        }
      }

      // Update the filters
      // PER FILTER
      for (filter_output_error, filter) in izip!(sample_output_error, self.filters.iter()) {
        // PER CHANNEL
        for (prev_channel_input, channel) in izip!(sample_prev_input, filter) {
          // Knm' = Knm - learning_rate * Xm * conv_valid * de/dy
          delta_filter_prev_input.push(prev_channel_input.clone());
          delta_filter_output_error.push(filter_output_error.clone());
        }
      }

      current_sample_index += 1;
    }

    let backprop_setup_time = start.elapsed();

    ///////////////////
    /// DELTA BIASES
    ///////////////////
    // b' = b - de/dy * learning_rate
    let bias_steps = self
      .packed_bias_optimizer
      .calculate_steps(bias_gradients, sample_count);

    element_subtract_packed_inplace(&self.biases, &bias_steps);

    let bias_update_time = start.elapsed() - backprop_setup_time;

    ///////////////////
    /// DELTA FILTERS
    ///////////////////
    // Knm' = Knm - learning_rate * Xm * conv_valid * de/dy
    let channel_gradients = correlate_packed(
      &delta_filter_prev_input,
      &delta_filter_output_error,
      PaddingType::VALID,
    );
    let correlation_time = start.elapsed() - bias_update_time - backprop_setup_time;

    let filter_steps = self
      .packed_filter_optimizer
      .calculate_steps(&channel_gradients, sample_count);

    let filter_step_time =
      start.elapsed() - correlation_time - bias_update_time - backprop_setup_time;

    let flattened_filters = self.filters.to_owned().into_iter().flatten().collect_vec();
    let flattened_filter_time = start.elapsed()
      - filter_step_time
      - correlation_time
      - bias_update_time
      - backprop_setup_time;
    element_subtract_packed_inplace(&flattened_filters, &filter_steps);
    let filter_update_time = start.elapsed()
      - flattened_filter_time
      - filter_step_time
      - correlation_time
      - bias_update_time
      - backprop_setup_time;

    // Print out all the filter update times
    // println!("correlation_time: {:?}", correlation_time);
    // println!("filter_step_time: {:?}", filter_step_time);
    // println!("flattened_filter_time: {:?}", flattened_filter_time);
    // println!("filter_update_time: {:?}", filter_update_time);

    let filter_update_time = start.elapsed() - bias_update_time - backprop_setup_time;

    ///////////////////
    /// DELTA INPUTS
    ///////////////////
    // deltaXm = sum(de/dy * conv_full * Knm)
    let delta_input_convolutions = convolve_packed(
      &delta_input_output_errors,
      &delta_input_filters,
      PaddingType::FULL,
    );

    let delta_input_sum_to = delta_input_to_sum_to
      .iter()
      .map(|&index| delta_input_convolutions[index].clone())
      .collect_vec();
    let delta_input_to_sum = delta_input_to_sum
      .iter()
      .map(|&index| delta_input_convolutions[index].clone())
      .collect_vec();
    element_add_packed_inplace(&delta_input_sum_to, &delta_input_to_sum);

    let sample_input_errors = delta_input_to_sum_to
      .into_iter()
      .unique()
      .map(|index| delta_input_convolutions[index].clone())
      .collect_vec();

    let delta_input_time =
      start.elapsed() - filter_update_time - bias_update_time - backprop_setup_time;

    let total_time = start.elapsed();
    // println!("Time to setup packed operations: {:?}", backprop_setup_time);
    // println!("Time to relu prime: {:?}", relu_prime_time);
    // println!("Time to bias update: {:?}", bias_update_time);
    // println!("Time to filter update: {:?}", filter_update_time);
    // println!("Time to delta input: {:?}", delta_input_time);
    // println!("Total backprop time: {:?}", total_time);

    return BatchData {
      data: sample_input_errors,
      sample_count,
      channel_count: self.input_dimensions.2,
    };
  }
}

struct MaxPoolLayerRust {
  pub prev_input_bm: BatchData,
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
      prev_input_bm: BatchData {
        data: Vec::new(),
        sample_count: 0,
        channel_count: 0,
      },
      input_dimensions,
      output_dimensions,
    };
  }
}

impl CnnLayer for MaxPoolLayerRust {
  fn get_input_dimensions(&self) -> (usize, usize, usize) {
    return self.input_dimensions;
  }

  fn get_output_dimensions(&self) -> (usize, usize, usize) {
    return self.output_dimensions;
  }

  fn feed_forward(&mut self, input: BatchData) -> BatchData {
    let start = Instant::now();

    // Pack the input and pool
    let to_pool = &input.data;
    let (pooled_samples, bitmasks) = max_pool_packed(to_pool);
    let pool_time = start.elapsed();

    // println!("Time to pool: {:?}", pool_time);

    self.prev_input_bm = BatchData {
      data: bitmasks,
      sample_count: input.sample_count,
      channel_count: input.channel_count,
    };
    return BatchData {
      data: pooled_samples,
      sample_count: input.sample_count,
      channel_count: input.channel_count,
    };
  }

  fn backpropogation(&mut self, error: BatchData) -> BatchData {
    let start = Instant::now();
    // Max pool is not differentiable, so we will just pass the error back to the max value
    // Element wise multiply max mask by nearest_neighbor upscaled error

    let odd_input_columns = self.input_dimensions.1 % 2 == 1;

    // Pack the error and bitmasks and upsample
    let to_upsample = &error.data;
    let bitmasks = &self.prev_input_bm.data;
    let upsampled = nearest_neighbor_2x_upsample_packed(to_upsample, odd_input_columns);
    element_multiply_packed_inplace(&upsampled, bitmasks);

    let total_time = start.elapsed();
    // println!("Time to upsample: {:?}", total_time);

    return BatchData {
      data: upsampled,
      sample_count: error.sample_count,
      channel_count: error.channel_count,
    };
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

  fn classify(&mut self, input: BatchData) -> Vec<f32> {
    let flattened_input = self.flatten(input);
    return self.fully_connected_layer.classify_matrix(&flattened_input);
  }

  fn train(&mut self, input: BatchData, encoded_labels: &Matrix) -> BatchData {
    let start = Instant::now();
    let flattened_input = self.flatten(input);
    let fc_error = self
      .fully_connected_layer
      .train_classification_observation_matrix(&flattened_input, encoded_labels);
    let error = self.unflatten(fc_error);

    let total_time = start.elapsed();
    // println!("Time to train FC: {:?}", total_time);
    return error;
  }

  fn flatten(&mut self, input: BatchData) -> Matrix {
    let mut flattened = flatten_matrix_array(&input.data);

    // Take tranpose to ensure each column is a sample
    flattened.reshape(
      input.sample_count,
      flattened.get_data_length() / input.sample_count,
    );

    // Now take tranpose
    flattened = flattened.transpose();

    return flattened;
  }

  fn unflatten(&mut self, fc_error: Matrix) -> BatchData {
    // The fc_error is with respect to classification neurons, not input neurons to fc layer
    // So we will quickly do that here (adding it to Basic Neural Network was clunky)
    let mut fc_error = self.fully_connected_layer.weights[0]
      .transpose()
      .matrix_multiply(&fc_error);

    // Now let's take the transpose of that to make observations the rows
    fc_error = fc_error.transpose();

    // Now unflatten the output error row by row back to input shape
    // Because I decided not to treat each sample separately to FC I need to double unflatten (sample -> depth)
    let sample_errors = unflatten_array_strided_to_matrices(&fc_error, 1, fc_error.get_columns());
    let (input_height, input_width, channel_count) = self.input_dimensions;
    let unflattened_errors = sample_errors
      .iter()
      .flat_map(|sample_error| {
        unflatten_array_strided_to_matrices(&sample_error, input_height, input_width)
      })
      .collect_vec();

    let sample_count = unflattened_errors.len() / channel_count;
    return BatchData {
      data: unflattened_errors,
      sample_count,
      channel_count,
    };
  }
}
