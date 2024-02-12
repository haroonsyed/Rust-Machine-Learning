use std::time::Instant;

use itertools::{izip, Itertools};
use tensor_lib::{
  cuda_bindings::{cuda_cnn_back_propogate, cuda_cnn_feed_forward, cuda_synchronize},
  *,
};

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

  pub fn train(&mut self, observations_matrices: Vec<Vec<Matrix>>, labels: Vec<f32>) {
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
      .train(filter_outputs, &labels);

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
    let start = Instant::now();

    let sample_count = input.sample_count;
    let filter_count = self.filters.len();

    let flattened_filters = self.filters.to_owned().into_iter().flatten().collect_vec();

    let mut raw_filter_outputs = Vec::with_capacity(sample_count * filter_count);
    unsafe {
      raw_filter_outputs.set_len(sample_count * filter_count);

      cuda_cnn_feed_forward(
        input.data.as_ptr(),
        flattened_filters.as_ptr(),
        self.biases.as_ptr(),
        input.channel_count,
        sample_count,
        filter_count,
        raw_filter_outputs.as_mut_ptr(),
      );
    }

    let time_to_setup_call_ff_kernel = start.elapsed();

    self.prev_input = input;
    self.prev_output = BatchData {
      data: raw_filter_outputs.clone(),
      sample_count,
      channel_count: self.filters.len(),
    };

    let total_time = start.elapsed();
    // println!(
    //   "Time to setup and call feed forward kernel: {:?}",
    //   time_to_setup_call_ff_kernel
    // );
    // println!("Total feed forward time: {:?}", total_time);

    return BatchData {
      data: raw_filter_outputs,
      sample_count,
      channel_count: filter_count,
    };
  }

  fn backpropogation(&mut self, sample_output_errors: BatchData) -> BatchData {
    let sample_count = sample_output_errors.sample_count;
    let filter_count = self.filters.len();
    let input_depth = self.input_dimensions.2;
    let normalization_factor = 1.0 / (sample_count as f32);

    let start = Instant::now();

    // Apply relu prime
    element_ReLU_prime_packed_inplace(&self.prev_output.data);
    element_multiply_packed_inplace(&sample_output_errors.data, &self.prev_output.data);

    let relu_prime_time = start.elapsed();

    let flattened_filters = self.filters.to_owned().into_iter().flatten().collect_vec();

    let mut delta_bias = Vec::with_capacity(filter_count);
    let mut delta_filter = Vec::with_capacity(filter_count * input_depth);
    let mut delta_input = Vec::with_capacity(sample_count * input_depth);
    unsafe {
      delta_bias.set_len(filter_count);
      delta_filter.set_len(filter_count * input_depth);
      delta_input.set_len(sample_count * input_depth);
      cuda_cnn_back_propogate(
        sample_output_errors.data.as_ptr(),
        self.prev_input.data.as_ptr(),
        flattened_filters.as_ptr(),
        sample_count,
        filter_count,
        input_depth,
        delta_bias.as_mut_ptr(),
        delta_filter.as_mut_ptr(),
        delta_input.as_mut_ptr(),
      );
    }

    let time_to_setup_call_bp_kernel = start.elapsed() - relu_prime_time;

    // Actually update weights, use optimizer (can remove sample count now or set to 1 for now)
    let db = self
      .packed_bias_optimizer
      .calculate_steps(&delta_bias, normalization_factor);
    let dK = self
      .packed_filter_optimizer
      .calculate_steps(&delta_filter, normalization_factor);

    element_subtract_packed_inplace(&self.biases, &db);
    element_subtract_packed_inplace(&flattened_filters, &dK);

    let weight_update_time = start.elapsed() - time_to_setup_call_bp_kernel - relu_prime_time;

    let total_time = start.elapsed();
    // println!("Time to relu prime: {:?}", relu_prime_time);
    // println!(
    //   "Time to setup and call backpropogation kernel: {:?}",
    //   time_to_setup_call_bp_kernel
    // );
    // println!("Time to weight update: {:?}", weight_update_time);
    // println!("Total backprop time: {:?}", total_time);

    return BatchData {
      data: delta_input,
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
      input_width / 2 + input_width % 2,
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

  fn train(&mut self, input: BatchData, labels: &Vec<f32>) -> BatchData {
    let start = Instant::now();
    let flattened_input = self.flatten(input);
    let fc_error = self
      .fully_connected_layer
      .train_classification_observation_matrix(&flattened_input, labels);
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
    // Now let's take the transpose of that to make observations the rows
    let fc_error = fc_error.transpose();

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
