use std::collections::HashMap;

use itertools::{izip, Itertools};
use matrix_lib::*;
use pyo3::prelude::*;

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
}

pub struct ConvolutionalNeuralNetworkRust {
  pub num_classifications: usize,
  pub max_pool_stride: usize, // Set to 0 to do no max pooling
  pub input_width: usize,
  pub input_height: usize,
  pub input_depth: usize,
  pub conv_layers: Vec<Vec<Vec<Matrix>>>, // Layer -> Filter -> Depth
  pub biases: Vec<Vec<Matrix>>,           // Layer -> Filter -> Bias
  pub fully_connected_layer: BasicNeuralNetworkRust,
  filter_output_dimensions_per_layer: Vec<(usize, usize)>, // Layer -> (width, height)
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
    // Calculate the output dimensions of each layer
    let mut filter_output_dimensions_per_layer = Vec::new();
    for layer_index in 0..filters_per_conv_layer.len() {
      let is_max_pool_layer = if max_pool_stride == 0 {
        false
      } else {
        layer_index % max_pool_stride == 0
      };

      let prev_out_dimensions = if layer_index == 0 {
        (input_width, input_height)
      } else {
        filter_output_dimensions_per_layer[layer_index - 1]
      };

      // This output_dimension = (input - filter + 1) assuming valid convolutions
      let mut output_dimension_width = prev_out_dimensions.0 - filter_dimension + 1;
      let mut output_dimension_height = prev_out_dimensions.1 - filter_dimension + 1;

      if is_max_pool_layer {
        output_dimension_width = output_dimension_width / 2;
        output_dimension_height = output_dimension_height / 2;
      }
      filter_output_dimensions_per_layer.push((output_dimension_height, output_dimension_width));
    }

    // Calculate the depth of the filters per layer
    let mut filter_depth_per_conv_layer = filters_per_conv_layer.clone();
    filter_depth_per_conv_layer.insert(0, input_depth);

    // Create the untied biases
    let biases = izip!(
      filter_output_dimensions_per_layer.iter(),
      filters_per_conv_layer.iter()
    )
    .map(|(filter_output_dimensions, filter_count)| {
      let (rows, columns) = filter_output_dimensions;
      (0..*filter_count)
        .map(|_| Matrix::zeros(*rows, *columns))
        .collect_vec()
    })
    .collect_vec();

    // Create the kernel weights
    let conv_layers = izip!(
      filters_per_conv_layer.iter(),
      filter_depth_per_conv_layer.iter()
    )
    .map(|(&filter_count, &filter_depth)| {
      (0..filter_count)
        .map(|_| {
          (0..filter_depth)
            .map(|_| {
              Matrix::new_random(
                0.0,
                f64::sqrt(2.0 / (filter_dimension * filter_dimension) as f64),
                filter_dimension,
                filter_dimension,
              )
            })
            .collect_vec()
        })
        .collect_vec()
    })
    .collect_vec();

    // Create the fully connected layer
    let (final_filtered_height, final_filtered_width) =
      filter_output_dimensions_per_layer.last().unwrap();
    let num_features =
      (final_filtered_width * final_filtered_height) * filter_depth_per_conv_layer.last().unwrap();
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
      filter_output_dimensions_per_layer,
    };
    return network;
  }

  pub fn classify(&mut self, features_test: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    // Obervations are of shape sample -> depth -> pixels
    let observations_matrices = features_test
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

    // Linearize the final output
    let flattened_sample_outputs = filter_outputs
      .iter()
      .map(|sample_output| {
        let mut flattened = flatten_matrix_array(sample_output.last().unwrap());

        // Take the transpose for fully connected layer
        flattened.reshape(flattened.columns, flattened.rows);
        return flattened;
      })
      .collect_vec();

    let classifications = flattened_sample_outputs
      .iter()
      .map(|flattened_output| self.fully_connected_layer.classify_matrix(flattened_output)[0])
      .collect_vec();

    return classifications;
  }

  pub fn train(
    &mut self,
    observations: &Vec<Vec<Vec<f32>>>, // sample -> depth -> pixels
    labels: &Vec<f32>,
    learning_rate: f32,
  ) {
    // Convert observations to matrices
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

    // Linearize the final output
    let flattened_sample_outputs = filter_outputs
      .iter()
      .map(|sample_output| {
        let mut flattened = flatten_matrix_array(sample_output.last().unwrap());

        // Take the transpose for fully connected layer
        flattened.reshape(flattened.columns, flattened.rows);
        return flattened;
      })
      .collect_vec();

    // Feed to fully connected layer
    let sample_errors = izip!(flattened_sample_outputs.iter(), labels.iter())
      .map(|(flattened_output, label)| {
        let fully_connected_error = self
          .fully_connected_layer
          .train_classification_observation_matrix(flattened_output, &vec![*label], learning_rate);

        let output_error = self.fully_connected_layer.weights[0]
          .transpose()
          .matrix_multiply(&fully_connected_error);

        let (output_height, output_width) = self.filter_output_dimensions_per_layer.last().unwrap();

        // Now unflatten the output error
        return unflatten_array_strided_to_matrices(&output_error, *output_height, *output_width);
      })
      .collect_vec();

    // Backpropogate conv layers
    self.backpropogation_hidden_layer(
      &observations_matrices,
      &filter_outputs,
      &sample_errors,
      learning_rate,
      self.conv_layers.len() - 1,
    );
  }

  //return Vec<Vec<Vec<Matrix>>> sample -> layer -> filters -> Matrix
  pub fn feed_forward(
    &self,
    observations: &Vec<Vec<Matrix>>, // sample -> depth -> data
  ) -> Vec<Vec<Vec<Matrix>>> {
    let mut filter_outputs = Vec::new();

    // Sample
    for sample in observations {
      let mut sample_outputs = Vec::new();
      let mut current_layer_input = sample;

      // Layer
      for (layer_index, layer_filters, layer_biases) in izip!(
        0..self.conv_layers.len(),
        self.conv_layers.iter(),
        self.biases.iter(),
      ) {
        let is_max_pool_layer =
          self.max_pool_stride != 0 && layer_index % self.max_pool_stride == 0;
        let mut layer_outputs = Vec::new();

        // Filter
        for (filter, bias) in izip!(layer_filters, layer_biases) {
          let (output_height, output_width) = self.filter_output_dimensions_per_layer[layer_index];
          let mut filter_output = Matrix::zeros(output_height, output_width);

          // Channel
          for (channel, kernel) in izip!(current_layer_input, filter) {
            let channel_output = channel.convolution(kernel, ConvolutionType::VALID);
            filter_output.element_add_inplace(&channel_output);
          }

          // Add the bias
          filter_output.element_add_inplace(bias);

          // relu
          filter_output.element_ReLU_inplace();

          // Pool the output
          if is_max_pool_layer {
            filter_output = filter_output.max_pool();
          }

          layer_outputs.push(filter_output);
        }

        sample_outputs.push(layer_outputs);
        current_layer_input = sample_outputs.last().unwrap();
      }

      filter_outputs.push(sample_outputs);
    }

    return filter_outputs;
  }

  pub fn backpropogation_hidden_layer(
    &mut self,
    observations: &Vec<Vec<Matrix>>,        // sample -> depth -> data
    filter_outputs: &Vec<Vec<Vec<Matrix>>>, // sample -> layer -> filter -> data
    next_layer_error: &Vec<Vec<Matrix>>,    // sample -> filter -> data
    learning_rate: f32,
    layer: usize,
  ) {
    // n is the filter
    // m is the channel
    let mut layer_error = Vec::new();

    // Apply relu_prime
    for sample_error in next_layer_error {
      for filter_error in sample_error {
        filter_error.element_ReLU_prime_inplace();
      }
    }

    // PER SAMPLE
    for (index, next_layer_sample_error) in next_layer_error.iter().enumerate() {
      let prev_layer_outputs = if layer == 0 {
        &observations[index]
      } else {
        &filter_outputs[index][layer - 1]
      };

      // Calculate error for this layer
      // PER FILTER
      let sample_error = izip!(
        self.conv_layers[layer].iter(),
        next_layer_sample_error.iter()
      )
      .map(|(filter, next_error)| {
        // Xm' = Xm - sum(de/dy * conv_full * Knm)

        // PER DEPTH
        let delta_xm = next_error.convolution(&filter[0].rotate_180(), ConvolutionType::FULL);
        filter.iter().enumerate().for_each(|(index, channel)| {
          if index > 0 {
            delta_xm.element_add_inplace(
              &next_error.convolution(&channel.rotate_180(), ConvolutionType::FULL),
            );
          }
        });
        return delta_xm;
      })
      .collect_vec();

      layer_error.push(sample_error);

      // Update the biases
      // PER FILTER
      izip!(self.biases[layer].iter(), next_layer_sample_error.iter()).for_each(|(bias, error)| {
        // b' = b - de/dy * learning_rate
        bias.element_subtract_inplace(&error.scalar_multiply(learning_rate));
      });

      // Update the kernels
      // PER FILTER
      for (filter, error) in izip!(
        self.conv_layers[layer].iter(),
        next_layer_sample_error.iter()
      ) {
        // PER DEPTH
        for (channel, prev_channel_output) in izip!(filter.iter(), prev_layer_outputs.iter()) {
          // Knm' = Knm - learning_rate * Xm * conv_valid * de/dy
          let delta_channel = prev_channel_output.convolution(error, ConvolutionType::VALID);
          channel.element_subtract_inplace(&delta_channel.scalar_multiply(learning_rate));
        }
      }
    }

    if layer != 0 {
      self.backpropogation_hidden_layer(
        observations,
        filter_outputs,
        &layer_error,
        learning_rate,
        layer - 1,
      );
    }
  }
}
