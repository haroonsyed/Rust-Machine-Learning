use std::collections::HashMap;

use itertools::{izip, Itertools};
use pyo3::prelude::*;
use tensor_lib::*;

use crate::{basic_neural_network::BasicNeuralNetworkRust, image_util::ImageBatchLoaderRust};

#[pyclass]
pub struct SimplifiedConvolutionalNeuralNetwork {
  pub network: SimplifiedConvolutionalNeuralNetworkRust,
  batch_loader: Option<ImageBatchLoaderRust>,
}

#[pymethods]
impl SimplifiedConvolutionalNeuralNetwork {
  #[new]
  fn new(
    num_classifications: usize,
    input_width: usize,
    input_height: usize,
    input_depth: usize,
    filters_per_conv_layer: Vec<usize>,
    filter_dimension: usize, // Must be odd
  ) -> Self {
    let network = SimplifiedConvolutionalNeuralNetworkRust::new(
      num_classifications,
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

// This is going to have no bias, no relU, and no max pooling
pub struct SimplifiedConvolutionalNeuralNetworkRust {
  pub num_classifications: usize,
  pub input_width: usize,
  pub input_height: usize,
  pub input_depth: usize,
  pub conv_layers: Vec<Tensor>, // Layer -> Filter -> Depth
  pub fully_connected_layer: BasicNeuralNetworkRust,
  pub filter_output_dimensions_per_layer: Vec<(usize, usize)>, // Layer -> (width, height)
}

impl SimplifiedConvolutionalNeuralNetworkRust {
  pub fn new(
    num_classifications: usize,
    input_width: usize,
    input_height: usize,
    input_depth: usize,
    filters_per_conv_layer: Vec<usize>,
    filter_dimension: usize, // Must be odd
  ) -> Self {
    // Calculate the output dimensions of each layer
    let mut filter_output_dimensions_per_layer = Vec::new();
    for layer_index in 0..filters_per_conv_layer.len() {
      let prev_out_dimensions = if layer_index == 0 {
        (input_width, input_height)
      } else {
        filter_output_dimensions_per_layer[layer_index - 1]
      };

      // This output_dimension = (input - filter + 1) assuming valid convolutions
      let output_dimension_width = prev_out_dimensions.0 - filter_dimension + 1;
      let output_dimension_height = prev_out_dimensions.1 - filter_dimension + 1;

      filter_output_dimensions_per_layer.push((output_dimension_width, output_dimension_height));
    }

    // Calculate the depth of the filters per layer
    let mut filter_depth_per_conv_layer = filters_per_conv_layer.clone();
    filter_depth_per_conv_layer.insert(0, input_depth);

    // Create the kernel weights
    let conv_layers = izip!(
      filters_per_conv_layer.iter(),
      filter_depth_per_conv_layer.iter()
    )
    .map(|(&filter_count, &filter_depth)| {
      Tensor::random(
        vec![
          filter_count,
          filter_depth,
          filter_dimension,
          filter_dimension,
        ],
        0.0,
        0.68,
      )
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
    let network = SimplifiedConvolutionalNeuralNetworkRust {
      num_classifications,
      input_width,
      input_height,
      input_depth,
      conv_layers,
      fully_connected_layer,
      filter_output_dimensions_per_layer,
    };
    return network;
  }

  fn create_tensor_from_input(&self, input: &Vec<Vec<Vec<f32>>>) -> Vec<Tensor> {
    // Map each sample to a tensor
    return input
      .iter()
      .map(|sample| {
        let matrices = sample
          .iter()
          .map(|channel_data| Matrix::new_1d(channel_data, self.input_height, self.input_width))
          .collect_vec();
        return Tensor::from_matrices(
          &matrices,
          vec![self.input_depth, self.input_height, self.input_width],
        );
      })
      .collect_vec();
  }

  pub fn classify(&mut self, features_test: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    let observations = self.create_tensor_from_input(features_test);

    // Feed forward
    let filter_outputs = self.feed_forward(&observations);

    // Linearize the final output
    let flattened_sample_outputs = filter_outputs
      .iter()
      .map(|sample_output| {
        let last_layer_output = sample_output.last().unwrap();
        let mut flattened = last_layer_output.flatten();

        // Take the transpose for fully connected layer
        flattened = flattened.transpose();
        return flattened;
      })
      .collect_vec();

    let classifications = flattened_sample_outputs
      .iter()
      .map(|flattened_output| {
        self
          .fully_connected_layer
          .classify_matrix(&flattened_output.get_data())[0]
      })
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
    let observations = self.create_tensor_from_input(observations);

    // Feed forward
    let filter_outputs = self.feed_forward(&observations);

    // Linearize the final output
    let flattened_sample_outputs = filter_outputs
      .iter()
      .map(|sample_output| {
        let last_layer_output = sample_output.last().unwrap();
        let mut flattened = last_layer_output.flatten();

        // Take the transpose for fully connected layer
        flattened = flattened.transpose();
        return flattened;
      })
      .collect_vec();

    // Feed to fully connected layer
    let sample_errors = izip!(flattened_sample_outputs.iter(), labels.iter())
      .map(|(flattened_output, label)| {
        let fully_connected_error = self
          .fully_connected_layer
          .train_classification_observation_matrix(
            &flattened_output.get_data(),
            &vec![*label],
            learning_rate,
          );

        let output_error = self.fully_connected_layer.weights[0]
          .transpose()
          .matrix_multiply(&fully_connected_error);

        let (output_height, output_width) = self.filter_output_dimensions_per_layer.last().unwrap();

        // Now unflatten the output error
        let unflattened_error =
          unflatten_array_strided_to_matrices(&output_error, *output_height, *output_width);

        return Tensor::from_matrices(
          &unflattened_error,
          vec![unflattened_error.len(), *output_height, *output_width],
        );
      })
      .collect_vec();

    // Backpropogate conv layers
    self.backpropogation_hidden_layer(
      &observations,
      &filter_outputs,
      &sample_errors,
      learning_rate,
      self.conv_layers.len() - 1,
    );
  }

  //return Vec<Vec<Tensor>> sample -> layer -> filter outputs
  pub fn feed_forward(
    &self,
    observations: &Vec<Tensor>, // sample -> depth -> data
  ) -> Vec<Vec<Tensor>> {
    let mut filter_outputs = Vec::new();

    // Sample
    for sample in observations {
      let mut sample_outputs = Vec::new();
      let mut current_layer_input = sample;

      // Layer
      for (layer_index, layer_filters) in izip!(0..self.conv_layers.len(), self.conv_layers.iter())
      {
        let mut layer_outputs = Vec::new();

        // Filter
        for filter in layer_filters.iter() {
          let filter_output = current_layer_input.convolution(filter, ConvolutionType::VALID);

          layer_outputs.push(filter_output);
        }

        let layer_outputs_tensor = Tensor::from_children(layer_outputs);
        sample_outputs.push(layer_outputs_tensor);
        current_layer_input = sample_outputs.last().unwrap();
      }

      filter_outputs.push(sample_outputs);
    }

    return filter_outputs;
  }

  pub fn rotate_filters_180(filter: &Tensor) -> Tensor {
    if filter.is_leaf() {
      let rotated_filter = filter.get_data().rotate_180();
      return Tensor::from_matrices(&vec![rotated_filter], filter.dimensions.clone());
    }

    if filter.get_rank() != 3 {
      panic!("Filter must be rank 2 or 3");
    }

    let mut rotated_filters = Vec::new();
    for channel in filter.iter() {
      rotated_filters.push(channel.get_data().rotate_180());
    }
    return Tensor::from_matrices(&rotated_filters, filter.dimensions.clone());
  }

  pub fn backpropogation_hidden_layer(
    &mut self,
    observations: &Vec<Tensor>,        // sample -> depth -> data
    filter_outputs: &Vec<Vec<Tensor>>, // sample -> layer -> filter -> data
    next_layer_error: &Vec<Tensor>,    // sample -> filter -> data
    learning_rate: f32,
    layer: usize,
  ) {
    // n is the filter
    // m is the channel
    let mut layer_error = Vec::new();

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

        // We can use 3D convolutions with tensors
        // First we have to duplicate the error for each channel
        let next_error = Tensor::from_children(vec![next_error.clone(); filter.dimensions[0]]);

        let rotated_filter = Self::rotate_filters_180(filter);
        let delta_xm = next_error.convolution(&rotated_filter, ConvolutionType::FULL);

        return delta_xm;
      })
      .collect_vec();

      let sample_error = Tensor::from_children(sample_error);
      layer_error.push(sample_error);

      // Update the kernels
      // PER FILTER
      for (filter, error) in izip!(
        self.conv_layers[layer].iter(),
        next_layer_sample_error.iter()
      ) {
        // PER DEPTH
        for (channel, prev_channel_output) in izip!(filter.iter(), prev_layer_outputs.iter()) {
          // Here the error is per channel. We cannot sum across the volume of filter.
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
