// use crate::py_util::py_print;
use itertools::{izip, Itertools};
use pyo3::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution};
use tensor_lib::{cuda_bindings::cuda_synchronize, Matrix};

use crate::optimizers::{
  AdagradOptimizer, AdamOptimizer, MomentumOptimizer, Optimizer, RMSPropOptimizer,
  StochasticGradientDescentOptimizer,
};

#[pyclass]
pub struct BasicNeuralNetwork {
  pub network: BasicNeuralNetworkRust,
}

#[pymethods]
impl BasicNeuralNetwork {
  #[new]
  fn new(
    hidden_layer_sizes: Vec<usize>,
    num_features: usize,
    num_classifications: usize, // Set to anything 1 to perform regression
  ) -> Self {
    let network =
      BasicNeuralNetworkRust::new(hidden_layer_sizes, num_features, num_classifications);

    // Cleanup and return
    return Self { network };
  }

  fn set_optimizer_stochastic_gradient_descent(&mut self, learning_rate: f32) {
    let optimizer = Box::new(StochasticGradientDescentOptimizer::new(learning_rate));
    self.network.set_optimizer(optimizer);
  }

  fn set_optimizer_momentum(&mut self, learning_rate: f32, beta: f32) {
    let optimizer = Box::new(MomentumOptimizer::new(learning_rate, beta));
    self.network.set_optimizer(optimizer);
  }

  fn set_optimizer_adagrad(&mut self, learning_rate: f32) {
    let optimizer = Box::new(AdagradOptimizer::new(learning_rate));
    self.network.set_optimizer(optimizer);
  }

  fn set_optimizer_RMSProp(&mut self, learning_rate: f32, beta: f32) {
    let optimizer = Box::new(RMSPropOptimizer::new(learning_rate, beta));
    self.network.set_optimizer(optimizer);
  }

  fn set_optimizer_adam(&mut self, learning_rate: f32, beta1: f32, beta2: f32) {
    let optimizer = Box::new(AdamOptimizer::new(learning_rate, beta1, beta2));
    self.network.set_optimizer(optimizer);
  }

  fn train(
    &mut self,
    features_train: Vec<Vec<f32>>,
    input_labels: Vec<f32>,
    num_iterations: usize,
    batch_size: usize,
  ) {
    self
      .network
      .train(features_train, input_labels, num_iterations, batch_size);
    unsafe { cuda_synchronize() }
  }

  fn classify(&mut self, features_test: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let classifications = self.network.classify(&features_test);

    return Ok(classifications);
  }

  fn regression(&mut self, features_test: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let predictions = self.network.regression(&features_test);
    unsafe { cuda_synchronize() }

    return Ok(predictions);
  }

  fn get_weights(&self) -> PyResult<Vec<Vec<Vec<f32>>>> {
    let mut weights = Vec::new();
    self
      .network
      .weights
      .iter()
      .for_each(|a| weights.push(a.get_data()));
    return Ok(weights);
  }

  fn get_biases(&self) -> PyResult<Vec<Vec<Vec<f32>>>> {
    let mut biases = Vec::new();
    self
      .network
      .biases
      .iter()
      .for_each(|a| biases.push(a.get_data()));
    return Ok(biases);
  }

  fn set_collect_performance_info(&mut self, collect_performance_info: bool) {
    self
      .network
      .set_collect_performance_info(collect_performance_info);
  }

  fn get_performance_info(&self) -> PyResult<Vec<(f32, f32)>> {
    let performance_info = self.network.get_performance_info();
    return Ok(performance_info);
  }
}

pub struct BasicNeuralNetworkRust {
  pub non_input_layer_sizes: Vec<usize>,
  pub weights: Vec<Matrix>,
  pub biases: Vec<Matrix>,
  pub weight_optimizers: Vec<Box<dyn Optimizer>>,
  pub bias_optimizers: Vec<Box<dyn Optimizer>>,
  pub neuron_outputs: Vec<Matrix>,
  curr_performance_info: (Matrix, Matrix), // (num_correct, loss)
  pub performance_info: Vec<(f32, f32, usize, usize)>, // (accuracy, loss, sample_count, iterations)
  pub collect_performance_info: bool,
}

impl BasicNeuralNetworkRust {
  pub fn new(
    hidden_layer_sizes: Vec<usize>,
    num_features: usize,
    num_classifications: usize, // Set to anything 1 to perform regression
  ) -> Self {
    // Gather data on dimensions for matrices
    let mut non_input_layer_sizes = hidden_layer_sizes.clone();
    non_input_layer_sizes.push(num_classifications);

    // Init the matrices
    let weights = (0..non_input_layer_sizes.len())
      .map(|layer| {
        let input_feature_count = if layer == 0 {
          num_features
        } else {
          non_input_layer_sizes[layer - 1]
        };
        Matrix::new_random(
          0.0,
          f64::sqrt(2.0 / input_feature_count as f64),
          non_input_layer_sizes[layer],
          input_feature_count,
        )
      })
      .collect_vec();

    let weight_optimizers = (0..non_input_layer_sizes.len())
      .map(|_| Box::new(AdamOptimizer::new(1e-2, 0.9, 0.999)) as Box<dyn Optimizer>)
      .collect_vec();

    let biases = (0..non_input_layer_sizes.len())
      .map(|layer| Matrix::zeros(non_input_layer_sizes[layer], 1))
      .collect_vec();

    let bias_optimizers = (0..non_input_layer_sizes.len())
      .map(|_| Box::new(AdamOptimizer::new(1e-2, 0.9, 0.999)) as Box<dyn Optimizer>)
      .collect_vec();

    // Create network
    let network = BasicNeuralNetworkRust {
      non_input_layer_sizes,
      weights,
      biases,
      weight_optimizers,
      bias_optimizers,
      neuron_outputs: Vec::new(),
      curr_performance_info: (Matrix::zeros(1, 1), Matrix::zeros(1, 1)),
      performance_info: vec![(0.0, 0.0, 0, 0)],
      collect_performance_info: true,
    };

    // Cleanup and return
    return network;
  }

  pub fn set_optimizer(&mut self, optimizer: Box<dyn Optimizer>) {
    self.weight_optimizers = (0..self.non_input_layer_sizes.len())
      .map(|_| optimizer.clone())
      .collect_vec();
    self.bias_optimizers = (0..self.non_input_layer_sizes.len())
      .map(|_| optimizer.clone())
      .collect_vec();
  }

  pub fn print_structure(&self) {
    println!("Number of layers in network: {}", self.weights.len());
    println!("Layer shapes:");
    for weight in self.weights.iter() {
      weight.print_shape();
    }
  }

  pub fn set_collect_performance_info(&mut self, collect_performance_info: bool) {
    self.collect_performance_info = collect_performance_info;
  }

  pub fn train(
    &mut self,
    features_train: Vec<Vec<f32>>,
    input_labels: Vec<f32>,
    num_iterations: usize,
    batch_size: usize,
  ) {
    let num_classifications = *self.non_input_layer_sizes.last().unwrap_or(&1);
    let num_observations = features_train.len();

    self.neuron_outputs = self
      .non_input_layer_sizes
      .iter()
      .map(|&layer_size| Matrix::no_fill(layer_size, num_observations))
      .collect();

    // Train model, choose regression or classification
    if num_classifications == 1 {
      self.train_regression(features_train, input_labels, num_iterations, batch_size);
    } else {
      self.train_classification(
        features_train,
        input_labels,
        num_classifications,
        num_iterations,
        batch_size,
      );
    }
  }

  pub fn regression(&mut self, features_test: &Vec<Vec<f32>>) -> Vec<f32> {
    let num_observations = features_test.len();

    // Feed forward through network
    let observations = Matrix::new_2d(&features_test).transpose();

    self.neuron_outputs = self
      .weights
      .iter()
      .map(|layer| Matrix::no_fill(layer.get_data_length(), num_observations))
      .collect_vec();

    self.feed_forward(&observations);

    return self.neuron_outputs[self.neuron_outputs.len() - 1].get_data()[0].to_vec();
  }

  pub fn classify(&mut self, features_test: &Vec<Vec<f32>>) -> Vec<f32> {
    let num_observations = features_test.len();

    // Feed forward through network
    let observations = Matrix::new_2d(&features_test).transpose();

    self.neuron_outputs = self
      .weights
      .iter()
      .map(|layer| Matrix::no_fill(layer.get_data_length(), num_observations))
      .collect_vec();

    self.feed_forward(&observations);
    let predicted_probabilities = Self::softmax(&self.neuron_outputs);
    let classifications = Self::get_classification(&predicted_probabilities);

    return classifications;
  }

  pub fn classify_matrix(&mut self, observations: &Matrix) -> Vec<f32> {
    let num_observations = observations.get_columns();
    self.neuron_outputs = self
      .weights
      .iter()
      .map(|layer| Matrix::no_fill(layer.get_data_length(), num_observations))
      .collect_vec();

    self.feed_forward(&observations);
    let predicted_probabilities = Self::softmax(&self.neuron_outputs);
    let classifications = Self::get_classification(&predicted_probabilities);

    return classifications;
  }

  pub fn get_classification(predicted_probabilities: &Matrix) -> Vec<f32> {
    return predicted_probabilities.argmax_by_column().get_data()[0].to_vec();
  }

  pub fn mini_batch(
    row_based_observations: &Vec<Vec<f32>>,
    labels: &Vec<f32>,
    batch_size: usize,
  ) -> (Matrix, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let row_dist = Uniform::new(0, row_based_observations.len());

    let mut mini_batch_data = Vec::new();
    let mut mini_batch_labels = vec![0.0; batch_size];
    for i in 0..batch_size {
      let row_index = row_dist.sample(&mut rng);
      let sampled_row = row_based_observations[row_index].to_vec();
      mini_batch_data.push(sampled_row);
      mini_batch_labels[i] = labels[row_index];
    }

    let mini_batch = Matrix::new_2d(&mini_batch_data).transpose();
    return (mini_batch, mini_batch_labels);
  }

  pub fn train_regression(
    &mut self,
    observations: Vec<Vec<f32>>,
    labels: Vec<f32>,
    num_iterations: usize,
    batch_size: usize,
  ) {
    let observations_matrix = Matrix::new_2d(&observations).transpose();

    for _ in 0..num_iterations {
      // Batch
      let batch_data = BasicNeuralNetworkRust::mini_batch(&observations, &labels, batch_size);

      let batch = if batch_size == 0 {
        &observations_matrix
      } else {
        &batch_data.0
      };
      let batch_labels = if batch_size == 0 {
        &labels
      } else {
        &batch_data.1
      };

      // Feed forward
      self.feed_forward(batch);

      // Calculate error from feed forward step
      let output_error = self.backpropogation_output_layer_regression(batch_labels);

      // Backpropogate hidden
      self.backpropogation_hidden_layer(
        batch,
        &output_error,
        self.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
      );
    }
  }

  pub fn print_performance_info(&self, sample_index: usize, display_index: usize) {
    let (accuracy, loss, _, _) = self.performance_info[sample_index];
    println!(
      "Iteration {}: Accuracy: {} Loss: {}",
      display_index, accuracy, loss
    );
  }

  pub fn get_performance_info(&self) -> Vec<(f32, f32)> {
    return self
      .performance_info
      .iter()
      .take(self.performance_info.len() - 1)
      .map(|(accuracy, loss, _, _)| (*accuracy, *loss))
      .collect_vec();
  }

  pub fn update_performance_info(
    &mut self,
    predicted_probabilities: &Matrix,
    encoded_labels: &Matrix,
  ) {
    if !self.collect_performance_info {
      return;
    }

    // Performance info is amalgamation of all samples for 50 iterations
    let iteration_limit = 50;
    let curr_iteration = self.performance_info.last().unwrap().3;

    // Add new entry if we have reached the iteration limit
    if curr_iteration == iteration_limit {
      let curr_performance_num_correct = self.curr_performance_info.0.get_data()[0][0];
      let curr_performance_loss = self.curr_performance_info.1.get_data()[0][0];

      self.curr_performance_info.0.scalar_multiply_inplace(0.0); // Reset
      self.curr_performance_info.1.scalar_multiply_inplace(0.0); // Reset

      {
        let (accuracy, loss, sample_count, _) = self.performance_info.last_mut().unwrap();

        *accuracy = 100.0 * (curr_performance_num_correct / *sample_count as f32);
        *loss = curr_performance_loss + 1e-8;
      }

      self.print_performance_info(
        self.performance_info.len() - 1,
        iteration_limit * self.performance_info.len(),
      );

      self.performance_info.push((0.0, 0.0, 0, 0));
    }

    let (num_correct, loss) = &mut self.curr_performance_info;

    let num_classifications = encoded_labels.get_rows();
    let batch_num_correct = predicted_probabilities
      .argmax_by_column()
      .one_hot_encode(num_classifications)
      .transpose()
      .element_multiply(encoded_labels)
      .sum_all_matrix_elements();
    num_correct.element_add_inplace(&batch_num_correct);

    let iter_loss = predicted_probabilities
      .element_multiply(encoded_labels)
      .element_ln_inplace()
      .sum_all_matrix_elements();

    loss.element_subtract_inplace(&iter_loss);

    // Increment iteration and num_samples
    self.performance_info.last_mut().unwrap().2 += encoded_labels.get_columns();
    self.performance_info.last_mut().unwrap().3 += 1;
  }

  pub fn train_classification(
    &mut self,
    observations: Vec<Vec<f32>>,
    labels: Vec<f32>,
    num_classifications: usize,
    num_iterations: usize,
    batch_size: usize,
  ) {
    let observations_matrix = Matrix::new_2d(&observations).transpose();

    // For now we will make the number of iterations a constant
    for _ in 0..num_iterations {
      // Batch
      let batch_data: (Matrix, Vec<f32>) =
        BasicNeuralNetworkRust::mini_batch(&observations, &labels, batch_size);

      let batch = if batch_size == 0 {
        &observations_matrix
      } else {
        &batch_data.0
      };
      let batch_labels = if batch_size == 0 {
        &labels
      } else {
        &batch_data.1
      };
      let batch_labels_encoded =
        Matrix::new_one_hot_encoded(batch_labels, num_classifications).transpose();

      // Feed forward
      self.feed_forward(batch);

      // Calculate error from feed forward step
      let predicted_probabilities = Self::softmax(&self.neuron_outputs);
      self.update_performance_info(&predicted_probabilities, &batch_labels_encoded);
      let output_error = self.backpropogation_output_layer_classification(
        &observations_matrix,
        &predicted_probabilities,
        &batch_labels_encoded,
      );

      // Backpropogate hidden
      self.backpropogation_hidden_layer(
        batch,
        &output_error,
        self.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
      );
    }
  }

  pub fn train_classification_observation_matrix(
    &mut self,
    observations: &Matrix, // Expects each sample is in a column (so like a transposed pd datatable)
    encoded_labels: &Matrix,
  ) -> Matrix {
    let num_observations = observations.get_columns();
    let num_classes = *self.non_input_layer_sizes.last().unwrap_or(&1);
    self.neuron_outputs = self
      .non_input_layer_sizes
      .iter()
      .map(|&layer_size| Matrix::no_fill(layer_size, num_observations))
      .collect();

    // Feed forward
    self.feed_forward(&observations);

    // Calculate error from feed forward step
    let predicted_probabilities = Self::softmax(&self.neuron_outputs);
    self.update_performance_info(&predicted_probabilities, &encoded_labels);
    let output_error = self.backpropogation_output_layer_classification(
      &observations,
      &predicted_probabilities,
      &encoded_labels,
    );

    if self.weights.len() == 1 {
      // Backpropogation no hidden layer
      return output_error;
    } else {
      // Backpropogate hidden
      return self.backpropogation_hidden_layer(
        &observations,
        &output_error,
        self.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
      );
    }
  }

  pub fn feed_forward(&mut self, observations: &Matrix) {
    let num_layers = self.weights.len();
    for layer in 0..num_layers {
      self.neuron_outputs[layer] = self.weights[layer]
        .matrix_multiply(if layer == 0 {
          observations
        } else {
          &self.neuron_outputs[layer - 1]
        })
        .add_vector(&self.biases[layer]);

      // Activation for final layer is softmax not ReLU
      if layer != num_layers - 1 {
        self.neuron_outputs[layer].element_ReLU_inplace();
      }
    }
  }

  pub fn softmax(neuron_outputs: &Vec<Matrix>) -> Matrix {
    return neuron_outputs.last().unwrap().softmax();
  }

  pub fn backpropogation_output_layer_regression(&mut self, labels: &Vec<f32>) -> Matrix {
    let output_layer_index = self.biases.len() - 1;
    let prev_layer_outputs = &self.neuron_outputs[output_layer_index - 1];
    let output_biases = &self.biases[output_layer_index];
    let output_weights = &self.weights[output_layer_index];
    let weight_optimizer = &mut self.weight_optimizers[output_layer_index];
    let bias_optimizer = &mut self.bias_optimizers[output_layer_index];
    let normalization_factor = 1.0 / prev_layer_outputs.get_columns() as f32;

    // Shared error calculations (dSSR)
    // neuron_output[output_layer_index][0] because there is only one output neuron
    let output_layer_data = self.neuron_outputs[output_layer_index].get_data(); // TODO: Measure performance impact of copying to host
    let error = Matrix::new_2d(&vec![(0..labels.len())
      .map(|index| -2.0 * (labels[index] - output_layer_data[0][index]))
      .collect_vec()]);

    // Update biases first
    // b' = b - learning_rate * batch_sum( if label==Output bias codes for {predicted coded for -1} else {predicted coded for} )
    let db = error.sum_rows_matrix();
    let step_db = bias_optimizer
      .calculate_step(&db)
      .scalar_multiply_inplace(normalization_factor);
    self.biases[output_layer_index] = output_biases.element_subtract_inplace(&step_db);

    // Update weights
    // w' = w - learning_rate * batch_sum( Output bias codes for Yin * {predicted coded for -1} else Yin * {predicted coded for} )
    let dw = error.matrix_multiply(&prev_layer_outputs.transpose());
    let step_dw = weight_optimizer
      .calculate_step(&dw)
      .scalar_multiply_inplace(normalization_factor);
    self.weights[output_layer_index] = output_weights.element_subtract_inplace(&step_dw);

    // Return error for use in other backpropogation
    return error;
  }

  pub fn backpropogation_output_layer_classification(
    &mut self,
    observations: &Matrix,
    predicted_probabilities: &Matrix,
    encoded_labels: &Matrix,
  ) -> Matrix {
    let output_layer_index = self.biases.len() - 1;
    let prev_layer_outputs = if output_layer_index == 0 {
      observations
    } else {
      &self.neuron_outputs[output_layer_index - 1]
    };
    let output_biases = &self.biases[output_layer_index];
    let output_weights = &self.weights[output_layer_index];
    let weight_optimizer = &mut self.weight_optimizers[output_layer_index];
    let bias_optimizer = &mut self.bias_optimizers[output_layer_index];
    let normalization_factor = 1.0 / prev_layer_outputs.get_columns() as f32;

    // Shared error calculations (dCE * dSoftmax)
    let error = predicted_probabilities.element_subtract(encoded_labels);

    // Update biases first
    // b' = b - learning_rate * batch_sum( if label==Output bias codes for {predicted coded for -1} else {predicted coded for} )
    let db = error.sum_rows_matrix();
    let step_db = bias_optimizer
      .calculate_step(&db)
      .scalar_multiply_inplace(normalization_factor);
    self.biases[output_layer_index] = output_biases.element_subtract_inplace(&step_db);

    // Update weights
    // w' = w - learning_rate * batch_sum( Output bias codes for Yin * {predicted coded for -1} else Yin * {predicted coded for} )
    let dw = error.matrix_multiply(&prev_layer_outputs.transpose());
    let step_dw = weight_optimizer
      .calculate_step(&dw)
      .scalar_multiply_inplace(normalization_factor);
    self.weights[output_layer_index] = output_weights.element_subtract_inplace(&step_dw);

    // Return error for use in other backpropogation
    return error;
  }

  pub fn backpropogation_hidden_layer(
    &mut self,
    observations: &Matrix,
    next_layer_error: &Matrix,
    layer: usize,
  ) -> Matrix {
    // Used for yin
    let prev_layer_outputs = if layer == 0 {
      observations
    } else {
      &self.neuron_outputs[layer - 1]
    };

    let weight_optimizer = &mut self.weight_optimizers[layer];
    let bias_optimizer = &mut self.bias_optimizers[layer];

    let normalization_factor = 1.0 / prev_layer_outputs.get_columns() as f32;

    // Used for wout
    let wout = &self.weights[layer + 1];

    // Used for x
    // ASSUMPTION: Since using ReLU, I can just take activation_function_prime of layer output.
    // Otherwise you may have to use raw input to original activation function
    let activation_prime_x = self.neuron_outputs[layer].element_ReLU_prime();

    // Shared error calculations
    let error = wout
      .transpose()
      .matrix_multiply(next_layer_error)
      .element_multiply_inplace(&activation_prime_x);

    // Update biases
    // b' = b - learning_rate * batch_sum(wout * activation'(x) * next_layer_error)
    let db = error.sum_rows_matrix();
    let step_db = bias_optimizer
      .calculate_step(&db)
      .scalar_multiply_inplace(normalization_factor);
    self.biases[layer] = self.biases[layer].element_subtract_inplace(&step_db);

    // Update weights
    // w' = w - learning_rate * batch_sum(wout * activation'(x) * next_layer_error * yin)
    let dw = error.matrix_multiply(&prev_layer_outputs.transpose());
    let step_dw = weight_optimizer
      .calculate_step(&dw)
      .scalar_multiply_inplace(normalization_factor);
    self.weights[layer] = self.weights[layer].element_subtract_inplace(&step_dw);

    if layer != 0 {
      self.backpropogation_hidden_layer(observations, &error, layer - 1);
    }
    return error;
  }
}
