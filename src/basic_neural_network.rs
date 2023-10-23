// use crate::py_util::py_print;
use itertools::{izip, Itertools};
use matrix_lib::Matrix;
use pyo3::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution};
use statrs::distribution::Normal;

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

  fn train(
    &mut self,
    features_train: Vec<Vec<f32>>,
    input_labels: Vec<f32>,
    learning_rate: f32,
    num_iterations: usize,
    batch_size: usize,
  ) {
    self.network.train(
      features_train,
      input_labels,
      learning_rate,
      num_iterations,
      batch_size,
    );
  }

  fn classify(&mut self, features_test: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let classifications = self.network.classify(&features_test);

    return Ok(classifications);
  }

  fn regression(&mut self, features_test: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let predictions = self.network.regression(&features_test);

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
}

pub struct BasicNeuralNetworkRust {
  pub non_input_layer_sizes: Vec<usize>,
  pub weights: Vec<Matrix>,
  pub biases: Vec<Matrix>,
  pub neuron_outputs: Vec<Matrix>,
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
    // Random seed for weights
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 0.68).unwrap();

    let weights = (0..non_input_layer_sizes.len())
      .map(|layer| {
        Matrix::new_2d(
          &(0..non_input_layer_sizes[layer])
            .map(|_| {
              (0..if layer == 0 {
                num_features
              } else {
                non_input_layer_sizes[layer - 1]
              })
                .map(|_| range.sample(&mut rng) as f32)
                .collect_vec()
            })
            .collect_vec(),
        )
      })
      .collect_vec();

    let biases = (0..non_input_layer_sizes.len())
      .map(|layer| Matrix::zeros(non_input_layer_sizes[layer], 1))
      .collect_vec();

    // Create network
    let network = BasicNeuralNetworkRust {
      non_input_layer_sizes,
      weights,
      biases,
      neuron_outputs: Vec::new(),
    };

    // Cleanup and return
    return network;
  }

  pub fn print_structure(&self) {
    println!("Number of layers in network: {}", self.weights.len());
    println!("Layer shapes:");
    for weight in self.weights.iter() {
      weight.print_shape();
    }
  }

  pub fn train(
    &mut self,
    features_train: Vec<Vec<f32>>,
    input_labels: Vec<f32>,
    learning_rate: f32,
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
      self.train_regression(
        features_train,
        input_labels,
        learning_rate,
        num_iterations,
        batch_size,
      );
    } else {
      self.train_classification(
        features_train,
        input_labels,
        learning_rate,
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
    let num_observations = observations.columns;
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
    let pred_data = predicted_probabilities.transpose().get_data();
    return pred_data
      .iter()
      .map(|outputs| {
        outputs
          .iter()
          .enumerate()
          .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
          .map(|a| a.0 as f32)
          .unwrap()
      })
      .collect_vec();
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
    learning_rate: f32,
    num_iterations: usize,
    batch_size: usize,
  ) {
    let observations_matrix = Matrix::new_2d(&observations).transpose();

    for i in 0..num_iterations {
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
      let output_error = self.backpropogation_output_layer_regression(batch_labels, learning_rate);

      // Backpropogate hidden
      self.backpropogation_hidden_layer(
        batch,
        &output_error,
        learning_rate,
        self.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
      );

      // Print progress
      if i % 50 == 0 {
        self.test_train_performance_regression(&observations_matrix, &labels);
      }
    }
  }

  pub fn train_classification(
    &mut self,
    observations: Vec<Vec<f32>>,
    labels: Vec<f32>,
    learning_rate: f32,
    num_iterations: usize,
    batch_size: usize,
  ) {
    let observations_matrix = Matrix::new_2d(&observations).transpose();

    // For now we will make the number of iterations a constant
    for i in 0..num_iterations {
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

      // Feed forward
      self.feed_forward(batch);

      // Calculate error from feed forward step
      let predicted_probabilities = Self::softmax(&self.neuron_outputs);
      let output_error = self.backpropogation_output_layer_classification(
        &observations_matrix,
        &predicted_probabilities,
        batch_labels,
        learning_rate,
      );

      // Backpropogate hidden
      self.backpropogation_hidden_layer(
        batch,
        &output_error,
        learning_rate,
        self.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
      );

      // Comment out for tests (pyo3 doesn't like test mode idk why)
      if i % 50 == 0 {
        self.test_train_performance_classification(&observations_matrix, &labels);
      }
    }
  }

  pub fn train_classification_observation_matrix(
    &mut self,
    observations: &Matrix, // Expects each sample is in a column (so like a transposed pd datatable)
    labels: &Vec<f32>,
    learning_rate: f32,
  ) -> Matrix {
    let num_observations = observations.columns;
    self.neuron_outputs = self
      .non_input_layer_sizes
      .iter()
      .map(|&layer_size| Matrix::no_fill(layer_size, num_observations))
      .collect();

    // Feed forward
    self.feed_forward(&observations);

    // Calculate error from feed forward step
    let predicted_probabilities = Self::softmax(&self.neuron_outputs);
    let output_error = self.backpropogation_output_layer_classification(
      &observations,
      &predicted_probabilities,
      &labels,
      learning_rate,
    );

    if self.weights.len() == 1 {
      // Backpropogation no hidden layer
      return output_error;
    } else {
      // Backpropogate hidden
      return self.backpropogation_hidden_layer(
        &observations,
        &output_error,
        learning_rate,
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
        .add_vector(&self.biases[layer])
        .element_ReLU();
    }
  }

  pub fn softmax(neuron_outputs: &Vec<Matrix>) -> Matrix {
    let outputs_exp = neuron_outputs[neuron_outputs.len() - 1].element_exp();

    let exp_final_layer_outputs_summed = outputs_exp.sum_columns_matrix();

    // Divide all data by col sum
    let predictions = outputs_exp.divide_by_vector(&exp_final_layer_outputs_summed);
    return predictions;
  }

  pub fn backpropogation_output_layer_regression(
    &mut self,
    labels: &Vec<f32>,
    learning_rate: f32,
  ) -> Matrix {
    let output_layer_index = self.biases.len() - 1;
    let prev_layer_outputs = &self.neuron_outputs[output_layer_index - 1];
    let output_biases = &self.biases[output_layer_index];
    let output_weights = &self.weights[output_layer_index];
    let normalization_factor = 1.0 / prev_layer_outputs.columns as f32;

    // Shared error calculations (dSSR)
    // neuron_output[output_layer_index][0] because there is only one output neuron
    let output_layer_data = self.neuron_outputs[output_layer_index].get_data(); // TODO: Measure performance impact of copying to host
    let error = Matrix::new_2d(&vec![(0..labels.len())
      .map(|index| -2.0 * (labels[index] - output_layer_data[0][index]))
      .collect_vec()]);

    // Update biases first
    // b' = b - learning_rate * batch_sum( if label==Output bias codes for {predicted coded for -1} else {predicted coded for} )
    let db = error.sum_rows_matrix();
    self.biases[output_layer_index] =
      output_biases.element_subtract(&db.scalar_multiply(learning_rate * normalization_factor));

    // Update weights
    // w' = w - learning_rate * batch_sum( Output bias codes for Yin * {predicted coded for -1} else Yin * {predicted coded for} )
    let dw = error.matrix_multiply(&prev_layer_outputs.transpose());
    self.weights[output_layer_index] =
      output_weights.element_subtract(&dw.scalar_multiply(learning_rate * normalization_factor));

    // Return error for use in other backpropogation
    return error;
  }

  pub fn backpropogation_output_layer_classification(
    &mut self,
    observations: &Matrix,
    predicted_probabilities: &Matrix,
    labels: &Vec<f32>,
    learning_rate: f32,
  ) -> Matrix {
    let output_layer_index = self.biases.len() - 1;
    let prev_layer_outputs = if output_layer_index == 0 {
      observations
    } else {
      &self.neuron_outputs[output_layer_index - 1]
    };
    let output_biases = &self.biases[output_layer_index];
    let output_weights = &self.weights[output_layer_index];
    let normalization_factor = 1.0 / prev_layer_outputs.columns as f32;

    // Shared error calculations (dCE * dSoftmax)
    let predicted_probabilities_data = predicted_probabilities.get_data(); // TODO: Measure performance impact of copying to host
    let error = Matrix::new_2d(
      &(0..output_biases.rows)
        .map(|index| {
          izip!(labels.iter(), predicted_probabilities_data[index].iter())
            .map(|(label, predicted_probability)| {
              if *label == index as f32 {
                *predicted_probability - 1.0
              } else {
                *predicted_probability
              }
            })
            .collect_vec()
        })
        .collect_vec(),
    );

    // Update biases first
    // b' = b - learning_rate * batch_sum( if label==Output bias codes for {predicted coded for -1} else {predicted coded for} )
    let db = error.sum_rows_matrix();
    self.biases[output_layer_index] =
      output_biases.element_subtract(&db.scalar_multiply(learning_rate * normalization_factor));

    // Update weights
    // w' = w - learning_rate * batch_sum( Output bias codes for Yin * {predicted coded for -1} else Yin * {predicted coded for} )
    let dw = error.matrix_multiply(&prev_layer_outputs.transpose());
    self.weights[output_layer_index] =
      output_weights.element_subtract(&dw.scalar_multiply(learning_rate * normalization_factor));

    // Return error for use in other backpropogation
    return error;
  }

  pub fn backpropogation_hidden_layer(
    &mut self,
    observations: &Matrix,
    next_layer_error: &Matrix,
    learning_rate: f32,
    layer: usize,
  ) -> Matrix {
    // Used for yin
    let prev_layer_outputs = if layer == 0 {
      observations
    } else {
      &self.neuron_outputs[layer - 1]
    };

    let normalization_factor = 1.0 / prev_layer_outputs.columns as f32;

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
      .element_multiply(&activation_prime_x);

    // Update biases
    // b' = b - learning_rate * batch_sum(wout * activation'(x) * next_layer_error)
    let db = error.sum_rows_matrix();
    self.biases[layer] = self.biases[layer]
      .element_subtract(&db.scalar_multiply(learning_rate * normalization_factor));

    // Update weights
    // w' = w - learning_rate * batch_sum(wout * activation'(x) * next_layer_error * yin)
    let dw = error.matrix_multiply(&prev_layer_outputs.transpose());
    self.weights[layer] = self.weights[layer]
      .element_subtract(&dw.scalar_multiply(learning_rate * normalization_factor));

    if layer != 0 {
      self.backpropogation_hidden_layer(observations, &error, learning_rate, layer - 1);
    }
    return error;
  }

  fn matrix_classify(&mut self, observations: &Matrix) -> Vec<f32> {
    let num_observations = observations.columns;

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

  fn matrix_regression(&mut self, observations: &Matrix) -> Vec<f32> {
    let num_observations = observations.columns;

    self.neuron_outputs = self
      .weights
      .iter()
      .map(|layer| Matrix::no_fill(layer.get_data_length(), num_observations))
      .collect_vec();

    self.feed_forward(&observations);
    let classifications = self.neuron_outputs[self.neuron_outputs.len() - 1].get_data()[0].to_vec();

    return classifications;
  }

  fn test_train_performance_regression(&mut self, observations: &Matrix, labels: &Vec<f32>) {
    let classifications = self.matrix_regression(&observations);
    let tolerance = 0.05;
    let num_correct =
      izip!(classifications.iter(), labels.iter()).fold(0.0, |acc, (classification, label)| {
        acc
          + if (classification - label).abs() <= (label * tolerance).abs() {
            1.0
          } else {
            0.0
          }
      });

    let percent_correct = 100.0 * num_correct / labels.len() as f32;

    println!("{}", &format!("% Correct: {}", percent_correct));
  }

  fn test_train_performance_classification(&mut self, observations: &Matrix, labels: &Vec<f32>) {
    let classifications = self.matrix_classify(&observations);
    let num_correct = izip!(classifications.iter(), labels.iter())
      .fold(0.0, |acc, (classification, label)| {
        acc + if classification == label { 1.0 } else { 0.0 }
      });

    let percent_correct = 100.0 * num_correct / labels.len() as f32;

    println!("{}", &format!("% Correct: {}", percent_correct));
  }
}
