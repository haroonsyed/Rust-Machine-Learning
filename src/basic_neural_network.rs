use crate::py_util::py_print;
use itertools::{izip, Itertools};
use matrix_lib::Matrix;
use pyo3::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution};
use statrs::distribution::Normal;

// So I was initially gonna use a class representation for each neuron
// But I think I can make things simpler and more efficient with matrices
/*
Neural Network Representation:
Matrix 1: The observations
Matrix 2: The weights
Matrix 3: The bias (conserved a bit of memory)
vector<Matrix 4>: The outputs of each neuron

Matrix 4 dimensions will be large enough to store
#neurons * #observations
where dimensions = [#layers, Matrix[#neurons_largest_hidden, #observations]]

Where
weights.matmult(observations).elem_add(bias) = raw_neuron_outputs
activation_function(raw_neuron_outputs) = neuron_outputs
 */

// ACTIVATION FUNCTIONS

pub trait ActivationFunction {
  fn activation_function(&self, x: f64) -> f64;
  fn activation_function_prime(&self, x: f64) -> f64;
}

pub struct Relu;
impl ActivationFunction for Relu {
  fn activation_function(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    } else {
      return x;
    }
  }

  fn activation_function_prime(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    } else {
      return 1.0;
    }
  }
}

#[pyclass]
pub struct BasicNeuralNetwork {
  pub weights: Vec<Matrix>,
  pub biases: Vec<Matrix>,
}

#[pymethods]
impl BasicNeuralNetwork {
  #[new]
  fn new(
    features_train: Vec<Vec<f64>>,
    input_labels: Vec<f64>,
    hidden_layer_sizes: Vec<usize>,
    num_classifications: usize, // Set to anything 1 to perform regression
    learning_rate: f64,
    num_iterations: usize,
    batch_size: usize, // Set to 0 to use stochastic GD
  ) -> Self {
    // Gather data on dimensions for matrices
    let num_observations = features_train.len();
    let num_features = features_train[0].len();
    let mut non_input_layer_sizes = hidden_layer_sizes.clone();
    non_input_layer_sizes.push(num_classifications);

    // Init the matrices
    let observations = Matrix::new_2d(features_train).transpose();

    // Random seed for weights
    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 0.68).unwrap();

    let weights = (0..non_input_layer_sizes.len())
      .map(|layer| {
        Matrix::new_2d(
          (0..non_input_layer_sizes[layer])
            .map(|_| {
              (0..if layer == 0 {
                num_features
              } else {
                non_input_layer_sizes[layer - 1]
              })
                .map(|_| range.sample(&mut rng))
                .collect_vec()
            })
            .collect_vec(),
        )
      })
      .collect_vec();

    let biases = (0..non_input_layer_sizes.len())
      .map(|layer| Matrix::zeros(non_input_layer_sizes[layer], 1))
      .collect_vec();

    let mut neuron_outputs: Vec<Matrix> = non_input_layer_sizes
      .iter()
      .map(|&layer_size| Matrix::zeros(layer_size, num_observations))
      .collect();

    // Create network
    let mut network = BasicNeuralNetwork { weights, biases };

    // Train model, choose regression or classification
    if num_classifications == 1 {
      network.train_regression(
        &observations,
        &mut neuron_outputs,
        &input_labels,
        learning_rate,
        Box::new(Relu {}),
        num_iterations,
        batch_size,
      );
    } else {
      network.train_classification(
        &observations,
        &mut neuron_outputs,
        &input_labels,
        learning_rate,
        Box::new(Relu {}),
        num_iterations,
        batch_size,
      );
    }

    unsafe {
      matrix_lib::bindings::test();
    }

    // Cleanup and return
    return network;
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let num_observations = features_test.len();

    // Feed forward through network
    let observations = Matrix::new_2d(features_test).transpose();

    let mut neuron_outputs: Vec<Matrix> = self
      .weights
      .iter()
      .map(|layer| Matrix::zeros(layer.data.len(), num_observations))
      .collect_vec();

    let activation_function: Box<dyn ActivationFunction> = Box::new(Relu {});
    self.feed_forward(&observations, &mut neuron_outputs, &activation_function);
    let predicted_probabilities = Self::softmax(&neuron_outputs);
    let classifications = Self::get_classification(&predicted_probabilities);

    return Ok(classifications);
  }

  fn regression(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let num_observations = features_test.len();

    // Feed forward through network
    let observations = Matrix::new_2d(features_test).transpose();

    let mut neuron_outputs: Vec<Matrix> = self
      .weights
      .iter()
      .map(|layer| Matrix::zeros(layer.data.len(), num_observations))
      .collect_vec();

    let activation_function: Box<dyn ActivationFunction> = Box::new(Relu {});
    self.feed_forward(&observations, &mut neuron_outputs, &activation_function);
    return Ok(neuron_outputs[neuron_outputs.len() - 1][0].to_vec());
  }
}

impl BasicNeuralNetwork {
  fn rust_classify(&self, observations: &Matrix) -> Vec<f64> {
    let num_observations = observations.columns;

    let mut neuron_outputs: Vec<Matrix> = self
      .weights
      .iter()
      .map(|layer| Matrix::zeros(layer.data.len(), num_observations))
      .collect_vec();

    let activation_function: Box<dyn ActivationFunction> = Box::new(Relu {});
    self.feed_forward(&observations, &mut neuron_outputs, &activation_function);
    let predicted_probabilities = Self::softmax(&neuron_outputs);
    let classifications = Self::get_classification(&predicted_probabilities);

    return classifications;
  }

  pub fn get_classification(predicted_probabilities: &Matrix) -> Vec<f64> {
    return predicted_probabilities
      .transpose()
      .iter()
      .map(|outputs| {
        outputs
          .iter()
          .enumerate()
          .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
          .map(|a| a.0 as f64)
          .unwrap()
      })
      .collect();
  }

  pub fn mini_batch(
    observations: &Matrix,
    labels: &Vec<f64>,
    batch_size: usize,
  ) -> (Matrix, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let column_dist = Uniform::new(0, observations.columns);

    let mut mini_batch = Matrix::zeros(observations.rows, batch_size);
    let mut mini_batch_labels = vec![0.0; batch_size];
    for i in 0..batch_size {
      let column_index = column_dist.sample(&mut rng);
      for j in 0..observations.rows {
        mini_batch[j][i] = observations[j][column_index];
      }
      mini_batch_labels[i] = labels[column_index];
    }

    return (mini_batch, mini_batch_labels);
  }

  pub fn train_regression(
    &mut self,
    observations: &Matrix,
    neuron_outputs: &mut Vec<Matrix>,
    labels: &Vec<f64>,
    learning_rate: f64,
    activation_function: Box<dyn ActivationFunction>,
    num_iterations: usize,
    batch_size: usize,
  ) {
    for i in 0..num_iterations {
      // Batch
      let batch_data = BasicNeuralNetwork::mini_batch(observations, labels, batch_size);

      let batch = if batch_size == 0 {
        observations
      } else {
        &batch_data.0
      };
      let batch_labels = if batch_size == 0 {
        labels
      } else {
        &batch_data.1
      };

      // Feed forward
      self.feed_forward(batch, neuron_outputs, &activation_function);

      // Calculate error from feed forward step
      let output_error =
        self.backpropogation_output_layer_regression(batch_labels, neuron_outputs, learning_rate);

      // Backpropogate hidden
      self.backpropogation_hidden_layer(
        batch,
        neuron_outputs,
        &output_error,
        learning_rate,
        &activation_function,
        self.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
      );

      // Print progress
      if i % 50 == 0 {
        self.test_train_performance_regression(observations, labels);
      }
    }
  }

  pub fn train_classification(
    &mut self,
    observations: &Matrix,
    neuron_outputs: &mut Vec<Matrix>,
    labels: &Vec<f64>,
    learning_rate: f64,
    activation_function: Box<dyn ActivationFunction>,
    num_iterations: usize,
    batch_size: usize,
  ) {
    // For now we will make the number of iterations a constant
    for i in 0..num_iterations {
      // Batch
      let batch_data = BasicNeuralNetwork::mini_batch(observations, labels, batch_size);

      let batch = if batch_size == 0 {
        observations
      } else {
        &batch_data.0
      };
      let batch_labels = if batch_size == 0 {
        labels
      } else {
        &batch_data.1
      };

      // Feed forward
      self.feed_forward(batch, neuron_outputs, &activation_function);

      // Calculate error from feed forward step
      let predicted_probabilities = Self::softmax(neuron_outputs);
      let output_error = self.backpropogation_output_layer_classification(
        &predicted_probabilities,
        batch_labels,
        neuron_outputs,
        learning_rate,
      );

      // Backpropogate hidden
      self.backpropogation_hidden_layer(
        batch,
        neuron_outputs,
        &output_error,
        learning_rate,
        &activation_function,
        self.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
      );

      if i % 50 == 0 {
        self.test_train_performance_classification(observations, labels);
      }
    }
  }

  pub fn feed_forward(
    &self,
    observations: &Matrix,
    neuron_outputs: &mut Vec<Matrix>,
    activation_function: &Box<dyn ActivationFunction>,
  ) {
    let num_layers = self.weights.len();
    for layer in 0..num_layers {
      neuron_outputs[layer] = self.weights[layer]
        .matrix_multiply(if layer == 0 {
          observations
        } else {
          &neuron_outputs[layer - 1]
        })
        .add_vector_to_columns(&self.biases[layer])
        .element_apply(&|x| activation_function.activation_function(x));
    }
  }

  pub fn softmax(neuron_outputs: &Vec<Matrix>) -> Matrix {
    let mut predictions = neuron_outputs[neuron_outputs.len() - 1].element_apply(&|x| f64::exp(x));
    let exp_final_layer_outputs_summed: Vec<f64> = predictions.sum_columns();

    // Divide all data by col sum
    for output_neuron in 0..predictions.rows {
      for observation_number in 0..predictions.columns {
        predictions[output_neuron][observation_number] /=
          exp_final_layer_outputs_summed[observation_number];
      }
    }

    return predictions;
  }

  pub fn backpropogation_output_layer_regression(
    &mut self,
    labels: &Vec<f64>,
    neuron_outputs: &Vec<Matrix>,
    learning_rate: f64,
  ) -> Matrix {
    let output_layer_index = self.biases.len() - 1;
    let prev_layer_outputs = &neuron_outputs[output_layer_index - 1];
    let output_biases = &self.biases[output_layer_index];
    let output_weights = &self.weights[output_layer_index];
    let normalization_factor = 1.0 / prev_layer_outputs.columns as f64;

    // Shared error calculations (dSSR)
    // neuron_output[output_layer_index][0] because there is only one output neuron
    let error = Matrix::new_2d(vec![(0..labels.len())
      .map(|index| -2.0 * (labels[index] - neuron_outputs[output_layer_index][0][index]))
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
    predicted_probabilities: &Matrix,
    labels: &Vec<f64>,
    neuron_outputs: &Vec<Matrix>,
    learning_rate: f64,
  ) -> Matrix {
    let output_layer_index = self.biases.len() - 1;
    let prev_layer_outputs = &neuron_outputs[output_layer_index - 1];
    let output_biases = &self.biases[output_layer_index];
    let output_weights = &self.weights[output_layer_index];
    let normalization_factor = 1.0 / prev_layer_outputs.columns as f64;

    // Shared error calculations (dCE * dSoftmax)
    let error = Matrix::new_2d(
      (0..output_biases.rows)
        .map(|index| {
          izip!(labels.iter(), predicted_probabilities[index].iter())
            .map(|(label, predicted_probability)| {
              if *label == index as f64 {
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
    neuron_outputs: &Vec<Matrix>,
    next_layer_error: &Matrix,
    learning_rate: f64,
    activation_function: &Box<dyn ActivationFunction>,
    layer: usize,
  ) {
    // Used for yin
    let prev_layer_outputs = if layer == 0 {
      observations
    } else {
      &neuron_outputs[layer - 1]
    };

    let normalization_factor = 1.0 / prev_layer_outputs.columns as f64;

    // Used for wout
    let wout = &self.weights[layer + 1];

    // Used for x
    //POSSIBLE OPTIMIZATION: Cache at feed forward step
    // ASSUMPTION: Since using ReLU, I can just take activation_function_prime of layer output
    let activation_prime_x =
      neuron_outputs[layer].element_apply(&|x| activation_function.activation_function_prime(x));

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
      self.backpropogation_hidden_layer(
        observations,
        neuron_outputs,
        &error,
        learning_rate,
        activation_function,
        layer - 1,
      );
    }
  }

  fn rust_regression(&self, observations: &Matrix) -> Vec<f64> {
    let num_observations = observations.columns;

    let mut neuron_outputs: Vec<Matrix> = self
      .weights
      .iter()
      .map(|layer| Matrix::zeros(layer.data.len(), num_observations))
      .collect_vec();

    let activation_function: Box<dyn ActivationFunction> = Box::new(Relu {});
    self.feed_forward(&observations, &mut neuron_outputs, &activation_function);
    let classifications = neuron_outputs[neuron_outputs.len() - 1][0].to_vec();

    return classifications;
  }

  fn test_train_performance_regression(&self, observations: &Matrix, labels: &Vec<f64>) {
    let classifications = self.rust_regression(&observations);
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

    let percent_correct = 100.0 * num_correct / labels.len() as f64;

    py_print(&format!("% Correct: {}", percent_correct));
  }

  fn test_train_performance_classification(&self, observations: &Matrix, labels: &Vec<f64>) {
    let classifications = self.rust_classify(&observations);
    let num_correct = izip!(classifications.iter(), labels.iter())
      .fold(0.0, |acc, (classification, label)| {
        acc + if classification == label { 1.0 } else { 0.0 }
      });

    let percent_correct = 100.0 * num_correct / labels.len() as f64;

    py_print(&format!("% Correct: {}", percent_correct));
  }
}
