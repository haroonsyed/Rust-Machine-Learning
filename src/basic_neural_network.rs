use crate::matrix_lib::Matrix;
use itertools::Itertools;
use pyo3::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution, Rng};

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

trait ActivationFunction {
  fn activation_function(&self, x: f64) -> f64;
  fn activation_function_prime(&self, x: f64) -> f64;
}

struct Relu;
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
  weights: Vec<Matrix>,
  bias: Vec<Matrix>,
}

#[pymethods]
impl BasicNeuralNetwork {
  #[new]
  fn new(
    features_train: Vec<Vec<f64>>,
    input_labels: Vec<f64>,
    hidden_layer_sizes: Vec<usize>,
    num_classifications: usize,
    learning_rate: f64,
  ) -> Self {
    // Gather data on dimensions for matrices
    let num_observations = features_train.len();
    let mut non_input_layer_sizes = hidden_layer_sizes.clone();
    non_input_layer_sizes.push(num_classifications);

    // Init the matrices
    let observations = Matrix {
      data: features_train,
    }
    .transpose();

    // Random seed for weights
    let mut rng = rand::thread_rng();
    let mut range = Uniform::new(-1.0, 1.0);
    let mut weights = (0..non_input_layer_sizes.len())
      .map(|layer| Matrix {
        data: vec![
          vec![
            range.sample(&mut rng);
            if layer == 0 {
              num_observations
            } else {
              non_input_layer_sizes[layer - 1]
            }
          ];
          non_input_layer_sizes[layer]
        ],
      })
      .collect_vec();

    let mut biases = (0..non_input_layer_sizes.len())
      .map(|layer| Matrix {
        data: vec![vec![0.0; non_input_layer_sizes[layer]]],
      })
      .collect_vec();

    let mut neuron_outputs: Vec<Matrix> = non_input_layer_sizes
      .iter()
      .map(|&layer_size| Matrix {
        data: vec![vec![0.0; layer_size]; num_observations],
      })
      .collect();

    // Create network
    let network = BasicNeuralNetwork {
      weights: Vec::new(),
      bias: Vec::new(),
    };

    // Train model
    network.train(
      &observations,
      &mut weights,
      &mut biases,
      &mut neuron_outputs,
      &input_labels,
      learning_rate,
      &Relu {},
    );

    // Cleanup and return
    return network;
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    // Feed forward through network

    return Ok(labels);
  }
}

impl BasicNeuralNetwork {
  fn train(
    &self,
    observations: &Matrix,
    weights: &mut Vec<Matrix>,
    biases: &mut Vec<Matrix>,
    neuron_outputs: &mut Vec<Matrix>,
    labels: &Vec<f64>,
    learning_rate: f64,
    activation_function: &dyn ActivationFunction,
  ) {
  }
}
