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
  biases: Vec<Matrix>,
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
    let network = BasicNeuralNetwork { weights, biases };

    // Train model
    network.train(
      &observations,
      &mut neuron_outputs,
      &input_labels,
      learning_rate,
      Box::new(Relu {}),
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
  fn feed_forward(
    &self,
    observations: &Matrix,
    neuron_outputs: &mut Vec<Matrix>,
    activation_function: Box<dyn ActivationFunction>,
  ) {
    let num_layers = self.weights.len();
    for layer in 0..num_layers {
      neuron_outputs[layer] = self.weights[layer]
        .matrix_multiply(if layer == 0 {
          observations
        } else {
          &neuron_outputs[layer - 1]
        })
        .element_add(&self.biases[layer])
        .element_apply(&|x| activation_function.activation_function(x));
    }
  }

  fn softmax(neuron_outputs: &Vec<Matrix>) -> Matrix {
    let mut exp_final_layer_outputs =
      neuron_outputs[neuron_outputs.len() - 1].element_apply(&|x| f64::exp(x));
    let exp_final_layer_outputs_summed: Vec<f64> = exp_final_layer_outputs
      .data
      .iter()
      .map(|predictions| predictions.iter().sum())
      .collect_vec();

    // Divide all data by layer data
    for observation_number in 0..exp_final_layer_outputs.get_rows() {
      for output in 0..exp_final_layer_outputs.get_columns() {
        exp_final_layer_outputs.data[observation_number][output] /=
          exp_final_layer_outputs_summed[observation_number];
      }
    }

    return exp_final_layer_outputs;
  }

  fn train(
    &self,
    observations: &Matrix,
    neuron_outputs: &mut Vec<Matrix>,
    labels: &Vec<f64>,
    learning_rate: f64,
    activation_function: Box<dyn ActivationFunction>,
  ) {
    // For now we will make the number of iterations a constant
    let num_iterations = 1;
  }
}
