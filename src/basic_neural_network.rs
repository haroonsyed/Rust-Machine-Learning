use crate::matrix_lib::Matrix;
use pyo3::prelude::*;

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
fn relu(x: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  } else {
    return x;
  }
}
fn relu_prime(x: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  } else {
    return 1.0;
  }
}

#[pyclass]
pub struct BasicNeuralNetwork {
  weights: Matrix,
  bias: Matrix,
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
    let num_layers = hidden_layer_sizes.len();
    let num_neurons_largest_layer = *hidden_layer_sizes.iter().max().unwrap();
    let num_observations = features_train.len();

    // Init the matrices
    // let observations = Matrix {
    //   data: features_train,
    // }; // NEEDS TO BE TRANSPOSED!

    let mut observations = Matrix { data: Vec::new() };
    let mut neuron_outputs: Vec<Matrix> = Vec::new();
    let activation_function = relu;

    // Train model

    return BasicNeuralNetwork {
      weights: Matrix { data: Vec::new() },
      bias: Matrix { data: Vec::new() },
    };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    // Feed forward through network

    return Ok(labels);
  }
}
