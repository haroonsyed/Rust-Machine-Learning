use crate::{matrix_lib::Matrix, py_util::py_print};
use itertools::{izip, Itertools};
use pyo3::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution};

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
    num_classifications: usize,
    learning_rate: f64,
    num_iterations: usize,
  ) -> Self {
    // Gather data on dimensions for matrices
    let num_observations = features_train.len();
    let num_features = features_train[0].len();
    let mut non_input_layer_sizes = hidden_layer_sizes.clone();
    non_input_layer_sizes.push(num_classifications);

    py_print(&format!("{} {}", num_observations, num_features));

    // Init the matrices
    let observations = Matrix {
      data: features_train,
    }
    .transpose();

    // Random seed for weights
    let mut rng = rand::thread_rng();
    let range = Uniform::new(-1.0, 1.0);

    let weights = (0..non_input_layer_sizes.len())
      .map(|layer| Matrix {
        data: (0..non_input_layer_sizes[layer])
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
      })
      .collect_vec();

    let biases = (0..non_input_layer_sizes.len())
      .map(|layer| Matrix {
        data: vec![vec![0.0]; non_input_layer_sizes[layer]],
      })
      .collect_vec();

    let mut neuron_outputs: Vec<Matrix> = non_input_layer_sizes
      .iter()
      .map(|&layer_size| Matrix {
        data: vec![vec![0.0; num_observations]; layer_size],
      })
      .collect();

    // Create network
    let mut network = BasicNeuralNetwork { weights, biases };

    // Train model
    network.train(
      &observations,
      &mut neuron_outputs,
      &input_labels,
      learning_rate,
      Box::new(Relu {}),
      num_iterations,
    );

    // Cleanup and return
    return network;
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let num_observations = features_test.len();

    // Feed forward through network
    let observations = Matrix {
      data: features_test,
    }
    .transpose();

    let mut neuron_outputs: Vec<Matrix> = self
      .weights
      .iter()
      .map(|layer| Matrix {
        data: vec![vec![0.0; num_observations]; layer.data.len()],
      })
      .collect_vec();

    observations.print();

    let activation_function: Box<dyn ActivationFunction> = Box::new(Relu {});
    self.feed_forward(&observations, &mut neuron_outputs, &activation_function);
    let predicted_probabilities = Self::softmax(&neuron_outputs);
    let classifications = Self::get_classification(&predicted_probabilities);

    return Ok(classifications);
  }
}

impl BasicNeuralNetwork {
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

    py_print(&"IN SOFTMAX");
    neuron_outputs[neuron_outputs.len() - 1].print();
    py_print(&"PREDICTIONS");
    predictions.print();
    py_print(&"EXP SUMMED");
    py_print(&exp_final_layer_outputs_summed);

    // Divide all data by col sum
    for output_neuron in 0..predictions.get_rows() {
      for observation_number in 0..predictions.get_columns() {
        predictions.data[output_neuron][observation_number] /=
          exp_final_layer_outputs_summed[observation_number];
      }
    }

    return predictions;
  }

  pub fn get_classification(predicted_probabilities: &Matrix) -> Vec<f64> {
    return predicted_probabilities
      .transpose()
      .data
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

  pub fn backpropogation_output_layer(
    &mut self,
    predicted_probabilities: &Matrix,
    labels: &Vec<f64>,
    neuron_outputs: &Vec<Matrix>,
    learning_rate: f64,
  ) {
    let output_layer_index = self.biases.len() - 1;
    let prev_layer_outputs = &neuron_outputs[output_layer_index - 1];

    // Update biases first
    // b' = b + learning_rate * batch_sum( if label==Output bias codes for {predicted coded for -1} else {predicted coded for} )
    let output_biases = &mut self.biases[output_layer_index].data;
    for bias_index in 0..output_biases.len() {
      let bias = &mut output_biases[bias_index][0];
      let bias_neuron_predicted = &predicted_probabilities.data[bias_index];

      let db = izip!(bias_neuron_predicted.iter(), labels.iter()).fold(
        0.0,
        |acc, (neuron_prediction, label)| {
          acc
            + if *label == bias_index as f64 {
              *neuron_prediction - 1.0
            } else {
              *neuron_prediction
            }
        },
      );

      *bias = *bias + learning_rate * db;
    }

    // Update weights
    // w' = w * learning_rate * batch_sum( Output bias codes for Yin * {predicted coded for -1} else Yin * {predicted coded for} )
    let output_weights = &mut self.weights[output_layer_index];
    for output_neuron_index in 0..output_weights.get_rows() {
      for incoming_weight_index in 0..output_weights.get_columns() {
        let weight = &mut output_weights.data[output_neuron_index][incoming_weight_index];
        let w_prev_layer_outputs = &prev_layer_outputs.data[incoming_weight_index];

        let dw = izip!(
          predicted_probabilities.data[output_neuron_index].iter(),
          labels.iter(),
          w_prev_layer_outputs.iter()
        )
        .fold(0.0, |acc, (sample_prediction, label, yin)| {
          acc
            + yin
              * (if *label == output_neuron_index as f64 {
                *sample_prediction - 1.0
              } else {
                *sample_prediction
              })
        });
        *weight = *weight + learning_rate * dw;
      }
    }
  }

  pub fn backpropogation_hidden_layer(
    &mut self,
    observations: &Matrix,
    neuron_outputs: &Vec<Matrix>,
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

    // Used for wout
    let wout = self.weights[layer + 1].sum_columns();

    // Used for x
    //POSSIBLE OPTIMIZATION: Calculate at feed forward step
    let activation_prime_x = &self.weights[layer]
      .matrix_multiply(prev_layer_outputs)
      .add_vector_to_columns(&self.biases[layer])
      .element_apply(&|x| activation_function.activation_function_prime(x));

    // Update biases
    // b' = b + learning_rate * sum(Wout) * batch_sum(activation'(x))
    let layer_biases = &mut self.biases[layer];
    let weight_summed_activation_prime_x = activation_prime_x.sum_rows();
    for bias_index in 0..layer_biases.data.len() {
      let bias = &mut layer_biases.data[bias_index][0];

      let db = wout[bias_index] * weight_summed_activation_prime_x[bias_index];

      *bias = *bias + learning_rate * db;
    }

    // Update weights
    // b' = b + learning_rate * sum(Wout) * batch_sum(activation'(x))
    let layer_weights = &mut self.weights[layer];
    for neuron_index in 0..layer_weights.get_rows() {
      for incoming_weight_index in 0..layer_weights.get_columns() {
        let weight = &mut layer_weights.data[neuron_index][incoming_weight_index];

        let dw = wout[neuron_index]
          * izip!(
            activation_prime_x.data[neuron_index].iter(),
            prev_layer_outputs.data[incoming_weight_index].iter()
          )
          .fold(0.0, |acc, (sample_active_prime_x, yin)| {
            acc + sample_active_prime_x * yin
          });

        *weight = *weight + learning_rate * dw;
      }
    }

    if layer != 0 {
      self.backpropogation_hidden_layer(
        observations,
        neuron_outputs,
        learning_rate,
        activation_function,
        layer - 1,
      );
    }
  }

  pub fn train(
    &mut self,
    observations: &Matrix,
    neuron_outputs: &mut Vec<Matrix>,
    labels: &Vec<f64>,
    learning_rate: f64,
    activation_function: Box<dyn ActivationFunction>,
    num_iterations: usize,
  ) {
    // For now we will make the number of iterations a constant
    for _i in 0..num_iterations {
      py_print(&format!("Starting iteration {}", _i));
      self.feed_forward(observations, neuron_outputs, &activation_function);
      py_print(&format!("Finished feed forward for iteration {}", _i));
      self.biases.iter().for_each(|x| x.print());
      self.weights.iter().for_each(|x| x.print());

      let predicted_probabilities = Self::softmax(neuron_outputs);
      py_print(&format!("Finished softmax for iteration {}", _i));
      self.biases.iter().for_each(|x| x.print());
      self.weights.iter().for_each(|x| x.print());

      self.backpropogation_output_layer(
        &predicted_probabilities,
        labels,
        neuron_outputs,
        learning_rate,
      );
      py_print(&format!(
        "Finished backprop output layer for iteration {}",
        _i
      ));
      self.biases.iter().for_each(|x| x.print());
      self.weights.iter().for_each(|x| x.print());

      self.backpropogation_hidden_layer(
        observations,
        neuron_outputs,
        learning_rate,
        &activation_function,
        self.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
      );
      py_print(&format!(
        "Finished backprop hidden layers for iteration {}",
        _i
      ));

      self.biases.iter().for_each(|x| x.print());
      self.weights.iter().for_each(|x| x.print());
    }
  }
}
