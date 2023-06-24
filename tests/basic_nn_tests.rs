#[cfg(test)]
mod basic_nn_tests {

  use itertools::izip;
  use Rust_Machine_Learning::basic_neural_network::{ActivationFunction, BasicNeuralNetwork, Relu};
  use Rust_Machine_Learning::matrix_lib::Matrix;

  #[test]
  fn feed_forward() {
    let observations = Matrix {
      data: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]], // 3 observations with 2 features
    };

    let weights = vec![
      Matrix {
        data: vec![
          // Layer 1 has 3 neurons, with 2 inputs per neuron
          vec![0.1, 0.2],
          vec![0.3, 0.4],
          vec![0.5, 0.6],
        ],
      },
      Matrix {
        data: vec![
          // Layer 2 has 2 neurons, with 3 inputs per neuron
          vec![0.1, 0.2, 0.3],
          vec![0.4, 0.5, 0.6],
        ],
      },
    ];

    let biases = vec![
      Matrix {
        data: vec![
          // Layer 1 biases, 3 neurons
          vec![0.1],
          vec![0.2],
          vec![0.3],
        ],
      },
      Matrix {
        data: vec![
          // Layer 2 biases, 2 neurons
          vec![0.1],
          vec![0.2],
        ],
      },
    ];

    let network = BasicNeuralNetwork { weights, biases };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let mut neuron_outputs = vec![
      Matrix {
        // 3 neurons x 3 observations
        data: vec![vec![0.0; 3]; 3],
      },
      Matrix {
        // 2 neurons x 3 observations
        data: vec![vec![0.0; 3]; 2],
      },
    ];

    let activation_function: Box<dyn ActivationFunction> = Box::new(Relu {});

    network.feed_forward(&observations, &mut neuron_outputs, &activation_function);

    let expected_neuron_outputs = vec![
      Matrix {
        // 3 neurons x 3 observations
        data: vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ],
      },
      Matrix {
        // 2 neurons x 3 observations
        data: vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]],
      },
    ];

    for (a, b) in izip!(expected_neuron_outputs, neuron_outputs) {
      a.print();
      b.print();
      assert!(matrix_are_equal(a, b, 12));
    }
  }

  #[test]
  fn soft_max() {
    let observations = Matrix {
      data: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]], // 3 observations with 2 features
    };

    let weights = vec![
      Matrix {
        data: vec![
          // Layer 1 has 3 neurons, with 2 inputs per neuron
          vec![0.1, 0.2],
          vec![0.3, 0.4],
          vec![0.5, 0.6],
        ],
      },
      Matrix {
        data: vec![
          // Layer 2 has 2 neurons, with 3 inputs per neuron
          vec![0.1, 0.2, 0.3],
          vec![0.4, 0.5, 0.6],
        ],
      },
    ];

    let biases = vec![
      Matrix {
        data: vec![
          // Layer 1 biases, 3 neurons
          vec![0.1],
          vec![0.2],
          vec![0.3],
        ],
      },
      Matrix {
        data: vec![
          // Layer 2 biases, 2 neurons
          vec![0.1],
          vec![0.2],
        ],
      },
    ];

    let network = BasicNeuralNetwork { weights, biases };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let mut neuron_outputs = vec![
      Matrix {
        // 3 neurons x 3 observations
        data: vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ],
      },
      Matrix {
        // 2 neurons x 3 observations
        data: vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]],
      },
    ];

    let softmax_output = BasicNeuralNetwork::softmax(&neuron_outputs);

    let expected_softmax_output = Matrix {
      // 2 neurons x 3 observations
      data: vec![vec![0.3872, 0.3742, 0.3596], vec![0.6079, 0.6257, 0.6403]],
    };

    expected_softmax_output.print();
    softmax_output.print();
    assert!(matrix_are_equal(expected_softmax_output, softmax_output, 2));
  }

  #[test]
  fn backpropogation_output_layer() {
    let observations = Matrix {
      data: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]], // 3 observations with 2 features
    };

    let labels = vec![1.0, 0.0, 1.0];

    let weights = vec![
      Matrix {
        data: vec![
          // Layer 1 has 3 neurons, with 2 inputs per neuron
          vec![0.1, 0.2],
          vec![0.3, 0.4],
          vec![0.5, 0.6],
        ],
      },
      Matrix {
        data: vec![
          // Layer 2 has 2 neurons, with 3 inputs per neuron
          vec![0.1, 0.2, 0.3],
          vec![0.4, 0.5, 0.6],
        ],
      },
    ];

    let biases = vec![
      Matrix {
        data: vec![
          // Layer 1 biases, 3 neurons
          vec![0.1],
          vec![0.2],
          vec![0.3],
        ],
      },
      Matrix {
        data: vec![
          // Layer 2 biases, 2 neurons
          vec![0.1],
          vec![0.2],
        ],
      },
    ];

    let mut network = BasicNeuralNetwork { weights, biases };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let mut neuron_outputs = vec![
      Matrix {
        // 3 neurons x 3 observations
        data: vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ],
      },
      Matrix {
        // 2 neurons x 3 observations
        data: vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]],
      },
    ];

    let predicted_probabilities = Matrix {
      data: vec![vec![0.3872, 0.3742, 0.3596], vec![0.6079, 0.6257, 0.6403]],
    };
    let learning_rate = 0.1;

    // Backprop output layer
    network.backpropogation_output_layer(
      &predicted_probabilities,
      &labels,
      &neuron_outputs,
      learning_rate,
    );

    let expected_weights = vec![
      Matrix {
        data: vec![
          // Layer 1 has 3 neurons, with 2 inputs per neuron
          vec![0.1, 0.2],
          vec![0.3, 0.4],
          vec![0.5, 0.6],
        ],
      },
      Matrix {
        data: vec![
          // Layer 2 has 2 neurons, with 3 inputs per neuron
          vec![0.1, 0.2, 0.3],
          vec![0.4, 0.5, 0.6],
        ],
      },
    ];

    let expected_biases = vec![
      Matrix {
        data: vec![
          // Layer 1 biases, 3 neurons
          vec![0.1],
          vec![0.2],
          vec![0.3],
        ],
      },
      Matrix {
        data: vec![
          // Layer 2 biases, 2 neurons
          vec![0.1121],
          vec![0.18739],
        ],
      },
    ];

    izip!(
      expected_weights,
      network.weights,
      expected_biases,
      network.biases
    )
    .for_each(|(ew, w, eb, b)| {
      eb.print();
      b.print();
      ew.print();
      w.print();

      assert!(matrix_are_equal(eb, b, 12));
      assert!(matrix_are_equal(ew, w, 12));
    })
  }

  fn matrix_are_equal(a: Matrix, b: Matrix, precision: usize) -> bool {
    if a.get_rows() != b.get_rows() || a.get_columns() != b.get_columns() {
      return false;
    }

    for i in 0..a.get_rows() {
      for j in 0..a.get_columns() {
        if !approx_equal(a.data[i][j], b.data[i][j], precision) {
          return false;
        }
      }
    }

    return true;
  }

  fn approx_equal(a: f64, b: f64, precision: usize) -> bool {
    let tolerance = f64::powf(10.0, -1.0 * precision as f64);
    return (a - b).abs() < tolerance;
  }
}
