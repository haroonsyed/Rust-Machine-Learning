#[cfg(test)]
mod basic_nn_tests {

  use itertools::izip;
  use Rust_Machine_Learning::basic_neural_network::{ActivationFunction, BasicNeuralNetwork, Relu};
  use Rust_Machine_Learning::matrix_lib::Matrix;

  #[test]
  fn feed_forward() {
    let observations = Matrix::new_2d(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]); // 3 observations with 2 features

    let weights = vec![
      Matrix::new_2d(vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
      ]),
    ];

    let biases = vec![
      Matrix::new_2d(vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 biases, 2 neurons
        vec![0.1],
        vec![0.2],
      ]),
    ];

    let network = BasicNeuralNetwork { weights, biases };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let mut neuron_outputs = vec![
      Matrix::new_2d(
        // 3 neurons x 3 observations
        vec![vec![0.0; 3]; 3],
      ),
      Matrix::new_2d(
        // 2 neurons x 3 observations
        vec![vec![0.0; 3]; 2],
      ),
    ];

    let activation_function: Box<dyn ActivationFunction> = Box::new(Relu {});

    network.feed_forward(&observations, &mut neuron_outputs, &activation_function);

    let expected_neuron_outputs = vec![
      Matrix::new_2d(
        // 3 neurons x 3 observations
        vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ],
      ),
      Matrix::new_2d(
        // 2 neurons x 3 observations
        vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]],
      ),
    ];

    for (a, b) in izip!(expected_neuron_outputs, neuron_outputs) {
      a.print();
      b.print();
      assert!(matrix_are_equal(a, b, 12));
    }
  }

  #[test]
  fn soft_max() {
    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let neuron_outputs = vec![
      Matrix::new_2d(
        // 3 neurons x 3 observations
        vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ],
      ),
      Matrix::new_2d(
        // 2 neurons x 3 observations
        vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]],
      ),
    ];

    let softmax_output = BasicNeuralNetwork::softmax(&neuron_outputs);

    let expected_softmax_output = Matrix::new_2d(
      // 2 neurons x 3 observations
      vec![vec![0.3872, 0.3742, 0.3596], vec![0.6079, 0.6257, 0.6403]],
    );

    expected_softmax_output.print();
    softmax_output.print();
    assert!(matrix_are_equal(expected_softmax_output, softmax_output, 2));
  }

  #[test]
  fn backpropogation_output_layer() {
    let labels = vec![1.0, 0.0, 1.0];

    let weights = vec![
      Matrix::new_2d(vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
      ]),
    ];

    let biases = vec![
      Matrix::new_2d(vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 biases, 2 neurons
        vec![0.1],
        vec![0.2],
      ]),
    ];

    let mut network = BasicNeuralNetwork { weights, biases };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let neuron_outputs = vec![
      Matrix::new_2d(
        // 3 neurons x 3 observations
        vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ],
      ),
      Matrix::new_2d(
        // 2 neurons x 3 observations
        vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]],
      ),
    ];

    let predicted_probabilities = Matrix::new_2d(vec![
      vec![0.3872, 0.3742, 0.3596],
      vec![0.6079, 0.6257, 0.6403],
    ]);
    let learning_rate = 0.1;

    // Backprop output layer
    network.backpropogation_output_layer_classification(
      &predicted_probabilities,
      &labels,
      &neuron_outputs,
      learning_rate,
    );

    let expected_weights = vec![
      Matrix::new_2d(vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![
          0.09914026666666667,
          0.19820906666666668,
          0.29727786666666667,
        ],
        vec![0.40089233333333335, 0.5018579333333333, 0.6028235333333333],
      ]),
    ];

    let expected_biases = vec![
      Matrix::new_2d(vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 biases, 2 neurons
        vec![0.09596666666666667],
        vec![0.20420333333333335],
      ]),
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

      assert!(matrix_are_equal(eb, b, 8));
      assert!(matrix_are_equal(ew, w, 8));
    })
  }

  #[test]
  fn backpropogation_hidden_layer() {
    let observations = Matrix::new_2d(
      vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]], // 3 observations with 2 features
    );

    let weights = vec![
      Matrix::new_2d(vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![
          0.09914026666666667,
          0.19820906666666668,
          0.29727786666666667,
        ],
        vec![0.40089233333333335, 0.5018579333333333, 0.6028235333333333],
      ]),
    ];

    let biases = vec![
      Matrix::new_2d(vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 biases, 2 neurons
        vec![0.09596666666666667],
        vec![0.20420333333333335],
      ]),
    ];

    let mut network = BasicNeuralNetwork { weights, biases };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let neuron_outputs = vec![
      Matrix::new_2d(
        // 3 neurons x 3 observations
        vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ],
      ),
      Matrix::new_2d(
        // 2 neurons x 3 observations
        vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]],
      ),
    ];

    let activation_func: Box<dyn ActivationFunction> = Box::new(Relu {});
    let learning_rate = 0.1;

    let output_error = Matrix::new_2d(vec![
      vec![0.3872, -0.6258, 0.3596],
      vec![-0.3921, 0.6257, -0.3596],
    ]);

    // Backprop output layer
    network.backpropogation_hidden_layer(
      &observations,
      &neuron_outputs,
      &output_error,
      learning_rate,
      &activation_func,
      network.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
    );

    let expected_weights = vec![
      Matrix::new_2d(vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.10022246731331112, 0.20060763193064446],
        vec![0.3002255393082444, 0.4006180473335778],
        vec![0.5002286113031778, 0.600628462736511],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![
          0.09914026666666667,
          0.19820906666666668,
          0.29727786666666667,
        ],
        vec![0.40089233333333335, 0.5018579333333333, 0.6028235333333333],
      ]),
    ];

    let expected_biases = vec![
      Matrix::new_2d(vec![
        // Layer 1 biases, 3 neurons
        vec![0.1012838820577777],
        vec![0.20130836008444444],
        vec![0.3013328381111111],
      ]),
      Matrix::new_2d(vec![
        // Layer 2 biases, 2 neurons
        vec![0.09596666666666667],
        vec![0.20420333333333335],
      ]),
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

      assert!(matrix_are_equal(eb, b, 8));
      assert!(matrix_are_equal(ew, w, 8));
    })
  }

  fn matrix_are_equal(a: Matrix, b: Matrix, precision: usize) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
      return false;
    }

    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a[i][j], b[i][j], precision) {
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
