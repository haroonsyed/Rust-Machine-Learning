#[cfg(test)]
mod basic_nn_tests {

  use itertools::{izip, Itertools};
  use matrix_lib::lib_cpu::MatrixCpu;
  use matrix_lib::Matrix;
  use rand::prelude::Distribution;
  use rust_machine_learning::basic_neural_network::BasicNeuralNetworkRust;
  use rust_machine_learning::cpu_basic_neural_network::{
    ActivationFunction, BasicNeuralNetworkCPURust, Relu,
  };
  use statrs::distribution::Normal;

  #[test]
  fn feed_forward() {
    let observations = Matrix::new_2d(&vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]); // 3 observations with 2 features

    let non_input_layer_sizes = vec![3, 2];

    let weights = vec![
      Matrix::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
      ]),
    ];

    let biases = vec![
      Matrix::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 biases, 2 neurons
        vec![0.1],
        vec![0.2],
      ]),
    ];

    let network = BasicNeuralNetworkRust {
      non_input_layer_sizes,
      weights,
      biases,
    };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let mut neuron_outputs = vec![
      Matrix::new_2d(&
        // 3 neurons x 3 observations
        vec![vec![0.0; 3]; 3]),
      Matrix::new_2d(&
        // 2 neurons x 3 observations
        vec![vec![0.0; 3]; 2]),
    ];

    network.feed_forward(&observations, &mut neuron_outputs);

    let expected_neuron_outputs = vec![
      Matrix::new_2d(&
        // 3 neurons x 3 observations
        vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ]),
      Matrix::new_2d(&
        // 2 neurons x 3 observations
        vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]]),
    ];

    for (a, b) in izip!(expected_neuron_outputs, neuron_outputs) {
      a.print();
      b.print();
      assert!(matrix_are_equal(&a, &b, 5));
    }
  }

  #[test]
  fn soft_max() {
    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let neuron_outputs = vec![
      Matrix::new_2d(&
        // 3 neurons x 3 observations
        vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ]),
      Matrix::new_2d(&
        // 2 neurons x 3 observations
        vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]]),
    ];

    let softmax_output = BasicNeuralNetworkRust::softmax(&neuron_outputs);

    let expected_softmax_output = Matrix::new_2d(&
      // 2 neurons x 3 observations
      vec![vec![0.3872, 0.3742, 0.3596], vec![0.6079, 0.6257, 0.6403]]);

    assert!(matrix_are_equal(
      &expected_softmax_output,
      &softmax_output,
      2
    ));
  }

  #[test]
  fn backpropogation_output_layer() {
    let labels = vec![1.0, 0.0, 1.0];
    let non_input_layer_sizes = vec![3, 2];

    let weights = vec![
      Matrix::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
      ]),
    ];

    let biases = vec![
      Matrix::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 biases, 2 neurons
        vec![0.1],
        vec![0.2],
      ]),
    ];

    let mut network = BasicNeuralNetworkRust {
      non_input_layer_sizes,
      weights,
      biases,
    };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let neuron_outputs = vec![
      Matrix::new_2d(&
        // 3 neurons x 3 observations
        vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ]),
      Matrix::new_2d(&
        // 2 neurons x 3 observations
        vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]]),
    ];

    let predicted_probabilities = Matrix::new_2d(&vec![
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
      Matrix::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(&vec![
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
      Matrix::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(&vec![
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

      assert!(matrix_are_equal(&eb, &b, 5));
      assert!(matrix_are_equal(&ew, &w, 5));
    })
  }

  #[test]
  fn backpropogation_hidden_layer() {
    let observations = Matrix::new_2d(
      &vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]], // 3 observations with 2 features
    );

    let non_input_layer_sizes = vec![3, 2];

    let weights = vec![
      Matrix::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(&vec![
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
      Matrix::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 biases, 2 neurons
        vec![0.09596666666666667],
        vec![0.20420333333333335],
      ]),
    ];

    let mut network = BasicNeuralNetworkRust {
      non_input_layer_sizes,
      weights,
      biases,
    };

    // Create a matrix to hold the actual output
    // Remember each output is product of matmul between weights and observations/neuron_outputs[layer-1] + (bias to each column)
    let neuron_outputs = vec![
      Matrix::new_2d(&
        // 3 neurons x 3 observations
        vec![
          vec![0.19, 0.22, 0.25],
          vec![0.39, 0.46, 0.53],
          vec![0.59, 0.7, 0.81],
        ]),
      Matrix::new_2d(&
        // 2 neurons x 3 observations
        vec![vec![0.374, 0.424, 0.474], vec![0.825, 0.938, 1.051]]),
    ];

    let learning_rate = 0.1;

    let output_error = Matrix::new_2d(&vec![
      vec![0.3872, -0.6258, 0.3596],
      vec![-0.3921, 0.6257, -0.3596],
    ]);

    // Backprop output layer
    network.backpropogation_hidden_layer(
      &observations,
      &neuron_outputs,
      &output_error,
      learning_rate,
      network.weights.len() - 2, // Start at final-1 layer, recursion will do the rest
    );

    let expected_weights = vec![
      Matrix::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.10022246731331112, 0.20060763193064446],
        vec![0.3002255393082444, 0.4006180473335778],
        vec![0.5002286113031778, 0.600628462736511],
      ]),
      Matrix::new_2d(&vec![
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
      Matrix::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1012838820577777],
        vec![0.20130836008444444],
        vec![0.3013328381111111],
      ]),
      Matrix::new_2d(&vec![
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

      assert!(matrix_are_equal(&eb, &b, 5));
      assert!(matrix_are_equal(&ew, &w, 5));
    })
  }

  #[test]
  fn full_classification_train() {
    let observations = vec![vec![0.1, 0.4], vec![0.2, 0.5], vec![0.3, 0.6]]; // 3 observations with 2 features
    let labels = vec![1.0, 0.0, 1.0];

    let non_input_layer_sizes = vec![3, 2];

    let weights = vec![
      Matrix::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
      ]),
    ];

    let biases = vec![
      Matrix::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 biases, 2 neurons
        vec![0.1],
        vec![0.2],
      ]),
    ];

    let mut neuron_outputs = vec![
      Matrix::new_2d(&
        // 3 neurons x 3 observations
        vec![vec![0.0; 3]; 3]),
      Matrix::new_2d(&
        // 2 neurons x 3 observations
        vec![vec![0.0; 3]; 2]),
    ];

    let mut network = BasicNeuralNetworkRust {
      non_input_layer_sizes,
      weights,
      biases,
    };
    let learning_rate = 0.1;
    network.train_classification(
      observations,
      &mut neuron_outputs,
      labels,
      learning_rate,
      1,
      0,
    );

    let expected_weights = vec![
      Matrix::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.10022246731331112, 0.20060763193064446],
        vec![0.3002255393082444, 0.4006180473335778],
        vec![0.5002286113031778, 0.600628462736511],
      ]),
      Matrix::new_2d(&vec![
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
      Matrix::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1012838820577777],
        vec![0.20130836008444444],
        vec![0.3013328381111111],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 biases, 2 neurons
        vec![0.09596666666666667],
        vec![0.20420333333333335],
      ]),
    ];

    for (a, b) in izip!(network.weights, expected_weights) {
      assert!(matrix_are_equal(&a, &b, 3));
    }
    for (a, b) in izip!(network.biases, expected_biases) {
      assert!(matrix_are_equal(&a, &b, 3));
    }
  }

  #[test]
  fn cpu_gpu_agreement() {
    let observations_gpu = Matrix::new_2d(
      &vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]], // 3 observations with 2 features
    );
    let observations_cpu = MatrixCpu::new_2d(
      &vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]], // 3 observations with 2 features
    );
    let labels = vec![1.0, 0.0, 1.0];

    let non_input_layer_sizes = vec![3, 2];

    let weights_gpu = vec![
      Matrix::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
      ]),
    ];

    let biases_gpu = vec![
      Matrix::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      Matrix::new_2d(&vec![
        // Layer 2 biases, 2 neurons
        vec![0.1],
        vec![0.2],
      ]),
    ];

    let weights_cpu = vec![
      MatrixCpu::new_2d(&vec![
        // Layer 1 has 3 neurons, with 2 inputs per neuron
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
      ]),
      MatrixCpu::new_2d(&vec![
        // Layer 2 has 2 neurons, with 3 inputs per neuron
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
      ]),
    ];

    let biases_cpu = vec![
      MatrixCpu::new_2d(&vec![
        // Layer 1 biases, 3 neurons
        vec![0.1],
        vec![0.2],
        vec![0.3],
      ]),
      MatrixCpu::new_2d(&vec![
        // Layer 2 biases, 2 neurons
        vec![0.1],
        vec![0.2],
      ]),
    ];

    let mut neuron_outputs_gpu = vec![
      Matrix::new_2d(&
        // 3 neurons x 3 observations
        vec![vec![0.0; 3]; 3]),
      Matrix::new_2d(&
        // 2 neurons x 3 observations
        vec![vec![0.0; 3]; 2]),
    ];

    let mut neuron_outputs_cpu = vec![
      MatrixCpu::new_2d(&
        // 3 neurons x 3 observations
        vec![vec![0.0; 3]; 3]),
      MatrixCpu::new_2d(&
        // 2 neurons x 3 observations
        vec![vec![0.0; 3]; 2]),
    ];

    // Create the networks
    let mut gpu_network = BasicNeuralNetworkRust {
      non_input_layer_sizes,
      weights: weights_gpu,
      biases: biases_gpu,
    };
    let mut cpu_network = BasicNeuralNetworkCPURust {
      weights: weights_cpu,
      biases: biases_cpu,
    };
    let learning_rate = 0.1;

    let activation_func: Box<dyn ActivationFunction> = Box::new(Relu {});

    let mut rng = rand::thread_rng();
    let range = Normal::new(0.0, 1.0).unwrap();
    // Run the networks, compare outputs at each stage
    for i in 0..1000 {
      println!("Iteration: {}", i);
      // FEED FORWARD
      println!("TESTING FEED FORWARD");
      gpu_network.feed_forward(&observations_gpu, &mut neuron_outputs_gpu);
      cpu_network.feed_forward(&observations_cpu, &mut neuron_outputs_cpu, &activation_func);
      networks_are_equal(
        &gpu_network,
        &neuron_outputs_gpu,
        &cpu_network,
        &neuron_outputs_cpu,
      );

      // Softmax
      println!("TESTING SOFTMAX");
      let predicted_gpu = BasicNeuralNetworkRust::softmax(&neuron_outputs_gpu);
      let predicted_cpu = BasicNeuralNetworkCPURust::softmax(&neuron_outputs_cpu);
      assert!(matrix_are_equal_gpu_cpu(&predicted_gpu, &predicted_cpu, 5));

      // Backprop output
      println!("TESTING BACKPROP");
      let next_layer_error_gpu = gpu_network.backpropogation_output_layer_classification(
        &predicted_gpu,
        &labels,
        &neuron_outputs_gpu,
        learning_rate,
      );
      let next_layer_error_cpu = cpu_network.backpropogation_output_layer_classification(
        &predicted_cpu,
        &labels,
        &neuron_outputs_cpu,
        learning_rate,
      );
      matrix_are_equal_gpu_cpu(&next_layer_error_gpu, &next_layer_error_cpu, 5);
      networks_are_equal(
        &gpu_network,
        &neuron_outputs_gpu,
        &cpu_network,
        &neuron_outputs_cpu,
      );

      // Backprop hidden
      println!("TESTING BACKPROP HIDDEN");
      gpu_network.backpropogation_hidden_layer(
        &observations_gpu,
        &neuron_outputs_gpu,
        &next_layer_error_gpu,
        learning_rate,
        gpu_network.weights.len() - 2,
      );
      cpu_network.backpropogation_hidden_layer(
        &observations_cpu,
        &neuron_outputs_cpu,
        &next_layer_error_cpu,
        learning_rate,
        &activation_func,
        cpu_network.weights.len() - 2,
      );
      networks_are_equal(
        &gpu_network,
        &neuron_outputs_gpu,
        &cpu_network,
        &neuron_outputs_cpu,
      );

      // Classification
      println!("TESTING CLASSIFICATIONS");
      let random_data = (0..100)
        .map(|_| vec![range.sample(&mut rng) as f32, range.sample(&mut rng) as f32])
        .collect_vec();

      let classifications_cpu = cpu_network.classify(&random_data);
      classifications_cpu.iter().for_each(|x| print!("{} ", x));
      println!();
      let classifications_gpu = gpu_network.classify(&random_data);
      classifications_gpu.iter().for_each(|x| print!("{} ", x));
      println!();
      izip!(classifications_gpu, classifications_cpu).for_each(|(a, b)| assert_eq!(a, b));
    }

    izip!(gpu_network.weights.iter(), cpu_network.weights.iter()).for_each(|(a, b)| {
      a.print();
      b.print();
    });
    izip!(gpu_network.biases.iter(), cpu_network.biases.iter()).for_each(|(a, b)| {
      a.print();
      b.print();
    });
    izip!(neuron_outputs_gpu.iter(), neuron_outputs_cpu.iter()).for_each(|(a, b)| {
      a.print();
      b.print();
    });
    // assert_eq!(1, 2);
  }

  fn matrix_are_equal(a: &Matrix, b: &Matrix, precision: usize) -> bool {
    if a.rows != b.rows || a.columns != b.columns {
      return false;
    }

    a.print();
    b.print();

    let a_data = a.get_data();
    let b_data = b.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b_data[i][j], precision) {
          return false;
        }
      }
    }

    return true;
  }

  fn matrix_are_equal_gpu_cpu(a: &Matrix, b: &MatrixCpu, precision: usize) -> bool {
    a.print();
    b.print();

    if a.rows != b.rows || a.columns != b.columns {
      println!("Matrices do not even share dimensions");
      return false;
    }

    let a_data = a.get_data();
    for i in 0..a.rows {
      for j in 0..a.columns {
        if !approx_equal(a_data[i][j], b[i][j], precision) {
          println!("Matrices not equal at {} {}", a_data[i][j], b[i][j]);
          return false;
        }
      }
    }

    return true;
  }

  fn networks_are_equal(
    gpu_net: &BasicNeuralNetworkRust,
    gpu_outputs: &Vec<Matrix>,
    cpu_net: &BasicNeuralNetworkCPURust,
    cpu_outputs: &Vec<MatrixCpu>,
  ) {
    // Check weights
    izip!(gpu_net.weights.iter(), cpu_net.weights.iter()).for_each(|(a, b)| {
      assert!(matrix_are_equal_gpu_cpu(&a, &b, 5));
    });

    // Check biases
    izip!(gpu_net.biases.iter(), cpu_net.biases.iter()).for_each(|(a, b)| {
      assert!(matrix_are_equal_gpu_cpu(&a, &b, 5));
    });

    // Check neuron outputs
    izip!(gpu_outputs, cpu_outputs).for_each(|(a, b)| {
      assert!(matrix_are_equal_gpu_cpu(&a, &b, 5));
    });
  }

  fn approx_equal(a: f32, b: f32, precision: usize) -> bool {
    let tolerance = f32::powf(10.0, -1.0 * precision as f32);
    return (a - b).abs() < tolerance;
  }
}
