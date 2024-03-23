use rust_machine_learning::{
  basic_neural_network::BasicNeuralNetworkRust, optimizers::AdamOptimizer,
};

use criterion::{criterion_group, criterion_main, Criterion};
use tensor_lib::cuda_bindings::cuda_synchronize;

// Define the benchmark function
pub fn nn_benchmark(criterion: &mut Criterion) {
  // Setup the NN
  let nn_hidden_layers = vec![32, 32];
  let mut nn = BasicNeuralNetworkRust::new(nn_hidden_layers, 784, 10);

  nn.set_optimizer(Box::new(AdamOptimizer::new(1e-2, 0.9, 0.999)));

  // Print working directory
  println!("Working directory: {:?}", std::env::current_dir());

  //   Create some fake data for the benchmark
  let observations = (0..1000)
    .map(|_| {
      (0..784)
        .map(|_| rand::random::<f32>())
        .collect::<Vec<f32>>()
    })
    .collect::<Vec<Vec<f32>>>();

  let labels = (0..1000).map(|label| (label % 10) as f32).collect();

  //   Now train the network
  nn.train(observations, labels, 500, 32);

  criterion.bench_function("nn_bench", |bench| {
    bench.iter(|| {
      // Just a dummy function, used mainly to make exe for nsight systems
      unsafe { cuda_synchronize() }
    });
  });
}

// Register the benchmark
criterion_group!(nn_bench, nn_benchmark);
criterion_main!(nn_bench);
