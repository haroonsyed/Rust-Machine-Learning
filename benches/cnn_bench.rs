use rust_machine_learning::{
  convolutional_neural_network::ConvolutionalNeuralNetworkRust,
  image_util::ImageBatchLoaderRust,
  packed_optimizers::{
    PackedAdamOptimizer, PackedMomentumOptimizer, PackedStochasticGradientDescentOptimizer,
  },
};

use criterion::{criterion_group, criterion_main, Criterion};
use tensor_lib::cuda_bindings::cuda_synchronize;

// Define the benchmark function
pub fn cnn_benchmark(criterion: &mut Criterion) {
  // Setup the CNN
  let input_height = 32;
  let input_width = 32;
  let mut cnn = ConvolutionalNeuralNetworkRust::new(10, input_height, input_width, 3);

  cnn.set_optimizer(Box::new(PackedStochasticGradientDescentOptimizer::new(
    1e-3,
  )));
  // cnn.set_optimizer(Box::new(PackedAdamOptimizer::new(1e-3, 0.9, 0.999)));

  cnn.add_convolutional_layer(3, 3, 32);
  cnn.add_convolutional_layer(3, 3, 64);
  cnn.add_max_pool_layer();
  cnn.add_fully_connected_layer();

  // Print working directory
  println!("Working directory: {:?}", std::env::current_dir());

  //   Now setup the image feeder
  let parent_folder = String::from("./data/cifar-10/");
  let mut batch_loader = ImageBatchLoaderRust::new(parent_folder, input_width, input_height);

  //   Now train the network
  for _ in 0..500 {
    let (observations, labels) = batch_loader.batch_sample_as_matrix(32, false);
    cnn.train(observations, labels);
  }

  criterion.bench_function("cnn_bench", |bench| {
    bench.iter(|| {
      // Just a dummy function, used mainly to make exe for nsight systems
      unsafe { cuda_synchronize() }
    });
  });
}

// Register the benchmark
criterion_group!(cnn_bench, cnn_benchmark);
criterion_main!(cnn_bench);
