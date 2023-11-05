use itertools::Itertools;
use rand::prelude::Distribution;
use statrs::distribution::Normal;

use criterion::{criterion_group, criterion_main, Criterion};
use tensor_lib::{cuda_bindings::cuda_synchronize, Matrix};

// Define the benchmark function
pub fn matrix_multiply_benchmark(criterion: &mut Criterion) {
  // Random numbers for generation
  let mut rng = rand::thread_rng();
  let range = Normal::new(0.0, 0.68).unwrap();

  let mat_dim = 4096;

  let matrix1 = Matrix::new_2d(
    &(0..mat_dim)
      .map(|_| {
        (0..mat_dim)
          .map(|_| range.sample(&mut rng) as f32)
          .collect_vec()
      })
      .collect_vec(),
  );
  let matrix2 = Matrix::new_2d(
    &(0..mat_dim)
      .map(|_| {
        (0..mat_dim)
          .map(|_| range.sample(&mut rng) as f32)
          .collect_vec()
      })
      .collect_vec(),
  );

  // Visible to nsight systems
  // (0..50).for_each(|_| {
  //   matrix1.matrix_multiply(&matrix2);
  // });

  criterion.bench_function("matrix_multiply", |bench| {
    bench.iter(|| {
      // Call the matrix_multiply function
      let _ = matrix1.matrix_multiply(&matrix2);
      unsafe { cuda_synchronize() }
    });
  });
}

// Register the benchmark
criterion_group!(matrix_bench, matrix_multiply_benchmark);
criterion_main!(matrix_bench);
