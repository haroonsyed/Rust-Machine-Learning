use rand::prelude::Distribution;
use statrs::distribution::Normal;

use criterion::{criterion_group, criterion_main, Criterion};
use Rust_Machine_Learning::matrix_lib::Matrix;

// Define the benchmark function
pub fn matrix_multiply_benchmark(criterion: &mut Criterion) {
  // Random numbers for generation
  let mut rng = rand::thread_rng();
  let range = Normal::new(0.0, 0.68).unwrap();

  let mat_dim = 200;

  let matrix1 = Matrix {
    data: (0..mat_dim * mat_dim)
      .map(|_| range.sample(&mut rng))
      .collect(),
    rows: mat_dim,
    columns: mat_dim,
  };
  let matrix2 = Matrix {
    data: (0..mat_dim * mat_dim)
      .map(|_| range.sample(&mut rng))
      .collect(),
    rows: mat_dim,
    columns: mat_dim,
  };
  criterion.bench_function("matrix_multiply", |bench| {
    bench.iter(|| {
      // Call the matrix_multiply function
      let _ = matrix1.matrix_multiply(&matrix2);
    });
  });
}

// Register the benchmark
criterion_group!(matrix_bench, matrix_multiply_benchmark);
criterion_main!(matrix_bench);
