use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::izip;
use Rust_Machine_Learning::regression_tree::RegressionTreeNode;

fn regression_benchmark(c: &mut Criterion) {
  let mut feature_data = vec![vec![
    0.0, 2.0, 4.0, 8.0, 11.0, 13.0, 15.0, 18.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 31.0,
    34.0, 36.0, 37.0,
  ]];

  let mut label_data = vec![
    0.0, 0.0, 0.0, 0.0, 5.0, 18.0, 100.0, 100.0, 100.0, 100.0, 60.0, 58.0, 56.0, 52.0, 48.0, 15.0,
    0.0, 0.0, 0.0,
  ];

  let mut combined: Vec<(Vec<f64>, f64)> = izip!(feature_data, label_data).collect();
  combined = black_box(combined);

  let datapoint_per_node = 1;

  c.bench_function("regression_tree", |b| {
    b.iter(|| RegressionTreeNode::build_tree(combined.clone(), datapoint_per_node))
  });
}

criterion_group!(benches, regression_benchmark);
criterion_main!(benches);
