use itertools::izip;
use pyo3::prelude::*;
use rand::Rng;

pub struct GradientBoostNode {}

impl GradientBoostNode {
  fn print(&self, level: usize) {}
  fn classify(&self, datapoint: &Vec<f64>) -> f64 {
    return 0.0;
  }
}

#[pyclass]
pub struct GradientBoost {
  forest: Vec<GradientBoostNode>,
}

#[pymethods]
impl GradientBoost {
  #[new]
  fn new(
    features_train: Vec<Vec<f64>>,
    is_categorical: Vec<bool>,
    input_labels: Vec<f64>,
    tree_depth: usize,
  ) -> Self {
    let mut forest = Vec::new();

    return GradientBoost { forest };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {}

    return Ok(labels);
  }

  fn print(&self) {
    for root in self.forest.iter() {
      root.print(0);
    }
  }
}

impl GradientBoost {}
