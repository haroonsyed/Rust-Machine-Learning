use crate::py_util::py_print;
use itertools::izip;
use pyo3::prelude::*;

pub struct AdaBoostNode {
  left_child: Option<Box<AdaBoostNode>>,
  right_child: Option<Box<AdaBoostNode>>,
  feature_col: usize,
  feature_val: f64,
  weight: f64,
  prediction: f64,
}

impl AdaBoostNode {
  fn print(&self, level: usize) {
    let is_leaf = self.left_child.is_none();
    let indent = "  ".repeat(level);

    if is_leaf {
      py_print(&format!("{} {} {}", indent, self.weight, self.prediction));
    } else {
      py_print(&format!(
        "{} {} {}",
        indent, self.feature_col, self.feature_val
      ));
      self.left_child.as_ref().unwrap().print(level + 1);
      self.right_child.as_ref().unwrap().print(level + 1);
    }
  }
}

#[pyclass]
pub struct AdaBoost {
  forest: Vec<AdaBoostNode>,
}

#[pymethods]
impl AdaBoost {
  #[new]
  fn new(features_train: Vec<Vec<f64>>, labels: Vec<f64>) -> Self {
    let mut combined_data: Vec<(Vec<f64>, f64)> = izip!(features_train, labels).collect();
    let mut forest = Vec::new();

    return AdaBoost { forest };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      // Travel down tree for each node and get the correct classificiation
    }

    return Ok(labels);
  }

  fn print(&self) {
    for root in self.forest.iter() {
      root.print(0);
    }
  }
}
