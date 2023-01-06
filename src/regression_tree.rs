use std::collections::{HashMap, HashSet};

use itertools::{izip, Itertools};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::{basic_stats::mode_f64, py_util::py_print};

struct RegressionTreeNode {
  is_categorical: bool, // Categorical is true/false. Else assumed ordered, continuous and numeric
  left_child: Option<Box<DecisionTreeNode>>,
  right_child: Option<Box<DecisionTreeNode>>,
  feature_col: usize,
  feature_val: f64, // Unused unless is_categorical == false
  classification: f64,
}

impl DecisionTreeNode {
  // new, insert, any calculations needed
}

#[pyclass]
pub struct RegressionTree {
  root: RegressionTreeNode,
}

#[pymethods]
impl DecisionTree {
  #[new]
  fn new(features_train: Vec<Vec<f64>>, is_categorical: Vec<bool>, labels: Vec<f64>) -> Self {
    let mut root = RegressionTreeNode::new();

    return DecisionTree {};
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      // Travel down tree for each node and get the correct classificiation
      labels.push(0.0);
    }

    return Ok(labels);
  }

  fn print(&self) {}
}
