use itertools::izip;
use pyo3::prelude::*;
use rand::Rng;

use crate::{
  basic_stats::{get_min_purity, get_residuals, mean, mean_2d_col},
  regression_tree::RegressionTreeRust,
};

#[pyclass]
pub struct GradientBoost {
  init_prediction: f64,
  forest: Vec<RegressionTreeRust>,
}

#[pymethods]
impl GradientBoost {
  #[new]
  fn new(
    features_train: Vec<Vec<f64>>,
    input_labels: Vec<f64>,
    tree_depth: usize,
    num_trees: usize,
    learning_rate: f64,
  ) -> Self {
    let mut forest = Vec::new();

    let num_features = features_train[0].len();

    // Initialize the residuals
    let init_prediction = mean(&input_labels);
    let mut predictions = vec![init_prediction; features_train.len()];
    let mut residuals = get_residuals(&input_labels, &predictions);

    for i in 1..num_trees {
      // Build a tree to predict the residuals
      let num_leafs = usize::pow(2, tree_depth as u32 - 1);
      let max_data_points_per_leaf = usize::max(features_train.len() / num_leafs, num_features);
      let tree = RegressionTreeRust::new(&features_train, &residuals, max_data_points_per_leaf);

      // Now update the predictions
      let residual_classifications = tree.classify(&features_train);
      for (prediction, residual_classification) in
        izip!(predictions.iter_mut(), residual_classifications)
      {
        *prediction += learning_rate * residual_classification;
      }

      // Recalculate the residuals
      for (observed, prediction, residual) in izip!(
        input_labels.iter(),
        predictions.iter(),
        residuals.iter_mut()
      ) {
        *residual = observed - prediction;
      }

      forest.push(tree);
    }

    return GradientBoost {
      init_prediction,
      forest,
    };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {}

    return Ok(labels);
  }

  fn print(&self) {
    for root in self.forest.iter() {
      root.print();
    }
  }
}

impl GradientBoost {}
