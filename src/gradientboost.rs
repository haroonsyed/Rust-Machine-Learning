use itertools::izip;
use pyo3::prelude::*;

use crate::{
  basic_stats::{get_residuals, mean},
  regression_tree::RegressionTreeRust,
};

#[pyclass]
pub struct GradientBoost {
  init_prediction: f64,
  learning_rate: f64,
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
    is_categorical: bool,
  ) -> Self {
    return GradientBoost::buildForestNumeric(
      features_train,
      input_labels,
      tree_depth,
      num_trees,
      learning_rate,
    );
    // if is_categorical {

    // }
    // else {

    // }
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = vec![self.init_prediction; features_test.len()];

    for tree in self.forest.iter() {
      let residuals = tree.classify(&features_test);
      for (label, residual) in labels.iter_mut().zip(residuals) {
        *label += self.learning_rate * residual;
      }
    }

    return Ok(labels);
  }

  fn print(&self) {
    for root in self.forest.iter() {
      root.print();
    }
  }
}

impl GradientBoost {
  fn buildForestNumeric(
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

    for _i in 1..num_trees {
      // Build a tree to predict the residuals
      let max_data_points_per_leaf = 20;
      let tree = RegressionTreeRust::new(
        &features_train,
        &residuals,
        max_data_points_per_leaf,
        tree_depth,
      );

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
      learning_rate,
      forest,
    };
  }
}
