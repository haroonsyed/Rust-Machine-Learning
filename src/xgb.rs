use itertools::{izip, Itertools};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::{
  basic_stats::{get_residuals, mean},
  py_util::py_print,
};

pub struct XBG_RegressionTreeNode {
  left_child: Option<Box<XBG_RegressionTreeNode>>,
  right_child: Option<Box<XBG_RegressionTreeNode>>,
  prediction: f64,
  prunned: bool,
}

impl XBG_RegressionTreeNode {
  fn get_output_value(residuals: &[f64], lambda: f64) -> f64 {
    let mut output = 0.0;
    for val in residuals {
      output += val;
    }
    output /= (residuals.len() as f64 + lambda);
    return output;
  }
  fn get_similarity_score(residuals: &[f64], lambda: f64) -> f64 {
    let mut sum_residuals_squared = 0.0;
    for i in 0..residuals.len() {
      sum_residuals_squared += residuals[i];
    }
    sum_residuals_squared *= sum_residuals_squared;

    return sum_residuals_squared / (residuals.len() as f64 + lambda);
  }
  fn build_tree(
    features_train: &[Vec<f64>],
    residuals: &[f64],
    lambda: f64,
    gamma: f64,
    max_depth: f64,
    sample_rate: f64,
  ) -> Self {
    // TODO: BASE CASE
    if (residuals.len() == 1 || max_depth == 0.0) {
      return XBG_RegressionTreeNode {
        left_child: None,
        right_child: None,
        prediction: Self::get_output_value(residuals, lambda),
        prunned: false,
      };
    }

    let root_similarity_score = Self::get_similarity_score(&residuals, lambda);

    // Determine grouping for children
    let mut highest_gain = 0.0;
    let mut split_feature = 0;
    let mut split_pos = 0; // Left inclusive

    let num_features = features_train[0].len();

    for feature in 0..num_features {
      // Sort the relevant data in range by this feature
      let sorted =
        izip!(features_train, residuals).sorted_by_key(|row| OrderedFloat(row.0[feature]));

      // Use running sum as optimization
      let mut sum_residuals_left = 0.0;
      let mut sum_residuals_right = sorted
        .clone()
        .fold(0.0, |acc, (_, residual)| acc + residual);
      let mut residuals_processed = 0;

      // Now try all combinations of split points
      // TODO: Use sampling/binning as optimization
      for (_, residual) in sorted {
        sum_residuals_left += residual;
        sum_residuals_right -= residual;

        let curr_similarity_left =
          (sum_residuals_left * sum_residuals_left) / (residuals_processed as f64 + 1.0 + lambda);
        let curr_similarity_right = (sum_residuals_right * sum_residuals_right)
          / ((residuals.len() - residuals_processed) as f64 - 1.0 + lambda);

        let gain = curr_similarity_left + curr_similarity_right - root_similarity_score;

        // Determine if this is a better split point
        if gain > highest_gain {
          highest_gain = gain;
          split_feature = feature;
          split_pos = residuals_processed;
        }

        residuals_processed += 1;
      }
    }

    // Now build subtrees based on the split position
    let child_data =
      izip!(features_train, residuals).sorted_by_key(|row| OrderedFloat(row.0[split_feature]));

    // Possibly expensive copy going on here...
    let child_features_train = child_data.clone().map(|row| row.0.clone()).collect_vec();
    let child_residuals = child_data.clone().map(|row| *row.1).collect_vec();

    let mut left_child = Some(Box::new(Self::build_tree(
      &child_features_train[0..=split_pos],
      &child_residuals[0..=split_pos],
      lambda,
      gamma,
      max_depth - 1.0,
      sample_rate,
    )));

    let mut right_child = Some(Box::new(Self::build_tree(
      &child_features_train[(split_pos + 1)..],
      &child_residuals[(split_pos + 1)..],
      lambda,
      gamma,
      max_depth - 1.0,
      sample_rate,
    )));

    // Now let's perform pruning as needed
    let children_performed_pruning =
      left_child.as_ref().unwrap().prunned || right_child.as_ref().unwrap().prunned;
    let should_prune = !children_performed_pruning && (highest_gain - gamma) < 0.0;
    if !children_performed_pruning && should_prune {
      left_child = None;
      right_child = None;
    }

    return XBG_RegressionTreeNode {
      left_child,
      right_child,
      prediction: Self::get_output_value(residuals, lambda),
      prunned: children_performed_pruning || should_prune,
    };
  }
}

#[pyclass]
pub struct XGB {
  forest: Vec<XBG_RegressionTreeNode>,
}

#[pymethods]
impl XGB {
  #[new]
  fn new(
    features_train: Vec<Vec<f64>>,
    input_labels: Vec<f64>,
    num_trees: usize,
    lamba: f64,
    gamma: f64,
    max_depth: f64,
    sample_rate: f64,
  ) -> Self {
    let mut forest = Vec::new();

    // Init the residuals
    let mut predictions = vec![0.5; features_train.len()];
    let mut residuals = get_residuals(&input_labels, &predictions);

    // Build forest
    for _i in 1..num_trees {
      // Build tree
      let tree = XBG_RegressionTreeNode::build_tree(
        &features_train,
        &residuals,
        lamba,
        gamma,
        max_depth,
        sample_rate,
      );
      forest.push(tree);
    }

    return XGB { forest };
  }
  #[staticmethod]
  fn default_lamba() -> PyResult<f64> {
    return Ok(1.0);
  }
  #[staticmethod]
  fn default_gamma() -> PyResult<f64> {
    return Ok(0.0);
  }
  #[staticmethod]
  fn default_eta() -> PyResult<f64> {
    return Ok(0.3);
  }
  #[staticmethod]
  fn default_max_depth() -> PyResult<f64> {
    return Ok(6.0);
  }
  #[staticmethod]
  fn default_sample() -> PyResult<f64> {
    return Ok(1.0);
  }
}
