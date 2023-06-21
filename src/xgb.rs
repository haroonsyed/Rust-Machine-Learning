use itertools::{izip, Itertools};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::basic_stats::get_residuals;

pub struct XbgRegressionTreeNode {
  left_child: Option<Box<XbgRegressionTreeNode>>,
  right_child: Option<Box<XbgRegressionTreeNode>>,
  prediction: f64,
  split_feature: usize,
  split_val: f64,
  prunned: bool,
}

impl XbgRegressionTreeNode {
  fn get_output_value(residuals: &[f64], lambda: f64) -> f64 {
    let mut output = 0.0;
    for val in residuals {
      output += val;
    }
    output /= residuals.len() as f64 + lambda;
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
    //py_print(&"Entering function");
    if residuals.len() == 1 || max_depth == 0.0 {
      return XbgRegressionTreeNode {
        left_child: None,
        right_child: None,
        split_feature: 0,
        split_val: 0.0,
        prediction: Self::get_output_value(residuals, lambda),
        prunned: true,
      };
    }

    //py_print(&"Getting similarity score");
    let root_similarity_score = Self::get_similarity_score(&residuals, lambda);

    // Determine grouping for children
    let mut highest_gain = f64::MIN;
    let mut split_feature = 0;
    let mut split_pos = 0; // Left inclusive

    let num_features = features_train[0].len();

    //py_print(&"Determining Split Position...");
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

    //py_print(&"Determined split position");

    // Now build subtrees based on the split position
    let child_data =
      izip!(features_train, residuals).sorted_by_key(|row| OrderedFloat(row.0[split_feature]));

    // Possibly expensive copy going on here...
    let child_features_train = child_data.clone().map(|row| row.0.clone()).collect_vec();
    let child_residuals = child_data.clone().map(|row| *row.1).collect_vec();

    //py_print(&split_pos);
    //py_print(&features_train);

    let mut left_child = None;
    let mut right_child = None;

    let mut children_performed_pruning = true;

    if split_pos != child_features_train.len() - 1 {
      left_child = Some(Box::new(Self::build_tree(
        &child_features_train[0..=split_pos],
        &child_residuals[0..=split_pos],
        lambda,
        gamma,
        max_depth - 1.0,
        sample_rate,
      )));

      right_child = Some(Box::new(Self::build_tree(
        &child_features_train[(split_pos + 1)..],
        &child_residuals[(split_pos + 1)..],
        lambda,
        gamma,
        max_depth - 1.0,
        sample_rate,
      )));
      children_performed_pruning =
        left_child.as_ref().unwrap().prunned || right_child.as_ref().unwrap().prunned;
    }

    //py_print(&"Split Data");

    // Now let's perform pruning as needed
    let should_prune = (highest_gain - gamma) < 0.0;
    if children_performed_pruning && should_prune {
      left_child = None;
      right_child = None;
    }

    //py_print(&"Prunned");
    //py_print(&child_features_train.len());
    //py_print(&child_features_train);
    //py_print(&split_feature);
    //py_print(&split_pos);

    return XbgRegressionTreeNode {
      left_child,
      right_child,
      prediction: Self::get_output_value(residuals, lambda),
      split_feature,
      split_val: child_features_train[split_pos][split_feature],
      prunned: children_performed_pruning && should_prune,
    };
  }

  fn classify_point(&self, datapoint: &Vec<f64>) -> f64 {
    // Travel down tree for each node and get the correct classificiation
    let comparison_value = datapoint[self.split_feature];

    if comparison_value < self.split_val {
      if self.left_child.is_none() {
        return self.prediction;
      }
      return self.left_child.as_ref().unwrap().classify_point(datapoint);
    } else {
      if self.right_child.is_none() {
        return self.prediction;
      }
      return self.right_child.as_ref().unwrap().classify_point(datapoint);
    }
  }
  fn classify(&self, features_test: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      labels.push(self.classify_point(&datapoint));
    }

    return labels;
  }
}

#[pyclass]
pub struct XGB {
  forest: Vec<XbgRegressionTreeNode>,
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
    learning_rate: f64,
  ) -> Self {
    let mut forest = Vec::new();

    // Init the residuals
    let mut predictions = vec![0.5; features_train.len()];
    let mut residuals = get_residuals(&input_labels, &predictions);

    // Build forest
    for _i in 1..num_trees {
      // Build tree
      let tree = XbgRegressionTreeNode::build_tree(
        &features_train,
        &residuals,
        lamba,
        gamma,
        max_depth,
        sample_rate,
      );

      //py_print(&"Built Tree");

      // Now update the predictions and residuals
      let residual_classifications = tree.classify(&features_train);
      for (prediction, residual_classification) in
        izip!(predictions.iter_mut(), residual_classifications)
      {
        *prediction += learning_rate * residual_classification;
      }

      //py_print(&"Re Predicted");

      // Recalculate the residuals
      residuals = get_residuals(&input_labels, &predictions);

      //py_print(&"Recalculated Residuals");

      forest.push(tree);
    }

    return XGB { forest };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>, learning_rate: f64) -> PyResult<Vec<f64>> {
    let mut labels = vec![0.5; features_test.len()];

    for tree in self.forest.iter() {
      let residuals = tree.classify(&features_test);
      for (label, residual) in labels.iter_mut().zip(residuals) {
        *label += learning_rate * residual;
      }
    }

    return Ok(labels);
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
