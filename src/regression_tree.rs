use std::collections::{HashMap, HashSet};

use itertools::{izip, Itertools};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::{
  basic_stats::{mean, mean_2d, mean_2d_col, mode_f64},
  py_util::py_print,
};

pub struct RegressionTreeNode {
  left_child: Option<Box<RegressionTreeNode>>,
  right_child: Option<Box<RegressionTreeNode>>,
  feature_col: usize,
  feature_val: f64,
  prediction: f64,
}

impl RegressionTreeNode {
  pub fn print(&self, level: usize) {
    let is_leaf = self.left_child.is_none();
    let indent = "  ".repeat(level);

    if is_leaf {
      py_print(&format!("{} {}", indent, self.prediction));
    } else {
      py_print(&format!(
        "{} {} {}",
        indent, self.feature_col, self.feature_val
      ));
      self.left_child.as_ref().unwrap().print(level + 1);
      self.right_child.as_ref().unwrap().print(level + 1);
    }
  }

  pub fn build_tree(
    mut features_train: Vec<&(&Vec<f64>, &f64)>,
    datapoints_per_node: usize,
    max_depth: usize,
  ) -> RegressionTreeNode {
    let num_features = features_train[0].0.len();

    let mut lowest_ssr_data = (0, f64::MAX, 0.0);
    for feature_col in 0..num_features {
      let ssr_data = Self::get_feature_ssr_avg(&mut features_train, feature_col);
      if ssr_data.1 < lowest_ssr_data.1 {
        lowest_ssr_data = ssr_data;
      }
    }
    let lowest_ssr_col = lowest_ssr_data.0;
    let lowest_ssr = lowest_ssr_data.1;
    let lowest_ssr_avg = lowest_ssr_data.2;

    // Now split the data
    let (mut less, mut gre) = Self::split_data(&features_train, lowest_ssr_col, lowest_ssr_avg);

    // Recurse only if data length is > DATAPOINTS_PER_NODE and no infinite recursion
    let mut left_child = None;
    let can_infinite_recurse = less.len() == 0 || gre.len() == 0;
    if less.len() > datapoints_per_node && !can_infinite_recurse && max_depth != 0 {
      left_child = Some(Box::new(Self::build_tree(
        less,
        datapoints_per_node,
        max_depth - 1,
      )));
    } else {
      left_child = Some(Box::new(RegressionTreeNode {
        left_child: None,
        right_child: None,
        feature_col: lowest_ssr_col,
        feature_val: lowest_ssr_avg,
        prediction: mean(&less.iter().map(|x| *x.1).collect()),
      }));
    }
    let mut right_child = None;
    if gre.len() > datapoints_per_node && !can_infinite_recurse && max_depth != 0 {
      right_child = Some(Box::new(Self::build_tree(
        gre,
        datapoints_per_node,
        max_depth - 1,
      )));
    } else {
      right_child = Some(Box::new(RegressionTreeNode {
        left_child: None,
        right_child: None,
        feature_col: lowest_ssr_col,
        feature_val: lowest_ssr_avg,
        prediction: mean(&gre.iter().map(|x| *x.1).collect()),
      }));
    }

    return RegressionTreeNode {
      left_child,
      right_child,
      feature_col: lowest_ssr_col,
      feature_val: lowest_ssr_avg,
      prediction: 0.0,
    };
  }

  /// Return feature_col, ssr, avg
  fn get_feature_ssr_avg(
    features_train: &mut Vec<&(&Vec<f64>, &f64)>,
    feature_col: usize,
  ) -> (usize, f64, f64) {
    Self::sort_data_by_feature(features_train, feature_col);
    let sorted = features_train;

    let mut min_residual_data = (feature_col, f64::MAX, 0.0);

    for i in 0..sorted.len() - 1 {
      let mut residual = 0.0;
      let split_point = (sorted[i].0[feature_col] + sorted[i + 1].0[feature_col]) / 2.0;
      let (less, gre) = Self::split_data(&sorted, feature_col, split_point);

      let labels_less: Vec<f64> = less.iter().map(|x| *x.1).collect();
      let labels_gre: Vec<f64> = gre.iter().map(|x| *x.1).collect();
      let avg_less = mean(&labels_less);
      let avg_gre = mean(&labels_gre);

      for (_datapoint, label) in less.iter() {
        residual += (*label - avg_less).powf(2.0);
      }
      for (_datapoint, label) in gre.iter() {
        residual += (*label - avg_gre).powf(2.0);
      }

      if residual < min_residual_data.1 {
        min_residual_data.1 = residual;
        min_residual_data.2 = split_point;
      }
    }

    return min_residual_data;
  }

  fn sort_data_by_feature(features_train: &mut Vec<&(&Vec<f64>, &f64)>, feature_col: usize) {
    features_train
      .sort_by(|a, b| OrderedFloat(a.0[feature_col]).cmp(&OrderedFloat(b.0[feature_col])));
  }

  fn split_data<'a, 'b, 'c>(
    features_train: &Vec<&'a (&'b Vec<f64>, &'c f64)>,
    feature_col: usize,
    feature_val: f64,
  ) -> (
    Vec<&'a (&'b Vec<f64>, &'c f64)>,
    Vec<&'a (&'b Vec<f64>, &'c f64)>,
  ) {
    let mut feature_less = Vec::new();
    let mut feature_gre = Vec::new();
    for row in features_train.iter() {
      if (row.0[feature_col] < feature_val) {
        feature_less.push(*row);
      } else {
        feature_gre.push(*row);
      }
    }
    return (feature_less, feature_gre);
  }

  fn classify(&self, datapoint: &Vec<f64>) -> f64 {
    let comparison_value = datapoint[self.feature_col];

    if comparison_value < self.feature_val {
      if (self.left_child.is_none()) {
        return self.prediction;
      }
      return self.left_child.as_ref().unwrap().classify(datapoint);
    } else {
      if (self.right_child.is_none()) {
        return self.prediction;
      }
      return self.right_child.as_ref().unwrap().classify(datapoint);
    }
  }
}

#[pyclass]
pub struct RegressionTree {
  root: RegressionTreeNode,
}

#[pymethods]
impl RegressionTree {
  #[new]
  fn new(features_train: Vec<Vec<f64>>, labels: Vec<f64>, datapoints_per_node: usize) -> Self {
    let combined_data: Vec<(&Vec<f64>, &f64)> =
      izip!(features_train.iter(), labels.iter()).collect();

    let mut combined_data_ref = combined_data.iter().collect();
    let mut root =
      RegressionTreeNode::build_tree(combined_data_ref, datapoints_per_node, usize::MAX);

    return RegressionTree { root };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      // Travel down tree for each node and get the correct classificiation
      labels.push(self.root.classify(&datapoint));
    }

    return Ok(labels);
  }

  fn print(&self) {
    self.root.print(0);
  }
}

pub struct RegressionTreeRust {
  root: RegressionTreeNode,
}

impl RegressionTreeRust {
  pub fn new(
    features_train: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
    datapoints_per_node: usize,
    max_depth: usize,
  ) -> Self {
    let mut combined_data = Vec::new();
    for (datapoint, label) in izip!(features_train, labels) {
      let pair = (datapoint, label);
      combined_data.push(pair);
    }

    let mut combined_data_ref = combined_data.iter().collect();

    let mut root =
      RegressionTreeNode::build_tree(combined_data_ref, datapoints_per_node, max_depth);

    return RegressionTreeRust { root };
  }

  pub fn classify(&self, features_test: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      // Travel down tree for each node and get the correct classificiation
      labels.push(self.root.classify(&datapoint));
    }

    return labels;
  }

  pub fn print(&self) {
    self.root.print(0);
  }
}
