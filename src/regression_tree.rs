use std::collections::{HashMap, HashSet};

use itertools::{izip, Itertools};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::{
  basic_stats::{mean, mean_2d, mean_2d_col, mode_f64},
  py_util::py_print,
};

struct RegressionTreeNode {
  left_child: Option<Box<RegressionTreeNode>>,
  right_child: Option<Box<RegressionTreeNode>>,
  feature_col: usize,
  feature_val: f64,
  prediction: f64,
}

impl RegressionTreeNode {
  // new, insert, any calculations needed
  const DATAPOINTS_PER_NODE: usize = 7;

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

  pub fn build_tree(features_train: &Vec<(Vec<f64>, f64)>) -> RegressionTreeNode {
    let num_features = features_train[0].0.len();

    let mut lowest_ssr_data = (0, f64::MAX, 0.0);
    for feature_col in 0..num_features {
      let ssr_data = Self::get_feature_ssr_avg(features_train, feature_col);
      if ssr_data.1 < lowest_ssr_data.1 {
        lowest_ssr_data = ssr_data;
      }
    }
    let lowest_ssr_col = lowest_ssr_data.0;
    let lowest_ssr = lowest_ssr_data.1;
    let lowest_ssr_avg = lowest_ssr_data.2;

    // Now split the data
    let (less, gre) = Self::split_data(features_train, lowest_ssr_col, lowest_ssr_avg);

    // Recurse only if data length is > DATAPOINTS_PER_NODE and no infinite recursion
    let mut left_child = None;
    let can_infinite_recurse = less.len() == 0 || gre.len() == 0;
    if less.len() > Self::DATAPOINTS_PER_NODE && !can_infinite_recurse {
      left_child = Some(Box::new(Self::build_tree(&less)));
    } else {
      left_child = Some(Box::new(RegressionTreeNode {
        left_child: None,
        right_child: None,
        feature_col: lowest_ssr_col,
        feature_val: lowest_ssr_avg,
        prediction: mean(&less.iter().map(|x| x.1).collect()),
      }));
    }
    let mut right_child = None;
    if gre.len() > Self::DATAPOINTS_PER_NODE && !can_infinite_recurse {
      right_child = Some(Box::new(Self::build_tree(&gre)));
    } else {
      right_child = Some(Box::new(RegressionTreeNode {
        left_child: None,
        right_child: None,
        feature_col: lowest_ssr_col,
        feature_val: lowest_ssr_avg,
        prediction: mean(&gre.iter().map(|x| x.1).collect()),
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
    features_train: &Vec<(Vec<f64>, f64)>,
    feature_col: usize,
  ) -> (usize, f64, f64) {
    let sorted = Self::sort_data_by_feature(features_train, feature_col);

    let mut min_residual_data = (feature_col, f64::MAX, 0.0);

    for i in 0..sorted.len() - 1 {
      let mut residual = 0.0;
      let split_point = (sorted[i].0[feature_col] + sorted[i + 1].0[feature_col]) / 2.0;
      let (less, gre) = Self::split_data(&sorted, feature_col, split_point);
      let labels_less: Vec<f64> = less.iter().map(|x| x.clone().1).collect();
      let labels_gre: Vec<f64> = gre.iter().map(|x| x.clone().1).collect();
      let avg_less = mean(&labels_less);

      let avg_gre = mean(&labels_gre);

      for (datapoint, label) in less.iter() {
        residual += (label - avg_less).powf(2.0);
      }
      for (datapoint, label) in gre.iter() {
        residual += (label - avg_gre).powf(2.0);
      }

      if residual < min_residual_data.1 {
        min_residual_data.1 = residual;
        min_residual_data.2 = split_point;
      }
    }

    return min_residual_data;
  }

  fn sort_data_by_feature(
    features_train: &Vec<(Vec<f64>, f64)>,
    feature_col: usize,
  ) -> (Vec<(Vec<f64>, f64)>) {
    let mut copy = features_train.clone();
    copy.sort_by(|a, b| OrderedFloat(a.0[feature_col]).cmp(&OrderedFloat(b.0[feature_col])));
    return copy;
  }

  fn split_data(
    features_train: &Vec<(Vec<f64>, f64)>,
    feature_col: usize,
    feature_val: f64,
  ) -> (Vec<(Vec<f64>, f64)>, Vec<(Vec<f64>, f64)>) {
    let mut feature_less = Vec::new();
    let mut feature_gre = Vec::new();
    for row in features_train {
      if (row.0[feature_col] < feature_val) {
        feature_less.push(row.clone());
      } else {
        feature_gre.push(row.clone());
      }
    }
    return (feature_less, feature_gre);
  }
}

#[pyclass]
pub struct RegressionTree {
  root: RegressionTreeNode,
}

#[pymethods]
impl RegressionTree {
  #[new]
  fn new(features_train: Vec<Vec<f64>>, labels: Vec<f64>) -> Self {
    let combined_data = izip!(features_train, labels).collect();
    let mut root = RegressionTreeNode::build_tree(&combined_data);

    return RegressionTree { root };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      // Travel down tree for each node and get the correct classificiation
      labels.push(0.0);
    }

    return Ok(labels);
  }

  fn print(&self) {
    self.root.print(0);
  }
}
