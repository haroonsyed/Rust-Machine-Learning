use std::collections::{HashMap, HashSet};

use itertools::{izip, Itertools};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::{
  basic_stats::{mean_2d, mode_f64},
  py_util::py_print,
};

struct RegressionTreeNode {
  is_categorical: bool, // Categorical is true/false. Else assumed ordered, continuous and numeric
  left_child: Option<Box<RegressionTreeNode>>,
  right_child: Option<Box<RegressionTreeNode>>,
  feature_col: usize,
  feature_val: f64,
}

impl RegressionTreeNode {
  // new, insert, any calculations needed
  const DATAPOINTS_PER_NODE: usize = 7;

  pub fn new(is_categorical: bool, feature_col: usize, feature_val: f64) -> Self {
    return RegressionTreeNode {
      is_categorical,
      left_child: None,
      right_child: None,
      feature_col,
      feature_val,
    };
  }

  pub fn build_tree(
    features_train: &Vec<(Vec<f64>, f64)>,
    is_categorical: &Vec<bool>,
  ) -> RegressionTreeNode {
    let num_features = features_train[0].0.len();

    let mut feature_ssr_avg = Vec::new();
    for feature_col in 0..num_features {
      feature_ssr_avg.push(Self::get_feature_ssr_avg(features_train, feature_col));
    }

    let lowest_ssr_data = feature_ssr_avg
      .iter()
      .min_by(|a, b| OrderedFloat(a.1).cmp(&OrderedFloat(b.1)))
      .unwrap();
    let lowest_ssr_col = lowest_ssr_data.0;
    let lowest_ssr = lowest_ssr_data.1;
    let lowest_ssr_avg = lowest_ssr_data.2;

    // Now split the data
    let (less, gre) = Self::split_data(features_train, lowest_ssr_col, lowest_ssr_avg);

    // Recurse only if data length is > DATAPOINTS_PER_NODE
    let mut left_child = None;
    if less.len() > Self::DATAPOINTS_PER_NODE {
      left_child = Some(Box::new(Self::build_tree(&less, is_categorical)));
    }
    let mut right_child = None;
    if gre.len() > Self::DATAPOINTS_PER_NODE {
      right_child = Some(Box::new(Self::build_tree(&gre, is_categorical)));
    }

    return RegressionTreeNode {
      is_categorical: is_categorical[lowest_ssr_col],
      left_child,
      right_child,
      feature_col: lowest_ssr_col,
      feature_val: lowest_ssr_avg,
    };
  }

  /// Return feature_col, ssr, avg
  fn get_feature_ssr_avg(
    features_train: &Vec<(Vec<f64>, f64)>,
    feature_col: usize,
  ) -> (usize, f64, f64) {
    let sorted = Self::sort_data_by_feature(features_train, feature_col);

    let mut min_residual_data = (feature_col, 0.0, 0.0);

    for i in 0..sorted.len() {
      let mut residual = 0.0;
      let split_point = if i == 0 {
        sorted[i].0[feature_col]
      } else {
        (sorted[i - 1].0[feature_col] + sorted[i].0[feature_col]) / 2.0
      };
      let (less, gre) = Self::split_data(&sorted, feature_col, split_point);
      let data_less = less.iter().map(|x| x.clone().0).collect();
      let data_gre = gre.iter().map(|x| x.clone().0).collect();
      let avg_less = mean_2d(&data_less)[feature_col];
      let avg_gre = mean_2d(&data_gre)[feature_col];

      for (datapoint, label) in less.iter() {
        residual += (datapoint[feature_col] - avg_less).powf(2.0);
      }
      for (datapoint, label) in gre.iter() {
        residual += (datapoint[feature_col] - avg_gre).powf(2.0);
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
  fn new(features_train: Vec<Vec<f64>>, is_categorical: Vec<bool>, labels: Vec<f64>) -> Self {
    let combined_data = izip!(features_train, labels).collect();
    let mut root = RegressionTreeNode::build_tree(&combined_data, &is_categorical);

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

  fn print(&self) {}
}
