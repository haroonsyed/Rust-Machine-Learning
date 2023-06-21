use std::collections::HashSet;

use itertools::izip;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::{
  basic_stats::{get_purities, mode_f64},
  py_util::py_print,
};

struct DecisionTreeNode {
  is_categorical: bool, // Categorical is true/false. Else assumed ordered, continuous and numeric
  left_child: Option<Box<DecisionTreeNode>>,
  right_child: Option<Box<DecisionTreeNode>>,
  feature_col: usize,
  feature_val: f64, // Unused unless is_categorical == false
  classification: f64,
}

impl DecisionTreeNode {
  pub fn new() -> Self {
    return DecisionTreeNode {
      is_categorical: false,
      feature_col: 0,
      feature_val: 0.0,
      right_child: None,
      left_child: None,
      classification: f64::NAN,
    };
  }

  fn traverse_print(&self, level: usize) {
    let str_non_leaf = format!(
      "{} {} {}",
      self.is_categorical, self.feature_col, self.feature_val,
    );
    let str_leaf = format!("{}", self.classification);
    let indent = "  ".repeat(level);

    let is_leaf = self.right_child.is_none();

    if is_leaf {
      let combined = format!("{}{}", indent, str_leaf);
      py_print(&combined);
    } else {
      let combined = format!("{}{}", indent, str_non_leaf);
      py_print(&combined);
      self.left_child.as_ref().unwrap().traverse_print(level + 1);
      self.right_child.as_ref().unwrap().traverse_print(level + 1);
    }
  }

  fn is_pure(labels: &Vec<f64>) -> bool {
    let mut true_count = 0;
    let mut false_count = 0;
    for label in labels {
      if *label == 1.0 {
        true_count += 1;
      } else {
        false_count += 1;
      }
    }
    return true_count == 0 || false_count == 0;
  }

  // Each node handles splitting of data to left/right child
  fn insert(
    &mut self,
    features_train: &Vec<Vec<f64>>,
    is_categorical: &Vec<bool>,
    labels: &Vec<f64>,
    mut features_to_process: HashSet<usize>,
    max_depth: usize,
  ) {
    let is_pure = Self::is_pure(labels);
    if is_pure {
      self.classification = mode_f64(labels);
    } else {
      let purities = get_purities(&features_to_process, is_categorical, features_train, labels);

      // Now based on the lowest purity determine show to label this node.
      // Then create the right and left if is not impure (later we can use a threshold)
      let lowest_purity_data = purities
        .iter()
        .min_by(|a, b| OrderedFloat(a.1 .0).cmp(&OrderedFloat(b.1 .0)))
        .unwrap();

      let lowest_purity_col = *lowest_purity_data.0;
      let lowest_purity_categorical = is_categorical[lowest_purity_col];
      let lowest_purity_feature_value = lowest_purity_data.1 .1;

      let (split_left, split_right) = if lowest_purity_categorical {
        Self::split_data_categorical(features_train, labels, lowest_purity_col)
      } else {
        Self::split_data_numeric(
          features_train,
          labels,
          lowest_purity_col,
          lowest_purity_feature_value,
        )
      };

      // Write data for non-leaf node
      self.is_categorical = lowest_purity_categorical;
      self.left_child = Some(Box::new(DecisionTreeNode::new()));
      self.right_child = Some(Box::new(DecisionTreeNode::new()));
      self.feature_col = lowest_purity_col;
      self.feature_val = lowest_purity_feature_value;
      self.classification = f64::NAN;

      // Remove this feature from the ones to process
      features_to_process.remove(&lowest_purity_col);

      // Recurse
      if max_depth == 1 {
        return;
      }

      self.left_child.as_mut().unwrap().insert(
        &split_left.0,
        is_categorical,
        &split_left.1,
        features_to_process.clone(),
        max_depth - 1,
      );
      self.right_child.as_mut().unwrap().insert(
        &split_right.0,
        is_categorical,
        &split_right.1,
        features_to_process.clone(),
        max_depth - 1,
      );
    }
  }

  // ret.0 is true (data, labels, is_categorical). ret.1 is false (data, labels, is_categorical).
  fn split_data_categorical(
    features_train: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
    feature: usize,
  ) -> ((Vec<Vec<f64>>, Vec<f64>), (Vec<Vec<f64>>, Vec<f64>)) {
    let mut feature_true = (Vec::new(), Vec::new());
    let mut feature_false = (Vec::new(), Vec::new());
    for row in izip!(features_train, labels) {
      if row.0[feature] == 1.0 {
        feature_true.0.push(row.0.to_vec());
        feature_true.1.push(*row.1);
      } else {
        feature_false.0.push(row.0.to_vec());
        feature_false.1.push(*row.1);
      }
    }
    return (feature_true, feature_false);
  }

  // ret.0 is less than. ret.1 is GRE.
  fn split_data_numeric(
    features_train: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
    feature: usize,
    feature_val: f64,
  ) -> ((Vec<Vec<f64>>, Vec<f64>), (Vec<Vec<f64>>, Vec<f64>)) {
    let mut feature_less = (Vec::new(), Vec::new());
    let mut feature_gre = (Vec::new(), Vec::new());
    for row in izip!(features_train, labels) {
      if row.0[feature] < feature_val {
        feature_less.0.push(row.0.to_vec());
        feature_less.1.push(*row.1);
      } else {
        feature_gre.0.push(row.0.to_vec());
        feature_gre.1.push(*row.1);
      }
    }
    return (feature_less, feature_gre);
  }

  fn classify(&self, datapoint: &Vec<f64>) -> f64 {
    let is_leaf = self.right_child.is_none();
    if is_leaf {
      return self.classification;
    }

    let comparison_value = datapoint[self.feature_col];

    if self.is_categorical {
      return if comparison_value == 1.0 {
        self.left_child.as_ref().unwrap().classify(datapoint)
      } else {
        self.right_child.as_ref().unwrap().classify(datapoint)
      };
    } else {
      return if comparison_value < self.feature_val {
        self.left_child.as_ref().unwrap().classify(datapoint)
      } else {
        self.right_child.as_ref().unwrap().classify(datapoint)
      };
    }
  }
}

#[pyclass]
pub struct DecisionTree {
  root: DecisionTreeNode,
}

#[pymethods]
impl DecisionTree {
  #[new]
  fn new(features_train: Vec<Vec<f64>>, is_categorical: Vec<bool>, labels: Vec<f64>) -> Self {
    let mut root = DecisionTreeNode::new();

    let num_features = features_train[0].len();
    let mut features_to_process = HashSet::new();
    for i in 0..num_features {
      features_to_process.insert(i);
    }

    root.insert(
      &features_train,
      &is_categorical,
      &labels,
      features_to_process,
      usize::MAX,
    );

    return DecisionTree { root };
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
    self.root.traverse_print(0);
  }
}

pub struct DecisionTreeRust {
  root: DecisionTreeNode,
}

impl DecisionTreeRust {
  pub fn new(
    features_train: &Vec<Vec<f64>>,
    is_categorical: &Vec<bool>,
    labels: &Vec<f64>,
    max_depth: usize,
  ) -> Self {
    let mut root = DecisionTreeNode::new();

    let num_features = features_train[0].len();
    let mut features_to_process = HashSet::new();
    for i in 0..num_features {
      features_to_process.insert(i);
    }

    root.insert(
      &features_train,
      &is_categorical,
      &labels,
      features_to_process,
      max_depth,
    );

    return DecisionTreeRust { root };
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
    self.root.traverse_print(0);
  }
}
