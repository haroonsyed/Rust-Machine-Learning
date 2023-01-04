use std::collections::{HashMap, HashSet};

use itertools::izip;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::py_util::py_print;

struct DecisionTreeNode {
  is_categorical: bool, // Categorical is true/false. Else assumed ordered, continuous and numeric
  right_child: Option<Box<DecisionTreeNode>>,
  left_child: Option<Box<DecisionTreeNode>>,
  feature_col: usize,
  feature_val: f64, // Unused unless is_categorical == false
  classification: i64,
}

impl DecisionTreeNode {
  // Some sort of insertion function that takes data, and goes till only pure leaves
  // Each node handles splitting of data to left/right child
  pub fn insert(
    &mut self,
    features_train: &Vec<Vec<f64>>,
    is_categorical: &Vec<bool>,
    labels: &Vec<f64>,
    features_to_process: &HashSet<usize>,
  ) {
    let purities = Self::get_purities(
      &features_to_process,
      &is_categorical,
      &features_train,
      &labels,
    );

    // Now based on the lowest purity determine how to label this node.
    // Then create the right and left if is not impure (later we can use a threshold)
    let lowest_purity_data = purities
      .iter()
      .min_by(|a, b| OrderedFloat(a.1 .0).cmp(&OrderedFloat(b.1 .0)))
      .unwrap();

    let lowest_purity_col = *lowest_purity_data.0;
    let lowest_purity_categorical = is_categorical[lowest_purity_col];
    let lowest_purity_value = lowest_purity_data.1 .1;

    if (lowest_purity_categorical) {
      // Split data based on this category.
      // Determine if pure. Conditionally recurse.
    } else {
      // Split data based on the value for this feature
      // Determine if pure. Conditionally recurse.
    }

    py_print(&purities);
  }

  // ret.0 is true. ret.1 is false.
  fn split_data_categorical(
    features_train: &Vec<Vec<f64>>,
    feature: usize,
  ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut cat_true = Vec::new();
    let mut cat_false = Vec::new();

    return (cat_true, cat_false);
  }

  // ret.0 is true. ret.1 is false.
  fn split_data_numeric(
    features_train: &Vec<Vec<f64>>,
    feature: usize,
    feature_val: f64,
  ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut cat_true = Vec::new();
    let mut cat_false = Vec::new();

    return (cat_true, cat_false);
  }

  fn get_purities(
    features_to_process: &HashSet<usize>,
    is_categorical: &Vec<bool>,
    data: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
  ) -> HashMap<usize, (f64, f64)> {
    // returns usize -> (purity, average)
    let mut purities = HashMap::new();

    for col in izip!(features_to_process) {
      let is_col_categorical = is_categorical[*col];
      purities.insert(
        *col,
        Self::get_purity(*col, is_col_categorical, data, labels),
      );
    }

    return purities;
  }

  fn get_purity(
    feature: usize,
    is_categorical: bool,
    data: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
  ) -> (f64, f64) {
    let mut purity = 0.0;
    let mut average = 0.0;

    if is_categorical {
      let mut label_true_feature_true = 0;
      let mut label_false_feature_true = 0;
      let mut label_true_feature_false = 0;
      let mut label_false_feature_false = 0;
      for (datapoint, label) in izip!(data, labels) {
        let label_true = *label == 1.0;
        let feature_true = datapoint[feature] == 1.0;
        match (label_true, feature_true) {
          (true, true) => label_true_feature_true += 1,
          (false, true) => label_false_feature_true += 1,
          (true, false) => label_true_feature_false += 1,
          (false, false) => label_false_feature_false += 1,
        }
      }

      let mut num_feature_true = label_true_feature_true + label_false_feature_true;
      let mut num_feature_false = label_true_feature_false + label_false_feature_false;
      // Avoid divison by zero by adding 1
      if (num_feature_true == 0) {
        num_feature_true += 1;
      }
      if (num_feature_false == 0) {
        num_feature_false += 1;
      }

      let purity_feature_true = 1.0
        - (label_true_feature_true as f64 / num_feature_true as f64).powf(2.0)
        - (label_false_feature_true as f64 / num_feature_true as f64).powf(2.0);
      let purity_feature_false = 1.0
        - (label_true_feature_false as f64 / num_feature_false as f64).powf(2.0)
        - (label_false_feature_false as f64 / num_feature_false as f64).powf(2.0);

      // Return weighted average of the two purities
      let num_datapoints = num_feature_true + num_feature_false;
      purity += purity_feature_true * (num_feature_true as f64 / num_datapoints as f64);
      purity += purity_feature_false * (num_feature_false as f64 / num_datapoints as f64);
    } else {
      // Assumes the data is continuous and ordered
      let mut combined: Vec<(&Vec<f64>, &f64)> = izip!(data.iter(), labels.iter()).collect();
      combined.sort_by(|a, b| OrderedFloat(a.0[feature]).cmp(&OrderedFloat(b.0[feature])));

      // From the perspective of data item at 0, everything is greater to start
      let mut label_true_feature_less = 0;
      let mut label_false_feature_less = 0;
      let mut label_true_feature_greater = 0;
      let mut label_false_feature_greater = 0;
      for (row, label) in combined.iter() {
        if **label == 1.0 {
          label_true_feature_greater += 1;
        } else {
          label_false_feature_greater += 1;
        }
      }

      let mut row_impurities = Vec::new();
      for i in 1..combined.len() {
        // First calculate the average
        let curr_average = (combined[i - 1].0[feature] + combined[i].0[feature]) / 2.0;

        // Now update the label counts
        let prev_label = *combined[i - 1].1 == 1.0;
        if prev_label {
          label_true_feature_less += 1;
          label_true_feature_greater -= 1;
        } else {
          label_false_feature_less += 1;
          label_false_feature_greater -= 1;
        }

        let mut num_feature_true = label_true_feature_less + label_false_feature_less;
        let mut num_feature_false = label_true_feature_greater + label_false_feature_greater;
        // Avoid divison by zero by adding 1
        if (num_feature_true == 0) {
          num_feature_true += 1;
        }
        if (num_feature_false == 0) {
          num_feature_false += 1;
        }

        let purity_feature_true = 1.0
          - (label_true_feature_less as f64 / num_feature_true as f64).powf(2.0)
          - (label_false_feature_less as f64 / num_feature_true as f64).powf(2.0);
        let purity_feature_false = 1.0
          - (label_true_feature_greater as f64 / num_feature_false as f64).powf(2.0)
          - (label_false_feature_greater as f64 / num_feature_false as f64).powf(2.0);

        // Return weighted average of the two purities
        let num_datapoints = num_feature_true + num_feature_false;
        let mut curr_purity = 0.0;
        curr_purity += purity_feature_true * (num_feature_true as f64 / num_datapoints as f64);
        curr_purity += purity_feature_false * (num_feature_false as f64 / num_datapoints as f64);
        row_impurities.push((curr_purity, curr_average));
      }

      row_impurities.sort_by(|a, b| OrderedFloat(a.0).cmp(&OrderedFloat(b.0)));
      purity = row_impurities[0].0;
      average = row_impurities[1].1;
    }

    return (purity, average);
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
    let mut root = DecisionTreeNode {
      is_categorical: false,
      feature_col: 0,
      feature_val: 0.0,
      right_child: None,
      left_child: None,
      classification: 0,
    };

    let num_features = features_train[0].len();
    let mut features_to_process = HashSet::new();
    for i in 0..num_features {
      features_to_process.insert(i);
    }

    root.insert(
      &features_train,
      &is_categorical,
      &labels,
      &features_to_process,
    );

    return DecisionTree { root };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<bool>> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      // Travel down tree for each node and get the correct classificiation
      labels.push(false);
    }

    return Ok(labels);
  }
}
