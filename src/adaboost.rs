use crate::{basic_stats::get_min_purity, py_util::py_print};
use itertools::izip;
use pyo3::prelude::*;

pub struct AdaBoostStump {
  feature_col: usize,
  feature_val: f64,
  say_weight: f64,
}

impl AdaBoostStump {
  fn print(&self, level: usize) {
    let indent = "  ".repeat(level);

    py_print(&format!(
      "{} {} {} {}",
      indent, self.say_weight, self.feature_col, self.feature_val
    ));
  }
}

#[pyclass]
pub struct AdaBoost {
  forest: Vec<AdaBoostStump>,
}

#[pymethods]
impl AdaBoost {
  #[new]
  fn new(features_train: Vec<Vec<f64>>, is_categorical: Vec<bool>, labels: Vec<f64>) -> Self {
    let mut forest = Vec::new();
    let sample_weights = vec![1.0 / features_train.len() as f64; features_train.len()];

    let (min_feature, puritiy, avg) = get_min_purity(&is_categorical, &features_train, &labels);

    let total_error =
      Self::calculate_total_error(&features_train, &labels, &sample_weights, min_feature, avg);

    // 0.5 * ln((1.0-error)/error)
    let say_weight = 0.5 * ((1.0 - total_error) / total_error).ln();

    // Now update the sample weights

    let stump = AdaBoostStump {
      feature_col: min_feature,
      feature_val: avg, // Will be 0 if categorical
      say_weight: say_weight,
    };

    return AdaBoost { forest };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      // Travel down tree for each node and get the correct classificiation
    }

    return Ok(labels);
  }

  fn print(&self) {
    for root in self.forest.iter() {
      root.print(0);
    }
  }
}

impl AdaBoost {
  fn calculate_total_error(
    samples: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
    sample_weights: &Vec<f64>,
    feature_col_compare: usize,
    feature_val_compare: f64,
  ) -> f64 {
    let mut total_error = 0.0;
    for (datapoint, label, weight) in izip!(samples, labels, sample_weights) {
      // Less means false
      // Greater means true for label
      let feature_val = datapoint[feature_col_compare];
      let incorrect_label = (*label == 1.0 && feature_val <= feature_val_compare)
        || (*label == 0.0 && feature_val > feature_val_compare);
      if incorrect_label {
        total_error += weight;
      }
    }
    return total_error;
  }
}
