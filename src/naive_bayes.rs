use std::collections::HashMap;

use itertools::izip;
use pyo3::prelude::*;

use crate::basic_stats::{self};

#[pyclass]
pub struct naive_bayes_model {
  class_stats: HashMap<i64, [Vec<f64>; 3]>, // Label -> Mean, Variance, Prior for each feature.
                                            // Prior is vec of size 1.
}

#[pymethods]
impl naive_bayes_model {
  #[new]
  fn new(features_train: Vec<Vec<f64>>, labels: Vec<i64>) -> Self {
    // First let's group the class data
    let mut grouped_class_data = HashMap::new();
    for (datapoint, label) in izip!(features_train.iter(), labels.iter()) {
      // Add the datapoint to the correct class
      grouped_class_data
        .entry(*label)
        .or_insert(Vec::new())
        .push(datapoint.clone());
    }

    // Now calculate mean, var, prior for each
    let mut class_stats = HashMap::new();
    for (label, class_data) in grouped_class_data {
      let means = basic_stats::mean_2d(&class_data);
      let variances = basic_stats::variance_2d(&class_data, false);

      // Calculate prior for this class as well
      let prior = class_data.len() as f64 / features_train.len() as f64;

      // Add the calculated data to the class_data
      class_stats.insert(label, [means, variances, vec![prior]]);
    }

    return naive_bayes_model { class_stats };
  }

  fn naive_bayes_gaussian(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<i64>> {
    let mut labels = Vec::new();
    labels = vec![0; features_test.len()];

    return Ok(labels);
  }
}
