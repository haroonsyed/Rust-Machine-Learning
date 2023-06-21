use std::collections::HashMap;

use itertools::izip;
use pyo3::prelude::*;

use crate::basic_stats::{self, gaussian_probability};

#[pyclass]
pub struct NaiveBayesModel {
  class_stats: HashMap<i64, [Vec<f64>; 3]>, // Label -> Mean, Variance, Prior for each feature.
                                            // Prior is vec of size 1.
}

#[pymethods]
impl NaiveBayesModel {
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

    return NaiveBayesModel { class_stats };
  }

  fn naive_bayes_gaussian(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<i64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      let mut highest_score = f64::MIN;
      let mut highest_score_label = 0;
      for (label, feature_stats) in self.class_stats.iter() {
        // grab the necessary trained stats
        let prior = feature_stats[2][0];
        let means = &feature_stats[0];
        let variances = &feature_stats[1];

        // Calculate the scores for being each label
        // ln is to prevent numbers from becoming out of range of float
        let mut score = prior.ln();
        let mut curr_feature_index = 0;
        for (mean, variance) in izip!(means, variances) {
          let curr_feature_val = datapoint[curr_feature_index];

          let std_deviation = variance.sqrt();

          score += gaussian_probability(curr_feature_val, *mean, std_deviation).ln();

          curr_feature_index += 1;
        }

        // Determine if this score is the highest
        if score > highest_score {
          highest_score = score;
          highest_score_label = *label;
        }
      }

      // Label this point as the one with highest score
      labels.push(highest_score_label);
    }

    return Ok(labels);
  }
}
