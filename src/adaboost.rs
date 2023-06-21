use crate::{
  basic_stats::{get_min_purity, normalize},
  py_util::py_print,
};
use itertools::izip;
use pyo3::prelude::*;
use rand::Rng;

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
  fn classify(&self, datapoint: &Vec<f64>) -> f64 {
    let sample_val = datapoint[self.feature_col];
    return if sample_val <= self.feature_val {
      0.0
    } else {
      1.0
    };
  }
}

#[pyclass]
pub struct AdaBoost {
  forest: Vec<AdaBoostStump>,
}

#[pymethods]
impl AdaBoost {
  #[new]
  fn new(
    features_train: Vec<Vec<f64>>,
    is_categorical: Vec<bool>,
    input_labels: Vec<f64>,
    num_stumps: usize,
  ) -> Self {
    let mut forest = Vec::new();
    let mut samples = features_train;
    let mut labels = input_labels;

    for _i in 0..num_stumps {
      let mut sample_weights = vec![1.0 / samples.len() as f64; samples.len()];

      let (min_feature, _puritiy, avg) = get_min_purity(&is_categorical, &samples, &labels);

      let mut total_error =
        Self::calculate_total_error(&samples, &labels, &sample_weights, min_feature, avg);

      total_error = total_error.clamp(1e-7, 0.9999);

      // 0.5 * ln((1.0-error)/error)
      let say_weight = 0.5 * ((1.0 - total_error) / total_error).ln();

      if say_weight.is_nan() {
        py_print(&format!("INVALID WEIGHT FROM: {}", total_error))
      }

      // Now update the sample weights
      Self::update_sample_weights(
        &samples,
        &labels,
        &mut sample_weights,
        min_feature,
        avg,
        say_weight,
      );
      normalize(&mut sample_weights);

      // Now resample using sample_weights
      (samples, labels) = Self::resample(&samples, &labels, &sample_weights);

      let stump = AdaBoostStump {
        feature_col: min_feature,
        feature_val: avg, // Will be 0 if categorical
        say_weight: say_weight,
      };

      forest.push(stump);
    }

    return AdaBoost { forest };
  }

  fn classify(&self, features_test: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let mut labels = Vec::new();

    for datapoint in features_test {
      let mut weight_true = 0.0;
      let mut weight_false = 0.0;
      for stump in self.forest.iter() {
        let label = stump.classify(&datapoint);
        if label == 1.0 {
          weight_true += stump.say_weight;
        } else {
          weight_false += stump.say_weight;
        }
      }
      let classification = if weight_true > weight_false { 1.0 } else { 0.0 };
      labels.push(classification);
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
  fn correctly_classified(feature_val: f64, label: f64, feature_val_compare: f64) -> bool {
    return (label == 1.0 && feature_val > feature_val_compare)
      || (label == 0.0 && feature_val <= feature_val_compare);
  }
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
      let correct_label = Self::correctly_classified(feature_val, *label, feature_val_compare);
      if !correct_label {
        total_error += weight;
      }
    }
    return total_error;
  }
  fn update_sample_weights(
    samples: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
    sample_weights: &mut Vec<f64>,
    feature_col_compare: usize,
    feature_val_compare: f64,
    say_weight: f64,
  ) {
    for (datapoint, label, sample_weight) in izip!(samples, labels, sample_weights) {
      let feature_val = datapoint[feature_col_compare];
      let correct_label = Self::correctly_classified(feature_val, *label, feature_val_compare);
      let exp = if !correct_label {
        say_weight
      } else {
        -1.0 * say_weight
      };
      *sample_weight = *sample_weight * f64::exp(exp);
    }
  }

  fn resample(
    samples: &Vec<Vec<f64>>,
    labels: &Vec<f64>,
    sample_weights: &Vec<f64>,
  ) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut new_samples = Vec::new();
    let mut new_labels = Vec::new();

    let mut rng = rand::thread_rng();
    for _i in 0..samples.len() {
      let rnd = rng.gen_range(0.0..1.0);
      // Find sample corresponding to this weight
      let mut curr_weight = sample_weights[0];
      let mut idx = 0;
      while curr_weight < rnd {
        curr_weight += sample_weights[idx + 1];
        idx += 1;
      }
      new_samples.push(samples[idx].clone());
      new_labels.push(labels[idx]);
    }

    return (new_samples, new_labels);
  }
}
