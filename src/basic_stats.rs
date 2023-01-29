use itertools::izip;
use num::{FromPrimitive, Zero};
use ordered_float::OrderedFloat;
use statrs::distribution::{Continuous, Normal};
use std::{
  collections::{HashMap, HashSet},
  fmt::Debug,
  hash::Hash,
  ops::{Add, AddAssign, Div, DivAssign, Sub, SubAssign},
};

use crate::py_util::py_print;

pub fn mean<
  T: Copy
    + Debug
    + FromPrimitive
    + Zero
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + DivAssign<T>,
>(
  numbers: &Vec<T>,
) -> T {
  let mut sum = T::zero();

  for val in numbers {
    sum += *val;
  }

  sum /= FromPrimitive::from_usize(numbers.len()).unwrap();

  return sum;
}

pub fn mean_2d<
  T: Copy
    + Debug
    + FromPrimitive
    + Zero
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + DivAssign<T>,
>(
  numbers: &Vec<Vec<T>>,
) -> Vec<T> {
  if numbers.len() == 0 {
    return Vec::new();
  }

  let num_features = numbers[0].len();
  let mut means = vec![T::zero(); num_features];

  for i in 0..num_features {
    means[i] = mean_2d_col(numbers, i);
  }

  return means;
}

pub fn mean_2d_col<
  T: Copy
    + Debug
    + FromPrimitive
    + Zero
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + DivAssign<T>,
>(
  numbers: &Vec<Vec<T>>,
  col: usize,
) -> T {
  if numbers.len() == 0 {
    return T::zero();
  }

  let mut mean = T::zero();

  for data_point in numbers {
    mean += data_point[col];
  }

  mean /= FromPrimitive::from_usize(numbers.len()).unwrap();

  return mean;
}

pub fn median<T: Copy + Debug + Ord + Clone>(numbers: &Vec<T>) -> T {
  let mut copy = numbers.clone();
  copy.sort();
  return copy[numbers.len() / 2];
}

pub fn mode<T: Copy + Debug + Hash + Eq>(numbers: &Vec<T>) -> T {
  let mut counts = HashMap::new();

  for val in numbers.iter() {
    if !counts.contains_key(val) {
      counts.insert(*val, 0);
    }
    *counts.get_mut(val).unwrap() += 1;
  }

  // Find the most common in the map
  return *counts
    .iter()
    .max_by(|a, b| a.1.cmp(&b.1))
    .map(|(k, _v)| k)
    .unwrap();
}

pub fn mode_f64(numbers: &Vec<f64>) -> f64 {
  let mut counts = HashMap::new();

  for val in numbers.iter() {
    let val_safe = OrderedFloat(*val);
    if !counts.contains_key(&val_safe) {
      counts.insert(val_safe, 0);
    }
    *counts.get_mut(&val_safe).unwrap() += 1;
  }

  // Find the most common in the map
  return counts
    .iter()
    .max_by(|a, b| a.1.cmp(&b.1))
    .map(|(k, _v)| k)
    .unwrap()
    .0;
}

pub fn variance<
  T: Copy
    + Debug
    + FromPrimitive
    + Zero
    + Sub<T, Output = T>
    + SubAssign<T>
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + Into<f64>
    + DivAssign<T>,
>(
  numbers: &Vec<T>,
  is_population: bool,
) -> f64 {
  let mut variance: f64 = 0.0;

  let mean = mean(&numbers);

  for val in numbers {
    variance += (*val - mean).into().powf(2.0);
  }

  variance /= match is_population {
    true => numbers.len() as f64,
    false => (numbers.len() - 1) as f64,
  };

  return variance;
}

pub fn variance_2d<
  T: Copy
    + Debug
    + FromPrimitive
    + Zero
    + Sub<T, Output = T>
    + SubAssign<T>
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + Into<f64>
    + DivAssign<T>,
>(
  numbers: &Vec<Vec<T>>,
  is_population: bool,
) -> Vec<f64> {
  let num_features = numbers[0].len();
  let mut variances = vec![0.0; num_features];

  let means = mean_2d(&numbers);

  for data_point in numbers.iter() {
    for (feature_index, val) in data_point.iter().enumerate() {
      variances[feature_index] += (*val - means[feature_index]).into().powf(2.0);
    }
  }

  variances.iter_mut().for_each(|x| {
    *x /= match is_population {
      true => numbers.len() as f64,
      false => (numbers.len() - 1) as f64,
    };
  });

  return variances;
}

pub fn std_deviation<
  T: Copy
    + Debug
    + FromPrimitive
    + Zero
    + Sub<T, Output = T>
    + SubAssign<T>
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + Into<f64>
    + DivAssign<T>,
>(
  numbers: &Vec<T>,
  is_population: bool,
) -> f64 {
  return variance(numbers, is_population).sqrt();
}

pub fn gaussian_probability(sample: f64, mean: f64, std_deviation: f64) -> f64 {
  let dist = Normal::new(mean, std_deviation).unwrap();
  return dist.pdf(sample);
}

/// Returns (feature col, purity, average for continuous data comparison)
pub fn get_min_purity(
  is_categorical: &Vec<bool>,
  data: &Vec<Vec<f64>>,
  labels: &Vec<f64>,
) -> (usize, f64, f64) {
  if (data.len() == 0) {
    py_print(&format!("INVALID DATA LENGTH GIVEN TO get_min_purity"));
  }
  let num_features = data[0].len();
  let mut min_feature_col = 0;
  let mut min_purity = f64::MAX;
  let mut min_average = 0.0;
  for feature in 0..num_features {
    let (purity, average) = get_purity(feature, is_categorical[feature], data, labels);
    if (purity < min_purity) {
      min_feature_col = feature;
      min_purity = purity;
      min_average = average;
    }
  }
  return (min_feature_col, min_purity, min_average);
}

pub fn get_purities(
  features_to_process: &HashSet<usize>,
  is_categorical: &Vec<bool>,
  data: &Vec<Vec<f64>>,
  labels: &Vec<f64>,
) -> HashMap<usize, (f64, f64)> {
  // returns feature -> (purity, average)
  let mut purities = HashMap::new();

  for col in features_to_process {
    let is_col_categorical = is_categorical[*col];
    purities.insert(*col, get_purity(*col, is_col_categorical, data, labels));
  }

  return purities;
}

pub fn get_purity(
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
    if (row_impurities.len() > 0) {
      purity = row_impurities[0].0;
      average = row_impurities[0].1;
    }
  }

  return (purity, average);
}

pub fn normalize<
  T: Copy
    + Debug
    + FromPrimitive
    + Zero
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + DivAssign<T>,
>(
  numbers: &mut Vec<T>,
) {
  let mut sum = T::zero();

  for val in numbers.iter() {
    sum += *val;
  }

  for val in numbers.iter_mut() {
    *val /= sum;
  }
}

pub fn get_residuals<T: Copy + Debug + FromPrimitive + Zero + Sub<T, Output = T> + SubAssign<T>>(
  observed: &Vec<T>,
  predicted: &Vec<T>,
) -> Vec<T> {
  let mut residuals = Vec::new();
  for (observed, predicted) in izip!(observed, predicted) {
    residuals.push(*observed - *predicted);
  }
  return residuals;
}
