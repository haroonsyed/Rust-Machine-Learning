use num::{FromPrimitive, Zero};
use statrs::distribution::{Continuous, Normal};
use std::{
  collections::HashMap,
  hash::Hash,
  ops::{Add, AddAssign, Div, DivAssign, Sub, SubAssign},
};

pub fn mean<
  T: Copy
    + FromPrimitive
    + Zero
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + DivAssign<T>,
>(
  numbers: &Vec<T>,
) -> T {
  let mut sum = numbers[0];

  for val in numbers {
    sum += *val;
  }

  sum /= FromPrimitive::from_usize(numbers.len()).unwrap();

  return sum;
}

pub fn mean_2d<
  T: Copy
    + FromPrimitive
    + Zero
    + Add<T, Output = T>
    + AddAssign<T>
    + Div<T, Output = T>
    + DivAssign<T>,
>(
  numbers: &Vec<Vec<T>>,
) -> Vec<T> {
  let num_features = numbers[0].len();
  let mut means = vec![T::zero(); num_features];

  for data_point in numbers {
    for (i, val) in data_point.iter().enumerate() {
      means[i] += *val;
    }
  }

  means
    .iter_mut()
    .for_each(|x| *x /= FromPrimitive::from_usize(num_features).unwrap());

  return means;
}

pub fn median<T: Copy + Ord + Clone>(numbers: &Vec<T>) -> T {
  let mut copy = numbers.clone();
  copy.sort();
  return copy[numbers.len() / 2];
}

pub fn mode<T: Copy + Hash + Eq>(numbers: &Vec<T>) -> T {
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

pub fn variance<
  T: Copy
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
