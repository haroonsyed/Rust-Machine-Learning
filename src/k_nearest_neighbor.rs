use itertools::izip;
use pyo3::prelude::*;
use std::cmp::Eq;
use std::collections::HashMap;
use std::hash::Hash;

fn get_mode<T: Eq + Hash + Copy>(data: &Vec<T>) -> T {
  let mut counts = HashMap::new();

  for val in data.iter() {
    if !counts.contains_key(val) {
      counts.insert(val, 0);
    }
    *counts.get_mut(val).unwrap() += 1;
  }

  // Find the most common in the map
  let mut highest_count = -1;
  let mut mode = None;
  for (val, count) in counts {
    if count > highest_count {
      highest_count = count;
      mode = Some(val);
    }
  }

  return *mode.unwrap();
}

// Can probably be accelerated and generalized with a k-d tree.
// But this gets the general idea.
// Will classify the points in the test dataset using the train set
#[pyfunction]
pub fn k_nearest_neighbor_2d(
  x_train: Vec<f64>,
  y_train: Vec<f64>,
  labels: Vec<i64>,
  x_test: Vec<f64>,
  y_test: Vec<f64>,
  k: i64,
) -> PyResult<Vec<i64>> {
  // Returns the index, classification of each train point
  let mut label = Vec::new();

  // For each point in the test dataset
  for (x_val_test, y_val_test) in x_test.iter().zip(y_test.iter()) {
    // Add all distances to the distances vector
    let mut closest_data = Vec::new();
    for (x_val, y_val, label) in izip!(&x_train, &y_train, &labels) {
      let dist_squared = (x_val - x_val_test).powf(2.0) + (y_val - y_val_test).powf(2.0);
      closest_data.push((dist_squared, label));
    }

    // Sort the distances
    closest_data.sort_by(|a, b| (a.0 as f64).partial_cmp(&b.0).unwrap());
    closest_data = closest_data[0..k as usize].to_vec();
    let top_labels: Vec<i64> = closest_data.iter().map(|&x| *x.1 as i64).collect();

    // Get the mode
    let mode = get_mode(&top_labels);
    label.push(mode);
  }

  return Ok(label);
}
