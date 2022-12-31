use num::{FromPrimitive, Zero};
use std::{
  collections::HashMap,
  hash::Hash,
  ops::{Add, AddAssign, Div, DivAssign},
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
