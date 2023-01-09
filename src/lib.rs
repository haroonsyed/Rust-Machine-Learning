pub mod basic_stats;
pub mod decision_tree;
pub mod gradient_descent;
pub mod k_means;
pub mod k_nearest_neighbor;
pub mod naive_bayes;
pub mod py_util;
pub mod regression_tree;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn Rust_Machine_Learning(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(k_means::k_means_cluster_2d, m)?)?;
  m.add_function(wrap_pyfunction!(k_means::get_closest_center_2d, m)?)?;
  m.add_function(wrap_pyfunction!(k_means::centers_are_equal, m)?)?;
  m.add_function(wrap_pyfunction!(
    k_nearest_neighbor::k_nearest_neighbor_2d,
    m
  )?)?;
  m.add_function(wrap_pyfunction!(gradient_descent::gradient_descent, m)?)?;
  m.add_class::<naive_bayes::naive_bayes_model>()?;
  m.add_class::<decision_tree::DecisionTree>()?;
  m.add_class::<regression_tree::RegressionTree>()?;
  Ok(())
}
