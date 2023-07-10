pub mod adaboost;
pub mod basic_neural_network;
pub mod basic_stats;
pub mod decision_tree;
pub mod gradient_descent;
pub mod gradientboost;
pub mod k_means;
pub mod k_nearest_neighbor;
pub mod naive_bayes;
pub mod py_util;
pub mod regression_tree;
pub mod xgb;

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
  m.add_class::<naive_bayes::NaiveBayesModel>()?;
  m.add_class::<decision_tree::DecisionTree>()?;
  m.add_class::<regression_tree::RegressionTree>()?;
  m.add_class::<adaboost::AdaBoost>()?;
  m.add_class::<gradientboost::GradientBoost>()?;
  m.add_class::<xgb::XGB>()?;
  m.add_class::<basic_neural_network::BasicNeuralNetwork>()?;
  Ok(())
}
