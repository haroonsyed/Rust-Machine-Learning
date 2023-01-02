mod basic_stats;
mod k_means;
mod k_nearest_neighbor;
mod naive_bayes;
mod py_util;

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
  m.add_class::<naive_bayes::naive_bayes_model>()?;
  Ok(())
}
