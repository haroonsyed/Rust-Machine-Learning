use pyo3::prelude::*;
use std::fmt::Debug;

pub fn py_print<T: Debug>(s: &T) {
  if cfg!(debug_assertions) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<()> {
      let command = format!("{}{:?}{}", "print(", s, ")");
      py.run(&command, None, None).expect("Failed to print!");
      Ok(())
    })
    .expect("Failed to print!");
  } else {
  }
}
