use pyo3::prelude::*;
use std::fmt::Debug;

pub fn py_print<T: Debug>(s: &T) {
  let gil = Python::acquire_gil();
  let py = gil.python();
  let command = format!("{}{:?}{}", "print(", s, ")");
  py.run(&command, None, None);
}
