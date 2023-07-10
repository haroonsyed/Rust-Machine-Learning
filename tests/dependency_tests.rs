#[cfg(test)]
mod basic_nn_tests {

  use matrix_lib::bindings::test;

  #[test]
  fn cuda_test() {
    unsafe {
      test();
    }
  }
}
