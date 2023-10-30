#[cfg(test)]
mod basic_nn_tests {

  use matrix_lib::cuda_bindings::test;

  #[test]
  fn cuda_test() {
    unsafe {
      test();
    }
  }
}
