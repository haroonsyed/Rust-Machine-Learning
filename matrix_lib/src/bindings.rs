use std::ffi::c_double;

extern "C" {
  pub fn test();
  pub fn test_array_fill(out: *mut c_double, length: usize);
}
