use cc;
use std::{env, path::PathBuf};

fn main() {
  println!("cargo:rerun-if-changed=cuda_kernels/cuda_kernels.cu");

  cc::Build::new()
    .cuda(true)
    .cudart("static")
    .flag("-gencode")
    .flag("arch=compute_86,code=sm_86")
    .file("cuda_kernels/src/cuda_kernels.cu")
    .compile("cuda_kernels");

  if let Ok(cuda_path) = env::var("CUDA_HOME") {
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
  } else {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  }
  println!("cargo:rustc-link-lib=dylib=cudart");
  println!("cargo:rustc-link-lib=dylib=cublas");
}
