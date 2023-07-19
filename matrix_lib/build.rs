use cc;
use std::{env, path::Path};

fn main() {
  println!("cargo:rerun-if-changed=cuda_kernels/cuda_kernels.cu");

  println!("cargo:rustc-link-lib=dylib=cublas");

  let cublas_path =
    Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/cublas.lib");

  cc::Build::new()
    .cuda(true)
    .cudart("static")
    // .object(cublas_path)
    .file("cuda_kernels/cuda_kernels.cu")
    .compile("cuda_kernels");

  if let Ok(cuda_path) = env::var("CUDA_HOME") {
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
  } else {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  }
  println!("cargo:rustc-link-lib=dylib=cudart");
}
