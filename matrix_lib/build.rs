use cc;
use std::env;

fn main() {
  if let Ok(cuda_path) = env::var("CUDA_HOME") {
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
  } else {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  }

  println!("cargo:rustc-link-lib=dylib=cuda");
  println!("cargo:rustc-link-lib=dylib=cudart");
  println!("cargo:rustc-link-lib=dylib=cublas");
  println!("cargo:rustc-link-lib=dylib=curand");

  cc::Build::new()
    .cuda(true)
    .file("cuda_kernels/cuda_kernels.cu")
    .compile("cuda_kernels");
}
