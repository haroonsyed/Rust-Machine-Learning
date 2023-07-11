use cc;
use std::env;

fn main() {
  println!("cargo:rerun-if-changed=cuda_kernels/cuda_kernels.cu");

  cc::Build::new()
    .cuda(true)
    .cudart("static")
    .file("cuda_kernels/cuda_kernels.cu")
    .compile("cuda_kernels");

  if let Ok(cuda_path) = env::var("CUDA_HOME") {
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
  } else {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  }
  println!("cargo:rustc-link-lib=dylib=cudart");
}
