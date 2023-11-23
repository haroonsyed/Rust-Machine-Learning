use cc;
use std::env;

fn main() {
  println!("cargo:rerun-if-changed=cuda_kernels/cuda_kernels.cu");
  println!("cargo:rerun-if-changed=cuda_kernels/cuda_exec_memory_manager.cu");
  println!("cargo:rerun-if-changed=cuda_kernels/err_check_util.cu");

  cc::Build::new()
    .cuda(true)
    .cudart("static")
    // 52;75;86;89;90
    // Generate -gencode and arch for each of theses
    .flag("-gencode")
    .flag("arch=compute_52,code=sm_52")
    .flag("-gencode")
    .flag("arch=compute_75,code=sm_75")
    .flag("-gencode")
    .flag("arch=compute_86,code=sm_86")
    .flag("-gencode")
    .flag("arch=compute_89,code=sm_89")
    .flag("-gencode")
    .flag("arch=compute_90,code=sm_90")
    .flag("--use_fast_math")
    .file("cuda_kernels/src/cuda_kernels.cu")
    .file("cuda_kernels/src/cuda_exec_memory_manager.cu")
    .file("cuda_kernels/src/err_check_util.cu")
    .compile("cuda_kernels");

  if let Ok(cuda_path) = env::var("CUDA_HOME") {
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
  } else {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  }
  println!("cargo:rustc-link-lib=dylib=cudart");
  println!("cargo:rustc-link-lib=dylib=cublas");
}
