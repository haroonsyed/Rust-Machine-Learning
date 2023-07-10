use std::{fs, path::Path};

fn main() {
  #[cfg(target_os = "windows")]
  {
    if let Ok(cuda_lib_dir) = std::env::var("CUDA_PATH").map(|path| format!("{}/lib/x64", path)) {
      println!("CUDA library directory: {}", cuda_lib_dir);
      println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    } else {
      println!("CUDA_PATH environment variable not found");
    }
  }

  #[cfg(target_os = "linux")]
  {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
  }

  let out_dir = std::env::var("OUT_DIR").unwrap();
  let dest_dir = Path::new(&out_dir).join("../../../cuda_kernels");
  fs::create_dir_all(&dest_dir).unwrap();
  fs::copy(
    "cuda_kernels/libcuda_kernels.so",
    dest_dir.join("libcuda_kernels.so"),
  )
  .unwrap();
  println!("cargo:rustc-link-search=native={}", dest_dir.display());

  println!("cargo:rustc-link-lib=dylib=cuda_kernels");
}
