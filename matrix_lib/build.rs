use std::{env, path::PathBuf};

fn main() {
  if let Ok(cuda_lib_dir) = env::var("CUDA_PATH").map(|path| {
    #[cfg(target_os = "windows")]
    {
      format!("{}/lib/x64", path)
    }
    #[cfg(target_os = "linux")]
    {
      format!("{}/lib64", path)
    }
  }) {
    println!("CUDA library directory: {}", cuda_lib_dir);
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
  } else {
    println!("CUDA_PATH environment variable not found");
  }

  // Get the directory containing the `matrix_lib` crate's `Cargo.toml`
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest directory");

  // Construct the relative path to the cuda_kernels directory
  let cuda_kernels_dir = PathBuf::from(manifest_dir).join("cuda_kernels");

  // Convert the path to a string
  let cuda_kernels_dir_str = cuda_kernels_dir
    .to_str()
    .expect("Failed to convert path to string");

  // Set the search path for the linker
  println!("cargo:rustc-link-search={}", cuda_kernels_dir_str);
  println!("cargo:rustc-link-lib=static=cuda_kernels");
  println!("cargo:rustc-link-lib=static=cudart");
}
