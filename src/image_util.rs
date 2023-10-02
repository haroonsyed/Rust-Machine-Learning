use pyo3::{pyfunction, PyResult};

// Returns each image with its metadata. Raw Image: Vec<Vec<f32>>, Metadata: (label, width, height)
#[pyfunction]
pub fn load_raw_image_training_data(
  parent_folder: String,
) -> PyResult<(Vec<Vec<f32>>, Vec<(f32, f32, f32)>)> {
  let result = (Vec::new(), Vec::new());
  return Ok(result);
}
