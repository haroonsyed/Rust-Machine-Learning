use image::GenericImageView;
use pyo3::{pyfunction, PyResult};
use std::fs;

// Returns each image with its metadata. Raw Image: Vec<Vec<f32>>, Metadata: (label, width, height)
#[pyfunction]
pub fn load_raw_image_training_data(
  parent_folder: String,
) -> PyResult<(Vec<Vec<f32>>, Vec<(String, f32, f32)>)> {
  let mut result = (Vec::new(), Vec::new());

  let entries = fs::read_dir(parent_folder);

  if let Ok(entries) = entries {
    for entry in entries {
      if let Ok(entry) = entry {
        let folder_path = entry.path();
        let folder_name = entry.file_name();

        if folder_path.is_dir() {
          // Go through all the images in the directory
          let image_entries = fs::read_dir(folder_path);
          if let Ok(image_entries) = image_entries {
            for image_entry in image_entries {
              if let Ok(image_entry) = image_entry {
                let image_entry_path = image_entry.path();

                // Double check it is an image
                let is_image = image_entry_path.ends_with("jpg")
                  || image_entry_path.ends_with("jpeg")
                  || image_entry_path.ends_with("png")
                  || image_entry_path.extension().is_none();
                if is_image {
                  if let Ok(img) = image::open(&image_entry_path) {
                    // Process Image
                    let (width, height) = img.dimensions();
                    let pixel_data = Vec::with_capacity((width * height) as usize);
                    for (_, _, pixel) in img.pixels() {
                      pixel_data.push(pixel.0); // SEPRATE OUT INTO R G B IMAGES
                    }
                    result.0.push(pixel_data);
                    result.1.push((folder_name, width, height));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return Ok(result);
}
