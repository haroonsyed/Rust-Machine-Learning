use image::GenericImageView;
use pyo3::{pyfunction, PyResult};
use std::{fs, path::PathBuf};

// Returns each image with its metadata. Raw Image: Vec<Vec<Vec<f32>>>, Metadata: Vec<(label, width, height)>
//                                                  Example->depth->data
#[pyfunction]
pub fn load_raw_image_training_data(
  parent_folder: String,
  max_images_to_load: usize, // Set to 0 to load all images
) -> PyResult<(Vec<Vec<Vec<f32>>>, Vec<(String, usize, usize)>)> {
  let mut result = (Vec::new(), Vec::new());

  let num_classifications = fs::read_dir(&parent_folder)
    .map(|entries| entries.filter_map(|entry| entry.ok()))
    .map(|entries| entries.filter(|entry| entry.path().is_dir()))
    .map(|entries| entries.count())
    .unwrap_or(1);

  let max_images_per_folder = if max_images_to_load == 0 {
    1000000
  } else {
    max_images_to_load / num_classifications
  };

  println!(
    "Reading directory {}, with {} classifications",
    &parent_folder, num_classifications
  );

  let entries = fs::read_dir(&parent_folder);
  if let Ok(entries) = entries {
    for entry in entries {
      if let Ok(entry) = entry {
        let folder_path = entry.path();
        let folder_name = entry.file_name();

        if folder_path.is_dir() {
          println!("Reading images for classification {:?}", &folder_name);
          let mut images_in_folder_loaded = 0;

          // Go through all the images in the directory
          let image_entries = fs::read_dir(&folder_path);
          if let Ok(image_entries) = image_entries {
            for image_entry in image_entries {
              if let Ok(image_entry) = image_entry {
                let image_entry_path = image_entry.path();

                // Double check it is an image
                let is_image = has_extension(&image_entry_path, "jpg")
                  || has_extension(&image_entry_path, "jpeg")
                  || has_extension(&image_entry_path, "png")
                  || image_entry_path.extension().is_none();
                if is_image {
                  if let Ok(img) = image::open(&image_entry_path) {
                    if images_in_folder_loaded >= max_images_per_folder {
                      break;
                    }
                    images_in_folder_loaded += 1;

                    // Process Image
                    let (width, height) = img.dimensions();
                    let mut pixel_data = vec![Vec::with_capacity((width * height) as usize); 3];
                    for (_, _, pixel) in img.pixels() {
                      // Write to each depth of the image data
                      // SEPRATE OUT INTO R G B IMAGES
                      pixel_data[0].push(pixel.0[0] as f32);
                      pixel_data[1].push(pixel.0[1] as f32);
                      pixel_data[2].push(pixel.0[2] as f32);
                    }

                    // Write the image data and the metadata to the result
                    result.0.push(pixel_data);
                    result.1.push((
                      folder_name.to_string_lossy().to_string(),
                      width as usize,
                      height as usize,
                    ));
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

fn has_extension(path: &PathBuf, target_extension: &str) -> bool {
  // Compare the file's extension (in lowercase) with the target extension
  if let Some(ext) = path.extension() {
    if let Some(ext_str) = ext.to_str() {
      return ext_str.to_lowercase() == target_extension.to_lowercase();
    }
  }

  false // Return false if the file has no extension or it couldn't be converted to a string
}
