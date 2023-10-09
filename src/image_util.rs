use image::GenericImageView;
use itertools::Itertools;
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::{fs, path::PathBuf, time::Instant};

#[pyclass]
pub struct ImageBatchLoader {
  pub loader: ImageBatchLoaderRust,
}

#[pymethods]
impl ImageBatchLoader {
  #[new]
  fn new(parent_folder: String, sample_width: usize, sample_height: usize) -> Self {
    return ImageBatchLoader {
      loader: ImageBatchLoaderRust::new(parent_folder, sample_width, sample_height),
    };
  }

  // Set batch size to zero to load all samples
  fn batch_sample(&self, batch_size: usize) -> PyResult<(Vec<Vec<Vec<f32>>>, Vec<String>)> {
    return Ok(self.loader.batch_sample(batch_size));
  }
}

pub struct ImageBatchLoaderRust {
  pub paths_classifications: Vec<(PathBuf, String)>,
  pub sample_width: usize,
  pub sample_height: usize,
}

impl ImageBatchLoaderRust {
  fn get_image_paths_classifications(parent_folder: &String) -> Vec<(PathBuf, String)> {
    let mut paths_classifications = Vec::new();

    let entries = fs::read_dir(&parent_folder);
    if let Ok(entries) = entries {
      for entry in entries {
        if let Ok(entry) = entry {
          let folder_path = entry.path();
          let folder_name = entry.file_name();

          if folder_path.is_dir() {
            println!("Reading images for classification {:?}", &folder_name);

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
                    paths_classifications
                      .push((image_entry_path, folder_name.to_string_lossy().to_string()));
                  }
                }
              }
            }
          }
        }
      }
    }

    return paths_classifications;
  }

  pub fn new(parent_folder: String, sample_width: usize, sample_height: usize) -> Self {
    // Collect all the paths of images in the training set
    let paths_classifications = Self::get_image_paths_classifications(&parent_folder);

    return ImageBatchLoaderRust {
      paths_classifications,
      sample_width,
      sample_height,
    };
  }

  // Set the batch_size to 0 to load all the images
  pub fn batch_sample(&self, batch_size: usize) -> (Vec<Vec<Vec<f32>>>, Vec<String>) {
    let total_sample_count = self.paths_classifications.len();

    let start = Instant::now();

    let batch_sample = (0..batch_size)
      .collect_vec()
      .par_iter()
      .filter_map(|_| {
        // Load the pixel data for that image
        let mut rng = rand::thread_rng(); // Create a random number generator
        let img_index = rng.gen_range(0..total_sample_count);
        let (img_path, img_classification) = &self.paths_classifications[img_index];

        if let Ok(img) = image::open(&img_path) {
          let img = img.thumbnail_exact(self.sample_width as u32, self.sample_height as u32);

          // Process Image
          let (width, height) = img.dimensions();
          let mut pixel_data = vec![Vec::with_capacity((width * height) as usize); 3];

          // Check image will fit center crop
          // let top_left_x_crop = (width as i32 / 2) - (self.sample_width as i32 / 2) - 1;
          // let top_left_y_crop = (height as i32 / 2) - (self.sample_height as i32 / 2) - 1;
          // if top_left_x_crop < 0 || top_left_y_crop < 0 {
          //   return None;
          // }

          for (_, _, pixel) in img
            // .crop_imm(
            //   top_left_x_crop as u32,
            //   top_left_y_crop as u32,
            //   self.sample_width as u32,
            //   self.sample_height as u32,
            // )
            .pixels()
          {
            // Write to each depth of the image data
            // SEPRATE OUT INTO R G B IMAGES
            pixel_data[0].push(pixel.0[0] as f32);
            pixel_data[1].push(pixel.0[1] as f32);
            pixel_data[2].push(pixel.0[2] as f32);
          }

          // Write the image data and the metadata to the result
          return Some((pixel_data, img_classification.to_owned()));
        } else {
          return None;
        }
      })
      .unzip();

    let end = Instant::now();
    let exec_time = end - start;

    println!("Time to load samples: {}", exec_time.as_millis());

    return batch_sample;
  }
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