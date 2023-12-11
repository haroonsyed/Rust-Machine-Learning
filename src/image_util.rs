use image::GenericImageView;
use itertools::{izip, Itertools};
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::{collections::HashMap, fs, mem::size_of, path::PathBuf, time::Instant};
use tensor_lib::{
  create_matrix_group, cuda_bindings::memory_manager_get_pinned_allocation, Matrix,
};

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
  fn batch_sample(&self, batch_size: usize) -> PyResult<(Vec<Vec<Vec<f32>>>, Vec<f32>)> {
    return Ok(self.loader.batch_sample(batch_size));
  }

  fn get_classifications_map(&self) -> PyResult<HashMap<String, f32>> {
    return Ok(self.loader.classifications_map.clone());
  }
}

pub struct ImageBatchLoaderRust {
  pub classifications_map: HashMap<String, f32>,
  pub paths: Vec<PathBuf>,
  pub sample_width: usize,
  pub sample_height: usize,
}

impl ImageBatchLoaderRust {
  fn get_image_paths_classifications(
    parent_folder: &String,
  ) -> (Vec<PathBuf>, HashMap<String, f32>) {
    let mut paths = Vec::new();
    let mut classifications_map = HashMap::new();

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
                    paths.push(image_entry_path);
                    let new_classification_encoded = classifications_map.len() as f32; // The classification if it is not in map
                    classifications_map
                      .entry(folder_name.to_string_lossy().to_string())
                      .or_insert(new_classification_encoded);
                  }
                }
              }
            }
          }
        }
      }
    } else {
      println!("Error reading directory {:?}", parent_folder);
    }

    return (paths, classifications_map);
  }

  pub fn new(parent_folder: String, sample_width: usize, sample_height: usize) -> Self {
    // Collect all the paths of images in the training set
    let (paths, classifications_map) = Self::get_image_paths_classifications(&parent_folder);

    return ImageBatchLoaderRust {
      classifications_map,
      paths,
      sample_width,
      sample_height,
    };
  }

  pub fn batch_sample_as_matrix(&self, batch_size: usize) -> (Vec<Vec<Matrix>>, Matrix) {
    let total_sample_count = self.paths.len();

    // Reserve pinned memory for the batch
    let pinned_buffers = (0..batch_size * 3)
      .map(|_| {
        let buffer;
        unsafe {
          buffer = memory_manager_get_pinned_allocation(
            size_of::<f32>() * self.sample_width * self.sample_height,
          );
        }

        // Cast to size_t to get around rayon lol
        return buffer as usize;
      })
      .collect_vec();

    let grouped_pinned_buffers = pinned_buffers
      .iter()
      .chunks(3)
      .into_iter()
      .map(|sample| sample.into_iter().collect_vec())
      .collect_vec();

    let labels = grouped_pinned_buffers
      .par_iter()
      // .iter()
      .map(|pinned_channel_buffers| {
        // Load the pixel data for that image
        let mut rng = rand::thread_rng(); // Create a random number generator
        let img_index = rng.gen_range(0..total_sample_count);
        let img_path = &self.paths[img_index];
        let img_classification_string = img_path
          .parent()
          .unwrap()
          .file_name()
          .unwrap()
          .to_string_lossy()
          .to_string();
        let img_classification = self
          .classifications_map
          .get(&img_classification_string)
          .unwrap_or(&-1.0);

        if let Ok(img) = image::open(&img_path) {
          let img = img.thumbnail_exact(self.sample_width as u32, self.sample_height as u32);

          // Process Image
          let red_pointer = *pinned_channel_buffers[0] as *mut f32;
          let green_pointer = *pinned_channel_buffers[1] as *mut f32;
          let blue_pointer = *pinned_channel_buffers[2] as *mut f32;

          for (pixel_index, (_, _, pixel)) in img.pixels().enumerate() {
            unsafe {
              *(red_pointer.add(pixel_index)) = (pixel.0[0] as f32 / 255.0) - 0.5;
              *(green_pointer.add(pixel_index)) = (pixel.0[1] as f32 / 255.0) - 0.5;
              *(blue_pointer.add(pixel_index)) = (pixel.0[2] as f32 / 255.0) - 0.5;
            }
          }
        }

        return *img_classification;
      })
      .collect();

    // Convert data in pinned buffer to matrix
    let matrices = create_matrix_group(self.sample_width, self.sample_width, batch_size * 3);

    // Upload the data to the GPU
    izip!(matrices.iter(), pinned_buffers).for_each(|(matrix, pinned_buffer)| {
      matrix.set_data_from_pinned_buffer_async(pinned_buffer as *mut f32);
    });

    // Group the matrices into samples and channels
    let grouped_matrices = matrices
      .into_iter()
      .chunks(3)
      .into_iter()
      .map(|sample| sample.into_iter().collect_vec())
      .collect_vec();

    let encoded_labels =
      Matrix::new_one_hot_encoded(&labels, self.classifications_map.len()).transpose();

    return (grouped_matrices, encoded_labels);
  }

  // Set the batch_size to 0 to load all the images
  pub fn batch_sample(&self, batch_size: usize) -> (Vec<Vec<Vec<f32>>>, Vec<f32>) {
    let total_sample_count = self.paths.len();

    let start = Instant::now();

    let batch_sample: (Vec<Vec<Vec<f32>>>, Vec<f32>) = (0..batch_size)
      .collect_vec()
      .par_iter()
      .filter_map(|_| {
        // Load the pixel data for that image
        let mut rng = rand::thread_rng(); // Create a random number generator
        let img_index = rng.gen_range(0..total_sample_count);
        let img_path = &self.paths[img_index];
        let img_classification_string = img_path
          .parent()
          .unwrap()
          .file_name()
          .unwrap()
          .to_string_lossy()
          .to_string();
        let img_classification = &self
          .classifications_map
          .get(&img_classification_string)
          .unwrap_or(&-1.0);

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
            pixel_data[0].push((pixel.0[0] as f32 / 255.0) - 0.5);
            pixel_data[1].push((pixel.0[1] as f32 / 255.0) - 0.5);
            pixel_data[2].push((pixel.0[2] as f32 / 255.0) - 0.5);
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

    // println!("Time to load samples: {}", exec_time.as_millis());

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
