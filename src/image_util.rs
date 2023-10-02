

#[pyfunction]
pub fn get_images_(clusters: Vec<Vec<f64>>, x: f64, y: f64) -> PyResult<(Vec<Vec<f32>>e)> {
  let mut best_center = 0;
  let mut best_center_dist = f64::MAX;
  for (i, center) in clusters.iter().enumerate() {
    // Minimize the distance. Add point to bestCenter in new centers
    // Optimize by not taking sqrt
    let dist_squared = (x - center[0]).powf(2.0) + (y - center[1]).powf(2.0);
    if dist_squared < best_center_dist {
      best_center = i;
      best_center_dist = dist_squared;
    }
  }
  return Ok(best_center);
}