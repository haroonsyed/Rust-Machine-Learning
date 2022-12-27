use pyo3::prelude::*;
use rand::prelude::*;

// I could probably make it generalized so it can be n-dimensional but I just want the concept
#[pyfunction]
pub fn k_means_cluster_2d(
    number_of_clusters: i64,
    x: Vec<f64>,
    y: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let mut centers = Vec::new();

    // Random seed the centers
    let mut rng = rand::thread_rng();

    let min_x: f64 = *x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_x: f64 = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_y: f64 = *y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_y: f64 = *y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    for _i in 0..number_of_clusters {
        centers.push(vec![
            rng.gen_range(min_x..max_x),
            rng.gen_range(min_y..max_y),
        ]);
    }

    // Run iteratively until the centers stop changing
    let mut centers_did_change: bool = true;
    while centers_did_change == true {
        // Clustering algorithm assigns each point to closest cluster.
        // Then the new centers becomes the avg of each of the clusters
        let mut new_centers = Vec::new();
        for _i in 0..number_of_clusters {
            new_centers.push(vec![0.0, 0.0, 0.0]); // x, y, numberOfPoints
        }

        for (x_val, y_val) in x.iter().zip(y.iter()) {
            let closest_center = get_closest_center_2d_internal(&centers, &x_val, &y_val);
            new_centers[closest_center][0] += x_val;
            new_centers[closest_center][1] += y_val;
            new_centers[closest_center][2] += 1.0;
        }

        // Divide the new_centers by the number of points assigned to each cluster
        for center in new_centers.iter_mut() {
            center[0] /= center[2];
            center[1] /= center[2];
        }

        centers_did_change = !centers_are_equal_internal(&centers, &new_centers);

        // centers = the new centers[0:1] (we don't care about number of points in each)
        centers = new_centers;
    }
    return Ok(centers);
}

#[pyfunction]
pub fn get_closest_center_2d(clusters: Vec<Vec<f64>>, x: f64, y: f64) -> PyResult<usize> {
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

fn get_closest_center_2d_internal(clusters: &Vec<Vec<f64>>, x: &f64, y: &f64) -> usize {
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
    return best_center;
}

#[pyfunction]
pub fn centers_are_equal(centers1: Vec<Vec<f64>>, centers2: Vec<Vec<f64>>) -> PyResult<bool> {
    if (centers1.len() != centers2.len()) {
        return Ok(false);
    }
    for (center1, center2) in centers1.iter().zip(centers2.iter()) {
        if (center1[0] != center2[0] || center1[1] != center2[1]) {
            return Ok(false);
        }
    }
    return Ok(true);
}

fn centers_are_equal_internal(centers1: &Vec<Vec<f64>>, centers2: &Vec<Vec<f64>>) -> bool {
    if (centers1.len() != centers2.len()) {
        return false;
    }
    for (center1, center2) in centers1.iter().zip(centers2.iter()) {
        if (center1[0] != center2[0] || center1[1] != center2[1]) {
            return false;
        }
    }
    return true;
}
