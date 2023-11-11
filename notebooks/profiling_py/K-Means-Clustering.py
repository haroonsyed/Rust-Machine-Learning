#!/usr/bin/env python
# coding: utf-8

# In[ ]:


:sccache 1
:dep ndarray
:dep plotly
:dep polars
:dep rand

extern crate ndarray;
extern crate plotly;
extern crate polars;
extern crate rand;

use ndarray::Array;
use plotly::*;
use plotly::common::Mode;
use polars::prelude::CsvReader;
use polars::prelude::*;
use rand::Rng;


# In[19]:


// Grab the dataset from local csv file
// Datasets are courtesy of:
// http://archive.ics.uci.edu/ml/datasets/
// https://github.com/milaan9/Clustering-Datasets
let df = CsvReader::from_path("./data/kmeans-dataset.csv").unwrap().finish().unwrap();
println!("{:?}", df);


# In[36]:


let x0:Vec<f64> = df[0].f64()?.into_no_null_iter().collect();
let y0:Vec<f64> = df[1].f64()?.into_no_null_iter().collect();

let trace = Scatter::new(x0.clone(), y0.clone()).mode(Mode::Markers);
let mut plot = Plot::new();
plot.add_trace(trace);

let layout = Layout::new().height(525);
plot.set_layout(layout);

plot.notebook_display();


# In[29]:


// I could probably make it generalized so it can be n-dimensional but I just want the concept
fn k_means_cluster_2d(number_of_clusters: i64, x: &Vec<f64>, y: &Vec<f64>) -> Vec<Vec<f64>> {
    let mut centers = Vec::new();
    
    // Random seed the centers
    let mut rng = rand::thread_rng();
    
    let min_x:f64 = * x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_x:f64 = * x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_y:f64 = * y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_y:f64 = * y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    for _i in 0..number_of_clusters {
        centers.push(vec![rng.gen_range(min_x..max_x),rng.gen_range(min_y..max_y)]);
    }
    
    // Run iteratively until the centers stop changing
    let mut centers_did_change:bool = true;
    while centers_did_change == true {
        // Clustering algorithm assigns each point to closest cluster.
        // Then the new centers becomes the avg of each of the clusters
        let mut new_centers = Vec::new();
        let mut point_count_centers = vec![1;number_of_clusters.try_into().unwrap()];
        for _i in 0..number_of_clusters {
            new_centers.push(vec![0.0,0.0,0.0]); // x, y, numberOfPoints
        }
        
        for (x_val, y_val) in x.iter().zip(y.iter()) {
            let closest_center = get_closest_center_2d(&centers, &x_val, &y_val);
            new_centers[closest_center][0] += x_val;
            new_centers[closest_center][1] += y_val;
            point_count_centers[closest_center] += 1;
        }
        
        // Divide the new_centers by the number of points assigned to each cluster
        for (i,center) in new_centers.iter_mut().enumerate() {
            center[0] /= point_count_centers[i] as f64;
            center[1] /= point_count_centers[i] as f64;
        }
        
        centers_did_change = !centers_are_equal(&centers, &new_centers);
        
        // centers = the new centers[0:1] (we don't care about number of points in each)
        centers = new_centers;
    }
    return centers;
}

fn get_closest_center_2d(clusters: &Vec<Vec<f64>>, x: &f64, y: &f64)->usize {
    let mut best_center = 0;
    let mut best_center_dist = f64::MAX;
    for (i,center) in clusters.iter().enumerate() {
        // Minimize the distance. Add point to bestCenter in new centers
        // Optimize by not taking sqrt
        let dist_squared = (x-center[0]).powf(2.0) + (y-center[1]).powf(2.0);
        if dist_squared < best_center_dist {
            best_center = i;
            best_center_dist = dist_squared;
        }
    }
    return best_center;
}

fn centers_are_equal(centers1:&Vec<Vec<f64>>, centers2:&Vec<Vec<f64>>)->bool {
    if(centers1.len() != centers2.len()){
        return false;
    }
    for (center1, center2) in centers1.iter().zip(centers2.iter()) {
        if(center1[0] != center2[0] || center1[1] != center2[1]) {
            return false;
        }
    }
    return true;
}


# In[49]:


let centers:Vec<Vec<f64>> = k_means_cluster_2d(9, &x0, &y0);

let mut clustered_data:Vec<Vec<Vec<f64>>> = vec![Vec::new(); centers.len()];
for (x_val, y_val) in x0.iter().zip(y0.iter()) {
    let closest_center = get_closest_center_2d(&centers, &x_val, &y_val);
    clustered_data[closest_center].push(vec![*x_val, *y_val]);
}

// let trace1 = Scatter::new(x0.clone(), y0.clone()).mode(Mode::Markers);
// let trace2 = Scatter::new(centers.iter().map(|center| center[0]).collect(),centers.iter().map(|center| center[1]).collect()).mode(Mode::Markers);
let mut plot2 = Plot::new();

// Add clustered data to plot
for cluster in clustered_data {
    let cluster_x_vals = cluster.iter().map(|point| point[0]).collect();
    let cluster_y_vals = cluster.iter().map(|point| point[1]).collect();
    let trace = Scatter::new(cluster_x_vals, cluster_y_vals).mode(Mode::Markers);
    plot2.add_trace(trace);
}

let layout2 = Layout::new().height(525);
plot2.set_layout(layout2);

plot2.notebook_display();


# In[ ]:




