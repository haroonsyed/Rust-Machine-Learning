use itertools::{izip, Itertools};
use pyo3::prelude::*;

#[pyfunction]
pub fn gradient_descent(
  x: Vec<f64>,
  y: Vec<f64>,
  learn_rate: f64,
  max_iter: usize,
) -> PyResult<(f64, f64)> {
  let mut slope = 1.0;
  let mut intercept = 1.0;

  // Find gradient for intercept and slope at each iteration
  for i in 0..max_iter {
    let g_intercept = ssr_gradient_intercept(&x, &y, &slope, &intercept);
    let g_slope = ssr_gradient_slope(&x, &y, &slope, &intercept);

    let step_intercept = learn_rate * g_intercept;
    let step_slope = learn_rate * g_slope;

    // Recalculate the intercept and slope based on steps
    // Old - step_size
    intercept -= step_intercept;
    slope -= step_slope;
  }

  return Ok((intercept, slope));
}

fn ssr(x: &Vec<f64>, y: &Vec<f64>, coefficients: &Vec<f64>) -> f64 {
  let mut ssr = 0.0;

  for (x, y) in izip!(x, y) {
    let expected = y;
    let predicted = coefficients[0] + x * coefficients[1];

    ssr += (expected - predicted).powf(2.0);
  }

  return ssr;
}

fn ssr_gradient_intercept(x: &Vec<f64>, y: &Vec<f64>, slope: &f64, intercept: &f64) -> f64 {
  let mut gradient = 0.0;

  // Original ssr equation is: (y - (intercept + slope * x))**2
  // d/intercept = 2 * -1 (y - (intercept + slope * x))

  for (x, y) in izip!(x, y) {
    gradient += -2.0 * (y - (intercept + slope * x));
  }

  return gradient;
}

fn ssr_gradient_slope(x: &Vec<f64>, y: &Vec<f64>, slope: &f64, intercept: &f64) -> f64 {
  let mut gradient = 0.0;

  // Original ssr equation is: (y - (intercept + slope * x))**2
  // d/slope = 2 * -x (y - (intercept + slope * x))

  for (x, y) in izip!(x, y) {
    gradient += -2.0 * x * (y - (intercept + slope * x));
  }

  return gradient;
}
