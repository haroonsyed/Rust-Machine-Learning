use tensor_lib::Matrix;

pub trait Optimizer: Send {
  fn calculate_step(&self, curr_gradient: &Matrix) -> Matrix;
  fn clone_box(&self) -> Box<dyn Optimizer>;
}

impl Clone for Box<dyn Optimizer> {
  fn clone(&self) -> Box<dyn Optimizer> {
    self.clone_box()
  }
}

#[derive(Clone)]
pub struct StochasticGradientDescentOptimizer {
  learning_rate: f32,
}

impl StochasticGradientDescentOptimizer {
  pub fn new(learning_rate: f32) -> StochasticGradientDescentOptimizer {
    StochasticGradientDescentOptimizer { learning_rate }
  }
}

impl Optimizer for StochasticGradientDescentOptimizer {
  fn calculate_step(&self, curr_gradient: &Matrix) -> Matrix {
    return curr_gradient.scalar_multiply(self.learning_rate);
  }
  fn clone_box(&self) -> Box<dyn Optimizer> {
    Box::new(self.clone())
  }
}

#[derive(Clone)]
pub struct MomentumOptimizer {
  learning_rate: f32,
  momentum: f32,
  prev_step: Matrix,
}

impl MomentumOptimizer {
  pub fn new(learning_rate: f32, momentum: f32) -> MomentumOptimizer {
    MomentumOptimizer {
      learning_rate,
      momentum,
      prev_step: Matrix::zeros(0, 0),
    }
  }
}
