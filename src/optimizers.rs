use tensor_lib::Matrix;

pub trait Optimizer: Send {
  fn calculate_step(&mut self, curr_gradient: &Matrix) -> Matrix;
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
  fn calculate_step(&mut self, curr_gradient: &Matrix) -> Matrix {
    return curr_gradient.scalar_multiply(self.learning_rate);
  }
  fn clone_box(&self) -> Box<dyn Optimizer> {
    Box::new(self.clone())
  }
}

#[derive(Clone)]
pub struct MomentumOptimizer {
  learning_rate: f32,
  beta: f32,
  prev_gradient: Option<Matrix>,
}

impl MomentumOptimizer {
  pub fn new(learning_rate: f32, beta: f32) -> MomentumOptimizer {
    MomentumOptimizer {
      learning_rate,
      beta,
      prev_gradient: None,
    }
  }
}

impl Optimizer for MomentumOptimizer {
  fn calculate_step(&mut self, curr_gradient: &Matrix) -> Matrix {
    let adjusted_gradient = match &self.prev_gradient {
      Some(prev_gradient) => {
        // dW = (beta * prev_grad + (1 - beta) * curr_gradient)
        prev_gradient
          .scalar_multiply(self.beta)
          .element_add(&(curr_gradient.scalar_multiply(1.0 - self.beta)))
      }
      None => curr_gradient.deep_copy(),
    };

    self.prev_gradient = Some(adjusted_gradient.clone());

    return adjusted_gradient.scalar_multiply(self.learning_rate);
  }
  fn clone_box(&self) -> Box<dyn Optimizer> {
    Box::new(self.clone())
  }
}
