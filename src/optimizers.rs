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

#[derive(Clone)]
pub struct AdagradOptimizer {
  learning_rate: f32,
  epsilon: f32,
  accumulated_gradient: Option<Matrix>,
}

impl AdagradOptimizer {
  pub fn new(learning_rate: f32) -> AdagradOptimizer {
    AdagradOptimizer {
      learning_rate,
      epsilon: 1e-8,
      accumulated_gradient: None,
    }
  }
}

impl Optimizer for AdagradOptimizer {
  fn calculate_step(&mut self, curr_gradient: &Matrix) -> Matrix {
    let adjusted_gradient = match &mut self.accumulated_gradient {
      Some(accumulated_gradient) => {
        // accumulated_gradient = accumulated_gradient + curr_gradient^2
        // adjusted_gradient = curr_gradient / (sqrt(accumulated_gradient) + epsilon)
        // The idea is the 1/sqrt(accumulated_gradient) will act as a learning rate decay, so we can more smartly update per weight
        // Otherwise we can see this is the same as SGD (no momentum involved here)

        *accumulated_gradient =
          accumulated_gradient.element_add(&curr_gradient.element_multiply(&curr_gradient));

        let adjusted_gradient = curr_gradient
          .element_divide(&(accumulated_gradient.element_sqrt().scalar_add(self.epsilon)));

        adjusted_gradient
      }
      None => {
        self.accumulated_gradient = Some(curr_gradient.element_multiply(&curr_gradient));
        curr_gradient.deep_copy()
      }
    };

    return adjusted_gradient.scalar_multiply(self.learning_rate);
  }

  fn clone_box(&self) -> Box<dyn Optimizer> {
    Box::new(self.clone())
  }
}

#[derive(Clone)]
pub struct RMSPropOptimizer {
  learning_rate: f32,
  beta: f32,
  epsilon: f32,
  accumulated_gradient: Option<Matrix>,
}

impl RMSPropOptimizer {
  pub fn new(learning_rate: f32, beta: f32) -> RMSPropOptimizer {
    RMSPropOptimizer {
      learning_rate,
      beta,
      epsilon: 1e-8,
      accumulated_gradient: None,
    }
  }
}

impl Optimizer for RMSPropOptimizer {
  fn calculate_step(&mut self, curr_gradient: &Matrix) -> Matrix {
    let adjusted_gradient = match &mut self.accumulated_gradient {
      Some(prev_accumulated_gradient) => {
        // adjusted_gradient = curr_gradient/sqrt(accumulated_gradient)
        // accumulated_gradient = (beta * prev_accumulated_gradient + (1 - beta) * curr_gradient ^ 2)

        // The idea expands on Adagrad, but adds a decay factor to the accumulated gradient
        // This means that the learning rate will decay over time, but not as aggressively as Adagrad
        // This is because older weights are continually decayed by beta

        *prev_accumulated_gradient = prev_accumulated_gradient
          .scalar_multiply(self.beta)
          .element_add(
            &(curr_gradient
              .element_multiply(&curr_gradient)
              .scalar_multiply(1.0 - self.beta)),
          );

        let adjusted_gradient = curr_gradient.element_divide(
          &prev_accumulated_gradient
            .element_sqrt()
            .scalar_add(self.epsilon),
        );
        adjusted_gradient
      }
      None => {
        let accumulated_gradient = curr_gradient
          .element_multiply(&curr_gradient)
          .scalar_multiply(1.0 - self.beta);

        self.accumulated_gradient = Some(accumulated_gradient.clone());

        let adjusted_gradient = curr_gradient
          .element_divide(&accumulated_gradient.element_sqrt().scalar_add(self.epsilon));
        adjusted_gradient
      }
    };

    return adjusted_gradient.scalar_multiply(self.learning_rate);
  }

  fn clone_box(&self) -> Box<dyn Optimizer> {
    Box::new(self.clone())
  }
}
