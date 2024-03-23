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
          .scalar_multiply_inplace(self.beta)
          .element_add_inplace(&(curr_gradient.scalar_multiply(1.0 - self.beta)));
        prev_gradient.clone()
      }
      None => {
        let gradient = curr_gradient.deep_copy();
        self.prev_gradient = Some(gradient.clone());
        gradient
      }
    };

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
        accumulated_gradient.element_add_inplace(&curr_gradient.element_multiply(&curr_gradient));

        let adjusted_gradient = curr_gradient.element_divide(
          &(accumulated_gradient
            .element_sqrt()
            .scalar_add_inplace(self.epsilon)),
        );

        adjusted_gradient
      }
      None => {
        self.accumulated_gradient = Some(curr_gradient.element_multiply(&curr_gradient));
        curr_gradient.deep_copy()
      }
    };

    return adjusted_gradient.scalar_multiply_inplace(self.learning_rate);
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

        prev_accumulated_gradient
          .scalar_multiply_inplace(self.beta)
          .element_add_inplace(
            &(curr_gradient
              .element_multiply(&curr_gradient)
              .scalar_multiply_inplace(1.0 - self.beta)),
          );

        let adjusted_gradient = curr_gradient.element_divide(
          &prev_accumulated_gradient
            .element_sqrt()
            .scalar_add_inplace(self.epsilon),
        );
        adjusted_gradient
      }
      None => {
        let accumulated_gradient = curr_gradient
          .element_multiply(&curr_gradient)
          .scalar_multiply_inplace(1.0 - self.beta);

        self.accumulated_gradient = Some(accumulated_gradient.clone());

        let adjusted_gradient = curr_gradient.element_divide(
          &accumulated_gradient
            .element_sqrt()
            .scalar_add_inplace(self.epsilon),
        );
        adjusted_gradient
      }
    };

    return adjusted_gradient.scalar_multiply(self.learning_rate);
  }

  fn clone_box(&self) -> Box<dyn Optimizer> {
    Box::new(self.clone())
  }
}

#[derive(Clone)]
pub struct AdamOptimizer {
  learning_rate: f32,
  beta1: f32,
  beta2: f32,
  epsilon: f32,
  t: usize,
  d_v: Option<Matrix>,
  d_s: Option<Matrix>,
}

impl AdamOptimizer {
  pub fn new(learning_rate: f32, beta1: f32, beta2: f32) -> AdamOptimizer {
    AdamOptimizer {
      learning_rate,
      beta1,
      beta2,
      epsilon: 1e-8,
      t: 1,
      d_v: None,
      d_s: None,
    }
  }
}

impl Optimizer for AdamOptimizer {
  fn calculate_step(&mut self, curr_gradient: &Matrix) -> Matrix {
    let momentum_gradient = match &mut self.d_v {
      Some(prev_d_v) => {
        // dv = (beta * prev_grad + (1 - beta) * curr_gradient)
        prev_d_v
          .scalar_multiply_inplace(self.beta1)
          .element_add_inplace(&(curr_gradient.scalar_multiply(1.0 - self.beta1)));
        prev_d_v.clone()
      }
      None => {
        let d_v = curr_gradient.scalar_multiply(1.0 - self.beta1);
        self.d_v = Some(d_v.clone());
        d_v
      }
    };

    let rms_prop_gradient = match &mut self.d_s {
      Some(prev_d_s) => {
        // adjusted_gradient = 1/sqrt(accumulated_gradient)
        // accumulated_gradient = (beta * prev_accumulated_gradient + (1 - beta) * curr_gradient ^ 2)
        prev_d_s
          .scalar_multiply_inplace(self.beta2)
          .element_add_inplace(
            &(curr_gradient
              .element_multiply(&curr_gradient)
              .scalar_multiply_inplace(1.0 - self.beta2)),
          );

        prev_d_s.element_sqrt().scalar_add_inplace(self.epsilon)
      }
      None => {
        let d_s = curr_gradient
          .element_multiply(&curr_gradient)
          .scalar_multiply_inplace(1.0 - self.beta2);
        self.d_s = Some(d_s.clone());
        d_s.element_sqrt().scalar_add_inplace(self.epsilon)
      }
    };

    // Correct gradients to have similar magnitude
    let corrected_momentum =
      momentum_gradient.scalar_multiply(1.0 / (1.0 - self.beta1.powf(self.t as f32)));
    let corrected_rms_prop =
      rms_prop_gradient.scalar_multiply(1.0 / (1.0 - self.beta2.powf(self.t as f32)));
    self.t += 1;

    // adjusted_gradient = momentum_gradient / rms_prop_gradient
    let adjusted_gradient = corrected_momentum.element_divide(&corrected_rms_prop);

    return adjusted_gradient.scalar_multiply_inplace(self.learning_rate);
  }

  fn clone_box(&self) -> Box<dyn Optimizer> {
    Box::new(self.clone())
  }
}
