use itertools::Itertools;
use tensor_lib::{
  element_add_packed, element_divide_packed, element_multiply_packed, element_sqrt_packed,
  scalar_add_packed, scalar_multiply_packed, Matrix,
};

use crate::optimizers::{
  AdagradOptimizer, AdamOptimizer, MomentumOptimizer, Optimizer, RMSPropOptimizer,
  StochasticGradientDescentOptimizer,
};

pub trait PackedOptimizer: Send {
  fn calculate_steps(
    &mut self,
    curr_gradients: &Vec<Matrix>,
    normalization_factor: f32,
  ) -> Vec<Matrix>;
  fn get_single_optimizer(&self) -> Box<dyn Optimizer>;
  fn clone_box(&self) -> Box<dyn PackedOptimizer>;
}

impl Clone for Box<dyn PackedOptimizer> {
  fn clone(&self) -> Box<dyn PackedOptimizer> {
    self.clone_box()
  }
}

#[derive(Clone)]
pub struct PackedStochasticGradientDescentOptimizer {
  learning_rate: f32,
}

impl PackedStochasticGradientDescentOptimizer {
  pub fn new(learning_rate: f32) -> PackedStochasticGradientDescentOptimizer {
    PackedStochasticGradientDescentOptimizer { learning_rate }
  }
}

impl PackedOptimizer for PackedStochasticGradientDescentOptimizer {
  fn calculate_steps(
    &mut self,
    curr_gradients: &Vec<Matrix>,
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    return scalar_multiply_packed(
      curr_gradients,
      self.learning_rate * normalization_factor,
      false,
    );
  }
  fn get_single_optimizer(&self) -> Box<dyn Optimizer> {
    Box::new(StochasticGradientDescentOptimizer::new(self.learning_rate))
  }
  fn clone_box(&self) -> Box<dyn PackedOptimizer> {
    Box::new(self.clone())
  }
}

#[derive(Clone)]
pub struct PackedMomentumOptimizer {
  learning_rate: f32,
  beta: f32,
  prev_gradients: Option<Vec<Matrix>>,
}

impl PackedMomentumOptimizer {
  pub fn new(learning_rate: f32, beta: f32) -> PackedMomentumOptimizer {
    PackedMomentumOptimizer {
      learning_rate,
      beta,
      prev_gradients: None,
    }
  }
}

impl PackedOptimizer for PackedMomentumOptimizer {
  fn calculate_steps(
    &mut self,
    curr_gradient: &Vec<Matrix>,
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    let adjusted_gradients = match &self.prev_gradients {
      Some(prev_gradients) => {
        // dW = (beta * prev_grad + (1 - beta) * curr_gradient)

        scalar_multiply_packed(prev_gradients, self.beta, true);
        let scaled_curr_gradient = scalar_multiply_packed(curr_gradient, 1.0 - self.beta, false);
        element_add_packed(prev_gradients, &scaled_curr_gradient, true);

        prev_gradients.clone()
      }
      None => {
        let gradients = curr_gradient
          .iter()
          .map(|grad| grad.deep_copy())
          .collect_vec();
        self.prev_gradients = Some(gradients.clone());
        gradients
      }
    };

    return scalar_multiply_packed(
      &adjusted_gradients,
      self.learning_rate * normalization_factor,
      false,
    );
  }
  fn get_single_optimizer(&self) -> Box<dyn Optimizer> {
    Box::new(MomentumOptimizer::new(self.learning_rate, self.beta))
  }
  fn clone_box(&self) -> Box<dyn PackedOptimizer> {
    Box::new(self.clone())
  }
}

#[derive(Clone)]
pub struct PackedAdagradOptimizer {
  learning_rate: f32,
  epsilon: f32,
  accumulated_gradients: Option<Vec<Matrix>>,
}

impl PackedAdagradOptimizer {
  pub fn new(learning_rate: f32) -> PackedAdagradOptimizer {
    PackedAdagradOptimizer {
      learning_rate,
      epsilon: 1e-8,
      accumulated_gradients: None,
    }
  }
}

impl PackedOptimizer for PackedAdagradOptimizer {
  fn calculate_steps(
    &mut self,
    curr_gradients: &Vec<Matrix>,
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    let adjusted_gradients = match &mut self.accumulated_gradients {
      Some(accumulated_gradients) => {
        // accumulated_gradient = accumulated_gradient + curr_gradient^2
        // adjusted_gradient = curr_gradient / (sqrt(accumulated_gradient) + epsilon)
        // The idea is the 1/sqrt(accumulated_gradient) will act as a learning rate decay, so we can more smartly update per weight
        // Otherwise we can see this is the same as SGD (no momentum involved here)

        let curr_gradients_squared = element_multiply_packed(curr_gradients, curr_gradients, false);
        element_add_packed(&accumulated_gradients, &curr_gradients_squared, true);

        let accumulated_gradient_sqrt = element_sqrt_packed(&accumulated_gradients, false);
        scalar_add_packed(&accumulated_gradient_sqrt, self.epsilon, true);

        let adjusted_gradients =
          element_divide_packed(curr_gradients, &accumulated_gradient_sqrt, false);

        adjusted_gradients
      }
      None => {
        self.accumulated_gradients = Some(element_multiply_packed(
          curr_gradients,
          curr_gradients,
          false,
        ));
        curr_gradients
          .iter()
          .map(|grad| grad.deep_copy())
          .collect_vec()
      }
    };

    return scalar_multiply_packed(
      &adjusted_gradients,
      self.learning_rate * normalization_factor,
      true,
    );
  }
  fn get_single_optimizer(&self) -> Box<dyn Optimizer> {
    Box::new(AdagradOptimizer::new(self.learning_rate))
  }
  fn clone_box(&self) -> Box<dyn PackedOptimizer> {
    Box::new(self.clone())
  }
}

#[derive(Clone)]
pub struct PackedRMSPropOptimizer {
  learning_rate: f32,
  beta: f32,
  epsilon: f32,
  accumulated_gradients: Option<Vec<Matrix>>,
}

impl PackedRMSPropOptimizer {
  pub fn new(learning_rate: f32, beta: f32) -> PackedRMSPropOptimizer {
    PackedRMSPropOptimizer {
      learning_rate,
      beta,
      epsilon: 1e-8,
      accumulated_gradients: None,
    }
  }
}

impl PackedOptimizer for PackedRMSPropOptimizer {
  fn calculate_steps(
    &mut self,
    curr_gradients: &Vec<Matrix>,
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    let adjusted_gradients = match &mut self.accumulated_gradients {
      Some(prev_accumulated_gradients) => {
        // adjusted_gradient = curr_gradient/sqrt(accumulated_gradient)
        // accumulated_gradient = (beta * prev_accumulated_gradient + (1 - beta) * curr_gradient ^ 2)

        // The idea expands on Adagrad, but adds a decay factor to the accumulated gradient
        // This means that the learning rate will decay over time, but not as aggressively as Adagrad
        // This is because older weights are continually decayed by beta

        scalar_multiply_packed(&prev_accumulated_gradients, self.beta, true);
        let curr_grad_squared = element_multiply_packed(curr_gradients, curr_gradients, false);
        scalar_multiply_packed(&curr_grad_squared, 1.0 - self.beta, true);
        element_add_packed(&prev_accumulated_gradients, &curr_grad_squared, true);

        let sqrt_accumulated_gradient = element_sqrt_packed(&prev_accumulated_gradients, false);
        scalar_add_packed(&sqrt_accumulated_gradient, self.epsilon, true);
        let adjusted_gradients =
          element_divide_packed(curr_gradients, &sqrt_accumulated_gradient, false);
        adjusted_gradients
      }
      None => {
        let curr_grad_squared = element_multiply_packed(curr_gradients, curr_gradients, false);
        let times_one_minus_beta =
          scalar_multiply_packed(&curr_grad_squared, 1.0 - self.beta, true);

        self.accumulated_gradients = Some(times_one_minus_beta.clone());

        let sqrt_accumulated = element_sqrt_packed(&times_one_minus_beta, false);
        scalar_add_packed(&sqrt_accumulated, self.epsilon, true);
        let adjusted_gradients = element_divide_packed(curr_gradients, &sqrt_accumulated, false);
        adjusted_gradients
      }
    };

    return scalar_multiply_packed(
      &adjusted_gradients,
      self.learning_rate * normalization_factor,
      true,
    );
  }

  fn get_single_optimizer(&self) -> Box<dyn Optimizer> {
    Box::new(RMSPropOptimizer::new(self.learning_rate, self.beta))
  }

  fn clone_box(&self) -> Box<dyn PackedOptimizer> {
    Box::new(self.clone())
  }
}

#[derive(Clone)]
pub struct PackedAdamOptimizer {
  learning_rate: f32,
  beta1: f32,
  beta2: f32,
  epsilon: f32,
  t: usize,
  d_v: Option<Vec<Matrix>>,
  d_s: Option<Vec<Matrix>>,
}

impl PackedAdamOptimizer {
  pub fn new(learning_rate: f32, beta1: f32, beta2: f32) -> PackedAdamOptimizer {
    PackedAdamOptimizer {
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

impl PackedOptimizer for PackedAdamOptimizer {
  fn calculate_steps(
    &mut self,
    curr_gradients: &Vec<Matrix>,
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    let momentum_gradient = match &mut self.d_v {
      Some(prev_d_v) => {
        // dv = (beta * prev_grad + (1 - beta) * curr_gradient)
        scalar_multiply_packed(prev_d_v, self.beta1, true);
        let scaled_curr_gradient = scalar_multiply_packed(curr_gradients, 1.0 - self.beta1, false);
        element_add_packed(prev_d_v, &scaled_curr_gradient, true);

        prev_d_v.clone()
      }
      None => {
        let d_v = scalar_multiply_packed(curr_gradients, 1.0 - self.beta1, false);
        self.d_v = Some(d_v.clone());
        d_v
      }
    };

    let rms_prop_gradient = match &mut self.d_s {
      Some(prev_d_s) => {
        // adjusted_gradient = 1/sqrt(accumulated_gradient)
        // accumulated_gradient = (beta * prev_accumulated_gradient + (1 - beta) * curr_gradient ^ 2)

        scalar_multiply_packed(prev_d_s, self.beta2, true);
        let curr_curr_times_one_minus_beta =
          element_multiply_packed(curr_gradients, curr_gradients, false);
        scalar_multiply_packed(&curr_curr_times_one_minus_beta, 1.0 - self.beta2, true);
        element_add_packed(prev_d_s, &curr_curr_times_one_minus_beta, true);

        let prev_d_s_sqrt = element_sqrt_packed(prev_d_s, false);
        scalar_add_packed(&prev_d_s_sqrt, self.epsilon, true)
      }
      None => {
        let curr_curr_times_one_minus_beta =
          element_multiply_packed(curr_gradients, curr_gradients, false);
        scalar_multiply_packed(&curr_curr_times_one_minus_beta, 1.0 - self.beta2, true);
        self.d_s = Some(curr_curr_times_one_minus_beta.clone());

        let d_s_sqrt = element_sqrt_packed(&curr_curr_times_one_minus_beta, false);
        scalar_add_packed(&d_s_sqrt, self.epsilon, true)
      }
    };

    // Correct gradients to have similar magnitude
    let corrected_momentum = scalar_multiply_packed(
      &momentum_gradient,
      1.0 / (1.0 - self.beta1.powf(self.t as f32)),
      true,
    );

    let corrected_rms_prop = scalar_multiply_packed(
      &rms_prop_gradient,
      1.0 / (1.0 - self.beta2.powf(self.t as f32)),
      true,
    );
    self.t += 1;

    // adjusted_gradient = momentum_gradient / rms_prop_gradient
    let adjusted_gradient = element_divide_packed(&corrected_momentum, &corrected_rms_prop, true);

    return scalar_multiply_packed(
      &adjusted_gradient,
      self.learning_rate * normalization_factor,
      true,
    );
  }

  fn get_single_optimizer(&self) -> Box<dyn Optimizer> {
    Box::new(AdamOptimizer::new(
      self.learning_rate,
      self.beta1,
      self.beta2,
    ))
  }

  fn clone_box(&self) -> Box<dyn PackedOptimizer> {
    Box::new(self.clone())
  }
}
