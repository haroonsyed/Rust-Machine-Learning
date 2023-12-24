use itertools::Itertools;
use tensor_lib::{
  element_add_packed_inplace, element_divide_packed, element_divide_packed_inplace,
  element_multiply_packed, element_sqrt_packed, scalar_add_packed_inplace, scalar_multiply_packed,
  scalar_multiply_packed_inplace, Matrix,
};

use crate::optimizers::{
  AdagradOptimizer, AdamOptimizer, MomentumOptimizer, Optimizer, RMSPropOptimizer,
  StochasticGradientDescentOptimizer,
};

pub trait PackedOptimizer: Send {
  fn calculate_steps(
    &mut self,
    curr_gradients: &[Matrix],
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
    curr_gradients: &[Matrix],
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    return scalar_multiply_packed(curr_gradients, self.learning_rate * normalization_factor);
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
    curr_gradient: &[Matrix],
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    let adjusted_gradients = match &self.prev_gradients {
      Some(prev_gradients) => {
        // dW = (beta * prev_grad + (1 - beta) * curr_gradient)

        scalar_multiply_packed_inplace(prev_gradients, self.beta);
        let scaled_curr_gradient = scalar_multiply_packed(curr_gradient, 1.0 - self.beta);
        element_add_packed_inplace(prev_gradients, &scaled_curr_gradient);

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
    curr_gradients: &[Matrix],
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    let adjusted_gradients = match &mut self.accumulated_gradients {
      Some(accumulated_gradients) => {
        // accumulated_gradient = accumulated_gradient + curr_gradient^2
        // adjusted_gradient = curr_gradient / (sqrt(accumulated_gradient) + epsilon)
        // The idea is the 1/sqrt(accumulated_gradient) will act as a learning rate decay, so we can more smartly update per weight
        // Otherwise we can see this is the same as SGD (no momentum involved here)

        let curr_gradients_squared = element_multiply_packed(curr_gradients, curr_gradients);
        element_add_packed_inplace(&accumulated_gradients, &curr_gradients_squared);

        let accumulated_gradient_sqrt = element_sqrt_packed(&accumulated_gradients);
        scalar_add_packed_inplace(&accumulated_gradient_sqrt, self.epsilon);

        let adjusted_gradients = element_divide_packed(curr_gradients, &accumulated_gradient_sqrt);

        adjusted_gradients
      }
      None => {
        self.accumulated_gradients = Some(element_multiply_packed(curr_gradients, curr_gradients));
        curr_gradients
          .iter()
          .map(|grad| grad.deep_copy())
          .collect_vec()
      }
    };

    scalar_multiply_packed_inplace(
      &adjusted_gradients,
      self.learning_rate * normalization_factor,
    );
    return adjusted_gradients;
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
    curr_gradients: &[Matrix],
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    let adjusted_gradients = match &mut self.accumulated_gradients {
      Some(prev_accumulated_gradients) => {
        // adjusted_gradient = curr_gradient/sqrt(accumulated_gradient)
        // accumulated_gradient = (beta * prev_accumulated_gradient + (1 - beta) * curr_gradient ^ 2)

        // The idea expands on Adagrad, but adds a decay factor to the accumulated gradient
        // This means that the learning rate will decay over time, but not as aggressively as Adagrad
        // This is because older weights are continually decayed by beta

        scalar_multiply_packed_inplace(&prev_accumulated_gradients, self.beta);
        let curr_grad_squared = element_multiply_packed(curr_gradients, curr_gradients);
        scalar_multiply_packed_inplace(&curr_grad_squared, 1.0 - self.beta);
        element_add_packed_inplace(&prev_accumulated_gradients, &curr_grad_squared);

        let sqrt_accumulated_gradient = element_sqrt_packed(&prev_accumulated_gradients);
        scalar_add_packed_inplace(&sqrt_accumulated_gradient, self.epsilon);
        let adjusted_gradients = element_divide_packed(curr_gradients, &sqrt_accumulated_gradient);
        adjusted_gradients
      }
      None => {
        let curr_grad_squared = element_multiply_packed(curr_gradients, curr_gradients);
        scalar_multiply_packed_inplace(&curr_grad_squared, 1.0 - self.beta);

        self.accumulated_gradients = Some(curr_grad_squared.clone());

        let sqrt_accumulated = element_sqrt_packed(&curr_grad_squared);
        scalar_add_packed_inplace(&sqrt_accumulated, self.epsilon);
        let adjusted_gradients = element_divide_packed(curr_gradients, &sqrt_accumulated);
        adjusted_gradients
      }
    };

    scalar_multiply_packed_inplace(
      &adjusted_gradients,
      self.learning_rate * normalization_factor,
    );

    return adjusted_gradients;
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
    curr_gradients: &[Matrix],
    normalization_factor: f32,
  ) -> Vec<Matrix> {
    let momentum_gradient = match &mut self.d_v {
      Some(prev_d_v) => {
        // dv = (beta * prev_grad + (1 - beta) * curr_gradient)
        scalar_multiply_packed_inplace(prev_d_v, self.beta1);
        let scaled_curr_gradient = scalar_multiply_packed(curr_gradients, 1.0 - self.beta1);
        element_add_packed_inplace(prev_d_v, &scaled_curr_gradient);

        prev_d_v.clone()
      }
      None => {
        let d_v = scalar_multiply_packed(curr_gradients, 1.0 - self.beta1);
        self.d_v = Some(d_v.clone());
        d_v
      }
    };

    let rms_prop_gradient = match &mut self.d_s {
      Some(prev_d_s) => {
        // adjusted_gradient = 1/sqrt(accumulated_gradient)
        // accumulated_gradient = (beta * prev_accumulated_gradient + (1 - beta) * curr_gradient ^ 2)

        scalar_multiply_packed_inplace(prev_d_s, self.beta2);
        let curr_curr_times_one_minus_beta =
          element_multiply_packed(curr_gradients, curr_gradients);
        scalar_multiply_packed_inplace(&curr_curr_times_one_minus_beta, 1.0 - self.beta2);
        element_add_packed_inplace(prev_d_s, &curr_curr_times_one_minus_beta);

        let prev_d_s_sqrt = element_sqrt_packed(prev_d_s);
        scalar_add_packed_inplace(&prev_d_s_sqrt, self.epsilon);
        prev_d_s_sqrt
      }
      None => {
        let curr_curr_times_one_minus_beta =
          element_multiply_packed(curr_gradients, curr_gradients);
        scalar_multiply_packed_inplace(&curr_curr_times_one_minus_beta, 1.0 - self.beta2);
        self.d_s = Some(curr_curr_times_one_minus_beta.clone());

        let d_s_sqrt = element_sqrt_packed(&curr_curr_times_one_minus_beta);
        scalar_add_packed_inplace(&d_s_sqrt, self.epsilon);
        d_s_sqrt
      }
    };

    // Correct gradients to have similar magnitude
    scalar_multiply_packed_inplace(
      &momentum_gradient,
      1.0 / (1.0 - self.beta1.powf(self.t as f32)),
    );

    scalar_multiply_packed_inplace(
      &rms_prop_gradient,
      1.0 / (1.0 - self.beta2.powf(self.t as f32)),
    );
    self.t += 1;

    // adjusted_gradient = momentum_gradient / rms_prop_gradient
    element_divide_packed_inplace(&momentum_gradient, &rms_prop_gradient);
    let adjusted_gradient = momentum_gradient;

    scalar_multiply_packed_inplace(
      &adjusted_gradient,
      self.learning_rate * normalization_factor,
    );
    return adjusted_gradient;
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
