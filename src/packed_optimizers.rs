use itertools::Itertools;
use tensor_lib::{
  cuda_bindings::{cuda_adam_optimizer_packed, cuda_scalar_multiply},
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
    let adjusted_gradient =
      scalar_multiply_packed(curr_gradients, self.learning_rate * normalization_factor);
    return adjusted_gradient;
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
    // dW = (beta * prev_grad + (1 - beta) * curr_gradient)

    if (self.prev_gradients.is_none()) {
      self.prev_gradients = Some(scalar_multiply_packed(curr_gradient, 1.0));
    } else {
      let scaled_prev_gradient =
        scalar_multiply_packed(self.prev_gradients.as_ref().unwrap(), self.beta);
      let scaled_curr_gradient = scalar_multiply_packed(curr_gradient, 1.0 - self.beta);
      element_add_packed_inplace(&scaled_prev_gradient, &scaled_curr_gradient);
    }

    return scalar_multiply_packed(
      &self.prev_gradients.as_ref().unwrap(),
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

// #[derive(Clone)]
// pub struct PackedAdagradOptimizer {
//   learning_rate: f32,
//   epsilon: f32,
//   accumulated_gradients: Option<Vec<Matrix>>,
// }

// impl PackedAdagradOptimizer {
//   pub fn new(learning_rate: f32) -> PackedAdagradOptimizer {
//     PackedAdagradOptimizer {
//       learning_rate,
//       epsilon: 1e-8,
//       accumulated_gradients: None,
//     }
//   }
// }

// impl PackedOptimizer for PackedAdagradOptimizer {
//   fn calculate_steps(
//     &mut self,
//     curr_gradients: &[Matrix], normalization_factor: f32,
//     normalization_factor: f32,
//   ) -> Vec<Matrix> {
//     let adjusted_gradients = match &mut self.accumulated_gradients {
//       Some(accumulated_gradients) => {
//         // accumulated_gradient = accumulated_gradient + curr_gradient^2
//         // adjusted_gradient = curr_gradient / (sqrt(accumulated_gradient) + epsilon)
//         // The idea is the 1/sqrt(accumulated_gradient) will act as a learning rate decay, so we can more smartly update per weight
//         // Otherwise we can see this is the same as SGD (no momentum involved here)

//         let curr_gradients_squared = element_multiply_packed(curr_gradients, curr_gradients);
//         element_add_packed_inplace(&accumulated_gradients, &curr_gradients_squared);

//         let accumulated_gradient_sqrt = element_sqrt_packed(&accumulated_gradients);
//         scalar_add_packed_inplace(&accumulated_gradient_sqrt, self.epsilon);

//         let adjusted_gradients = element_divide_packed(curr_gradients, &accumulated_gradient_sqrt);

//         adjusted_gradients
//       }
//       None => {
//         self.accumulated_gradients = Some(element_multiply_packed(curr_gradients, curr_gradients));
//         curr_gradients
//           .iter()
//           .map(|grad| grad.deep_copy())
//           .collect_vec()
//       }
//     };

//     scalar_multiply_packed_inplace(
//       &adjusted_gradients,
//       self.learning_rate * normalization_factor,
//     );
//     return adjusted_gradients;
//   }
//   fn get_single_optimizer(&self) -> Box<dyn Optimizer> {
//     Box::new(AdagradOptimizer::new(self.learning_rate))
//   }
//   fn clone_box(&self) -> Box<dyn PackedOptimizer> {
//     Box::new(self.clone())
//   }
// }

// #[derive(Clone)]
// pub struct PackedRMSPropOptimizer {
//   learning_rate: f32,
//   beta: f32,
//   epsilon: f32,
//   accumulated_gradients: Option<Vec<Matrix>>,
// }

// impl PackedRMSPropOptimizer {
//   pub fn new(learning_rate: f32, beta: f32) -> PackedRMSPropOptimizer {
//     PackedRMSPropOptimizer {
//       learning_rate,
//       beta,
//       epsilon: 1e-8,
//       accumulated_gradients: None,
//     }
//   }
// }

// impl PackedOptimizer for PackedRMSPropOptimizer {
//   fn calculate_steps(
//     &mut self,
//     curr_gradients: &[Matrix], normalization_factor: f32,
//     normalization_factor: f32,
//   ) -> Vec<Matrix> {
//     let adjusted_gradients = match &mut self.accumulated_gradients {
//       Some(prev_accumulated_gradients) => {
//         // adjusted_gradient = curr_gradient/sqrt(accumulated_gradient)
//         // accumulated_gradient = (beta * prev_accumulated_gradient + (1 - beta) * curr_gradient ^ 2)

//         // The idea expands on Adagrad, but adds a decay factor to the accumulated gradient
//         // This means that the learning rate will decay over time, but not as aggressively as Adagrad
//         // This is because older weights are continually decayed by beta

//         scalar_multiply_packed_inplace(&prev_accumulated_gradients, self.beta);
//         let curr_grad_squared = element_multiply_packed(curr_gradients, curr_gradients);
//         scalar_multiply_packed_inplace(&curr_grad_squared, 1.0 - self.beta);
//         element_add_packed_inplace(&prev_accumulated_gradients, &curr_grad_squared);

//         let sqrt_accumulated_gradient = element_sqrt_packed(&prev_accumulated_gradients);
//         scalar_add_packed_inplace(&sqrt_accumulated_gradient, self.epsilon);
//         let adjusted_gradients = element_divide_packed(curr_gradients, &sqrt_accumulated_gradient);
//         adjusted_gradients
//       }
//       None => {
//         let curr_grad_squared = element_multiply_packed(curr_gradients, curr_gradients);
//         scalar_multiply_packed_inplace(&curr_grad_squared, 1.0 - self.beta);

//         self.accumulated_gradients = Some(curr_grad_squared.clone());

//         let sqrt_accumulated = element_sqrt_packed(&curr_grad_squared);
//         scalar_add_packed_inplace(&sqrt_accumulated, self.epsilon);
//         let adjusted_gradients = element_divide_packed(curr_gradients, &sqrt_accumulated);
//         adjusted_gradients
//       }
//     };

//     scalar_multiply_packed_inplace(
//       &adjusted_gradients,
//       self.learning_rate * normalization_factor,
//     );

//     return adjusted_gradients;
//   }

//   fn get_single_optimizer(&self) -> Box<dyn Optimizer> {
//     Box::new(RMSPropOptimizer::new(self.learning_rate, self.beta))
//   }

//   fn clone_box(&self) -> Box<dyn PackedOptimizer> {
//     Box::new(self.clone())
//   }
// }

#[derive(Clone)]
pub struct PackedAdamOptimizer {
  learning_rate: f32,
  beta1: f32,
  beta2: f32,
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
    let matrix_count = curr_gradients.len();
    if (self.d_s.is_none()) {
      self.d_s = Some(
        (0..matrix_count)
          .map(|_| {
            Matrix::zeros(
              curr_gradients[0].get_rows(),
              curr_gradients[0].get_columns(),
            )
          })
          .collect_vec(),
      );
      self.d_v = Some(
        (0..matrix_count)
          .map(|_| {
            Matrix::zeros(
              curr_gradients[0].get_rows(),
              curr_gradients[0].get_columns(),
            )
          })
          .collect_vec(),
      )
    }

    let mut result_gradients = Vec::with_capacity(matrix_count);

    // Dip into unsafe because there isi a specialized cuda function for this
    unsafe {
      result_gradients.set_len(matrix_count);
      cuda_adam_optimizer_packed(
        self.d_v.as_ref().unwrap().as_ptr() as *const Matrix,
        self.d_s.as_ref().unwrap().as_ptr() as *const Matrix,
        curr_gradients.as_ptr() as *const Matrix,
        result_gradients.as_ptr() as *mut Matrix,
        matrix_count,
        self.beta1,
        self.beta2,
        self.learning_rate * normalization_factor,
      )
    }

    return result_gradients;
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
