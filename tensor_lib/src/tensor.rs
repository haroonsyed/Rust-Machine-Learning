use crate::matrix::*;

pub struct Tensor {
  pub dimensions: Vec<usize>,
  child_tensors: Box<Vec<Tensor>>, // This goes all the way down to either a matrix or a vector. Meaning rank 2 or rank 1.
  data: Option<Matrix>, // For 1d tensors, it will still be a matrix, but with only one row/column.
}

/*
TODO:
1. Flatten
2. Dot Product
3. Indexing
4. Transpose?
5. Reshape?
6. Max Pooling?
*/

impl Tensor {
  pub fn get_rank(&self) -> usize {
    return self.dimensions.len();
  }

  pub fn is_leaf(&self) -> bool {
    return self.data.is_some();
  }

  pub fn is_matrix(&self) -> bool {
    return self.dimensions.len() == 2;
  }

  pub fn is_vector(&self) -> bool {
    return self.dimensions.len() == 1;
  }

  pub fn get_data(&self) -> &Matrix {
    return self.data.as_ref().unwrap();
  }

  pub fn get_data_length(&self) -> usize {
    if self.dimensions.len() == 0 {
      return 0;
    };

    let mut result = 1;
    for dimension in self.dimensions.iter() {
      result *= dimension;
    }
    return result;
  }

  pub fn dimensions_are_the_same(&self, other: &Tensor) -> bool {
    for (this, other) in self.dimensions.iter().zip(other.dimensions.iter()) {
      if this != other {
        return false;
      }
    }
    return true;
  }

  // Utils for tensor operations
  fn element_wise_op<F>(self, op: F) -> Tensor
  where
    F: Fn(Matrix) -> Matrix,
  {
    // Define a helper function for applying the operation, apparently rust hates closures with recursion https://stackoverflow.com/a/72862424/10085824
    fn apply_operation<F>(this: Tensor, op: &F) -> Tensor
    where
      F: Fn(Matrix) -> Matrix,
    {
      if this.is_leaf() {
        return Tensor {
          dimensions: this.dimensions.clone(),
          child_tensors: Box::new(Vec::new()),
          data: Some(op(this.data.unwrap())),
        };
      }

      let mut result = Tensor {
        dimensions: this.dimensions.clone(),
        child_tensors: Box::new(Vec::new()),
        data: None,
      };

      for child in this.child_tensors.into_iter() {
        result.child_tensors.push(apply_operation(child, op));
      }

      result
    }

    // Call the helper function with the closure
    apply_operation(self, &op)
  }

  fn between_tensor_element_wise_op<F>(self, other: &Tensor, op: F) -> Tensor
  where
    F: Fn(Matrix, &Matrix) -> Matrix,
  {
    // Define a helper function for applying the operation, apparently rust hates closures with recursion https://stackoverflow.com/a/72862424/10085824
    fn apply_operation<F>(this: Tensor, other: &Tensor, op: &F) -> Tensor
    where
      F: Fn(Matrix, &Matrix) -> Matrix,
    {
      if !this.dimensions_are_the_same(other) {
        panic!("Dimensions of tensors are not the same");
      }

      // Base case
      if this.is_leaf() {
        return Tensor {
          dimensions: this.dimensions.clone(),
          child_tensors: Box::new(Vec::new()),
          data: Some(op(this.data.unwrap(), other.get_data())),
        };
      }

      // Recursive case
      let mut result = Tensor {
        dimensions: this.dimensions.clone(),
        child_tensors: Box::new(Vec::new()),
        data: None,
      };

      for (this, other) in this
        .child_tensors
        .into_iter()
        .zip(other.child_tensors.iter())
      {
        result.child_tensors.push(apply_operation(this, other, op));
      }

      return result;
    }
    // Call the helper function with the closure
    apply_operation(self, other, &op)
  }

  // The operations implemented will only be between tensors.
  // Doing anything on self is strange, because it's not clear what the dimensions of the result should be.
  pub fn element_add(self, other: &Tensor) -> Tensor {
    return self.between_tensor_element_wise_op(other, |a, b| a.element_add(b));
  }

  pub fn element_add_inplace(self, other: &Tensor) {
    self.between_tensor_element_wise_op(other, |a, b| {
      a.element_add_inplace(b);
      return a;
    });
  }

  pub fn element_subtract(self, other: &Tensor) -> Tensor {
    return self.between_tensor_element_wise_op(other, |a, b| a.element_subtract(b));
  }

  pub fn element_subtract_inplace(self, other: &Tensor) {
    self.between_tensor_element_wise_op(other, |a, b| {
      a.element_subtract_inplace(b);
      return a;
    });
  }

  pub fn element_multiply(self, other: &Tensor) -> Tensor {
    return self.between_tensor_element_wise_op(other, |a, b| a.element_multiply(b));
  }

  pub fn element_multiply_inplace(self, other: &Tensor) {
    self.between_tensor_element_wise_op(other, |a, b| {
      a.element_multiply_inplace(b);
      return a;
    });
  }

  pub fn element_relu(self) -> Tensor {
    return self.element_wise_op(|a| a.element_ReLU());
  }

  pub fn element_relu_inplace(self) {
    self.element_wise_op(|a| {
      a.element_ReLU_inplace();
      return a;
    });
  }

  pub fn element_relu_prime(self) -> Tensor {
    return self.element_wise_op(|a| a.element_ReLU_prime());
  }

  pub fn element_relu_prime_inplace(self) {
    self.element_wise_op(|a| {
      a.element_ReLU_prime_inplace();
      return a;
    });
  }

  pub fn element_exp(self) -> Tensor {
    return self.element_wise_op(|a| a.element_exp());
  }

  pub fn element_exp_inplace(self) {
    self.element_wise_op(|a| {
      a.element_exp_inplace();
      return a;
    });
  }

  pub fn scalar_multiply(self, scalar: f32) -> Tensor {
    let func = |a: Matrix| a.scalar_multiply(scalar);
    return self.element_wise_op(func);
  }

  pub fn scalar_multiply_inplace(self, scalar: f32) {
    self.element_wise_op(|a| {
      a.scalar_multiply_inplace(scalar);
      return a;
    });
  }
}
