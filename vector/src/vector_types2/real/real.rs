use RealNumber;
use multicore_support::*;
use simd_extensions::*;
use super::super::{
    Vector, MetaData,
    DspVec, ToSliceMut,
    Domain, RealNumberSpace,
};

/// Operations on real types.
///
/// # Failures
///
/// If one of the methods is called on complex data then `self.len()` will be set to `0`.
/// To avoid this it's recommended to use the `to_real_time_vec`, `to_real_freq_vec`
/// `to_complex_time_vec` and `to_complex_freq_vec` constructor methods since
/// the resulting types will already check at compile time (using the type system) that the data is real.
pub trait RealOps {
	/// Gets the absolute value of all vector elements.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let mut vector = vec!(1.0, -2.0).to_real_time_vec();
    /// vector.abs();
    /// assert_eq!([1.0, 2.0], vector[..]);
    /// ```
    fn abs(&mut self);
}

/// Operations on real types.
///
/// # Failures
///
/// If one of the methods is called on complex data then `self.len()` will be set to `0`.
/// To avoid this it's recommended to use the `to_real_time_vec`, `to_real_freq_vec`
/// `to_complex_time_vec` and `to_complex_freq_vec` constructor methods since
/// the resulting types will already check at compile time (using the type system) that the data is real.
pub trait ModuloOps<T>
    where T: RealNumber {
	/// Each value in the vector is dividable by the divisor and the remainder is stored in the resulting
    /// vector. This the same a modulo operation or to phase wrapping.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).to_real_time_vec();
    /// vector.wrap(4.0);
    /// assert_eq!([1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0], vector[..]);
    /// ```
    fn wrap(&mut self, divisor: T);

    /// This function corrects the jumps in the given vector which occur due to wrap or modulo operations.
    /// This will undo a wrap operation only if the deltas are smaller than half the divisor.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.).to_real_time_vec();
    /// vector.unwrap(4.0);
    /// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vector[..]);
    /// ```
    fn unwrap(&mut self, divisor: T);
}

macro_rules! assert_real {
    ($self_: ident) => {
        if $self_.is_complex() {
            $self_.valid_len = 0;
            return;
        }
    }
}

impl<S, T, N, D> RealOps for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain {
      fn abs(&mut self) {
          assert_real!(self);
          self.simd_real_operation(|x, _arg| (x * x).sqrt(), |x, _arg| x.abs(), (), Complexity::Small);
      }
  }

impl<S, T, N, D> ModuloOps<T> for DspVec<S, T, N, D>
      where S: ToSliceMut<T>,
            T: RealNumber,
            N: RealNumberSpace,
            D: Domain {

      fn wrap(&mut self, divisor: T) {
          assert_real!(self);
          self.pure_real_operation(|x, y| x % y, divisor, Complexity::Small);
      }

      fn unwrap(&mut self, divisor: T) {
          assert_real!(self);
          let data_length = self.len();
          let mut data = self.data.to_slice_mut();
          let mut i = 0;
          let mut j = 1;
          let half = divisor / T::from(2.0).unwrap();
          while j < data_length {
              let mut diff = data[j] - data[i];
              if diff > half {
                  diff = diff % divisor;
                  diff = diff - divisor;
                  data[j] = data[i] + diff;
              }
              else if diff < -half {
                  diff = diff % divisor;
                  diff = diff + divisor;
                  data[j] = data[i] + diff;
              }

              i += 1;
              j += 1;
          }
      }
}
