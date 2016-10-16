use RealNumber;
use super::super::{Owner, ToComplexResult, TransRes, Buffer, InsertZerosOps,
                   InsertZerosOpsBuffered, DspVec, ToSliceMut, MetaData, Domain, RealNumberSpace,
                   RededicateForceOps, ErrorReason};

/// Defines transformations from real to complex number space.
///
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the type isn't in the real number space.
pub trait RealToComplexTransformsOps<T>: ToComplexResult
    where T: RealNumber
{
    /// Converts the real vector into a complex vector.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let vector = vec!(1.0, 2.0).to_real_time_vec();
    /// let result = vector.to_complex().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], result[..]);
    /// ```
    fn to_complex(self) -> TransRes<Self::ComplexResult>;
}

/// Defines transformations from real to complex number space.
///
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the type isn't in the real number space.
pub trait RealToComplexTransformsOpsBuffered<S, T>: ToComplexResult
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Converts the real vector into a complex vector. The buffer allows
    /// this operation to succeed even if the storage type doesn't allow resizing.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let vector = vec!(1.0, 2.0).to_real_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.to_complex_b(&mut buffer);
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], result[..]);
    /// ```
    fn to_complex_b<B>(self, buffer: &mut B) -> Self::ComplexResult where B: Buffer<S, T>;
}

impl<S, T, N, D> RealToComplexTransformsOps<T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToComplexResult + InsertZerosOps<T>,
          <DspVec<S, T, N, D> as ToComplexResult>::ComplexResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain {
      fn to_complex(mut self) -> TransRes<Self::ComplexResult> {
          if self.is_complex() {
              self.number_space.to_complex();
              return Err((ErrorReason::InputMustBeReal, Self::ComplexResult::rededicate_from_force(self)));
          }

          let result = self.zero_interleave(2);
          let domain = self.domain();
          match result {
              Err(reason) => Err((reason, Self::ComplexResult::rededicate_with_runtime_data(self, true, domain))),
              Ok(()) =>  Ok(Self::ComplexResult::rededicate_with_runtime_data(self, true, domain))
          }
      }
}

impl<S, T, N, D> RealToComplexTransformsOpsBuffered<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToComplexResult + InsertZerosOpsBuffered<S, T>,
          <DspVec<S, T, N, D> as ToComplexResult>::ComplexResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain {
      fn to_complex_b<B>(mut self, buffer: &mut B) -> Self::ComplexResult
          where B: Buffer<S, T> {
          if self.is_complex() {
              self.number_space.to_complex();
              self.valid_len = 0;
              return Self::ComplexResult::rededicate_from_force(self);
          }
          self.zero_interleave_b(buffer, 2);
          self.number_space.to_complex();
          Self::ComplexResult::rededicate_from_force(self)
      }
}
