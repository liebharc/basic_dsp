use RealNumber;
use super::super::{
    Owner, ToComplexResult,
    Buffer, Vector, InsertZerosOps,
    DspVec, ToSliceMut,
    Domain, RealNumberSpace, RededicateForceOps
};

/// Defines transformations from real to complex number space.
///
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the type isn't in the real number space.
pub trait RealToComplexTransformsOps<S, T> : ToComplexResult
    where S: ToSliceMut<T>,
          T: RealNumber {
	/// Converts the real vector into a complex vector.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let vector = vec!(1.0, 2.0).to_real_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.to_complex(&mut buffer);
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], result[..]);
    /// ```
    fn to_complex<B>(self, buffer: &mut B) -> Self::ComplexResult
        where B: Buffer<S, T>;
}

macro_rules! assert_real {
    ($self_: ident) => {
        if $self_.is_complex() {
            $self_.number_space.to_complex();
            $self_.valid_len = 0;
            return Self::ComplexResult::rededicate_from_force($self_);
        }
    }
}

impl<S, T, N, D> RealToComplexTransformsOps<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToComplexResult + InsertZerosOps<S, T>,
          <DspVec<S, T, N, D> as ToComplexResult>::ComplexResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain {
    fn to_complex<B>(mut self, buffer: &mut B) -> Self::ComplexResult
        where B: Buffer<S, T> {
        assert_real!(self);
        self.zero_interleave(buffer, 2);
        self.number_space.to_complex();
        Self::ComplexResult::rededicate_from_force(self)
    }
}
