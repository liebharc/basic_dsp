//! Fundamental math operations
use RealNumber;
use multicore_support::Complexity;
use simd_extensions::*;
use num::Complex;
use super::{
    ErrorReason, VoidResult,
    DspVec, ToSliceMut,
    Domain, RealNumberSpace, ComplexNumberSpace
};


/// An operation which multiplies each vector element with a constant
pub trait ScaleOps<T> : Sized
    where T: Sized {
    /// Multiplies the vector element with a scalar.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let mut vector = vec!(1.0, 2.0).to_real_time_vec();
    /// vector.scale(2.0).expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 4.0], vector[0..]);
    /// # }
    /// ```
    fn scale(&mut self, factor: T) -> VoidResult;
}

/// An operation which adds a constant to each vector element
pub trait OffsetOps<T> : Sized
    where T: Sized {
    /// Adds a scalar to each vector element.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let mut vector = vec!(1.0, 2.0).to_real_time_vec();
    /// vector.offset(2.0).expect("Ignoring error handling in examples");
    /// assert_eq!([3.0, 4.0], vector[0..]);
    /// # }
    /// ```
    fn offset(&mut self, offset: T) -> VoidResult;
}

macro_rules! assert_real {
    ($self_: ident) => {
        if $self_.is_complex() {
            return Err(ErrorReason::InputMustBeReal);
        }
    }
}

macro_rules! assert_complex {
    ($self_: ident) => {
        if !$self_.is_complex() {
            return Err(ErrorReason::InputMustBeComplex);
        }
    }
}

impl<S, T, N, D> OffsetOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain {
    fn  offset(&mut self, offset: T) -> VoidResult {
        assert_real!(self);
        self.simd_real_operation(|x, y| x.add_real(y), |x, y| x + y, offset, Complexity::Small);
        Ok(())
    }
}

impl<S, T, N, D> OffsetOps<Complex<T>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain {
    fn  offset(&mut self, offset: Complex<T>) -> VoidResult {
        assert_complex!(self);
        let vector_offset = T::Reg::from_complex(offset);
        self.simd_complex_operation(|x,y| x + y, |x,y| x + Complex::<T>::new(y.extract(0), y.extract(1)), vector_offset, Complexity::Small);
        Ok(())
    }
}

impl<S, T, N, D> ScaleOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain {
    fn  scale(&mut self, factor: T) -> VoidResult {
        assert_real!(self);
        self.simd_real_operation(|x, y| x.scale_real(y), |x, y| x * y, factor, Complexity::Small);
        Ok(())
    }
}

impl<S, T, N, D> ScaleOps<Complex<T>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain {
    fn  scale(&mut self, factor: Complex<T>) -> VoidResult {
        assert_complex!(self);
        self.simd_complex_operation(|x,y| x.scale_complex(y), |x,y| x * y, factor, Complexity::Small);
        Ok(())
    }
}
