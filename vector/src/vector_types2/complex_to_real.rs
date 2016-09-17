use RealNumber;
use multicore_support::Complexity;
use simd_extensions::Simd;
use super::{
    TransRes, ErrorReason,
    Owner, ToRealResult,
    Buffer,
    DspVec, ToSliceMut,
    Domain, ComplexNumberSpace, RededicateForceOps
};

/// Defines all operations which are valid on complex data and result in real data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeComplex` if the vector isn't in the complex number space.
pub trait ComplexToRealTransformsOps<S, T, B> : ToRealResult
    where S: ToSliceMut<T>,
          T: RealNumber,
          B: Buffer<S, T> {

    /// Gets the absolute value, magnitude or norm of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let vector = vec!(3.0, -4.0, -3.0, 4.0).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.magnitude(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([5.0, 5.0], result[0..]);
    /// # }
    /// ```
    fn magnitude(self, buffer: &mut B) -> TransRes<Self::RealResult>;

    /// Gets the square root of the absolute value of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let vector = vec!(3.0, -4.0, -3.0, 4.0).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.magnitude_squared(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([25.0, 25.0], result[0..]);
    /// # }
    /// ```
    fn magnitude_squared(self, buffer: &mut B) -> TransRes<Self::RealResult>;

    /// Gets all real elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.to_real(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 3.0], result[0..]);
    /// # }
    /// ```
    fn to_real(self, buffer: &mut B) -> TransRes<Self::RealResult>;

    /// Gets all imag elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.to_imag(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 4.0], result[0..]);
    /// # }
    /// ```
    fn to_imag(self, buffer: &mut B) -> TransRes<Self::RealResult>;

    /// Gets the phase of all elements in [rad].
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::vector_types2::*;
    /// # fn main() {
    /// let data: Vec<f32> = vec!(1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0);
    /// let vector = data.to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.phase(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982], result[0..]);
    /// # }
    /// ```
    fn phase(self, buffer: &mut B) -> TransRes<Self::RealResult>;
}

macro_rules! assert_complex {
    ($self_: ident) => {
        if !$self_.is_complex() {
            $self_.number_space.to_real();
            $self_.valid_len = 0;
            return Err((ErrorReason::InputMustBeComplex, Self::RealResult::rededicate_from_force($self_)));
        }
    }
}

impl<S, T, N, D, B> ComplexToRealTransformsOps<S, T, B> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToRealResult,
          <DspVec<S, T, N, D> as ToRealResult>::RealResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain,
          B: Buffer<S, T> {
    fn magnitude(mut self, buffer: &mut B) -> TransRes<Self::RealResult> {
        assert_complex!(self);
        self.simd_complex_to_real_operation(buffer, |x,_arg| x.complex_abs(), |x,_arg| x.norm(), (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }

    fn magnitude_squared(mut self, buffer: &mut B) -> TransRes<Self::RealResult> {
        assert_complex!(self);
        self.simd_complex_to_real_operation(buffer, |x,_arg| x.complex_abs_squared(), |x,_arg| x.re * x.re + x.im * x.im, (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }

    fn to_real(mut self, buffer: &mut B) -> TransRes<Self::RealResult> {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x,_arg|x.re, (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }

    fn to_imag(mut self, buffer: &mut B) -> TransRes<Self::RealResult> {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x,_arg|x.im, (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }

    fn phase(mut self, buffer: &mut B) -> TransRes<Self::RealResult> {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x,_arg|x.arg(), (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }
}
