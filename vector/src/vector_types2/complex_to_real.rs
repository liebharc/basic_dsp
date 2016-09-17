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
pub trait ComplexToRealTransformsOps<S, T> : ToRealResult
    where S: ToSliceMut<T>,
          T: RealNumber {

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
    fn magnitude<B>(self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T>;

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
    fn magnitude_squared<B>(self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T>;

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
    fn to_real<B>(self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T>;

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
    fn to_imag<B>(self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T>;

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
    fn phase<B>(self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T>;
}

pub trait ComplexToRealGetterSetterOps {
    /// Copies all real elements into the given vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOps, RealIndex};
    /// # fn main() {
    /// let mut result = RealTimeVector32::from_array(&[0.0, 0.0]);
    /// let vector = ComplexTimeVector32::from_real_imag(&[1.0, 3.0], &[2.0, 4.0]);
    /// vector.get_real(&mut result).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 3.0], result.real(0..));
    /// # }
    /// ```
    fn get_real(&self, destination: &mut Self::RealPartner) -> VoidResult;

    /// Copies all imag elements into the given vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOps, DataVec, RealIndex};
    /// # fn main() {
    /// let mut result = RealTimeVector32::from_array(&[0.0, 0.0]);
    /// let vector = ComplexTimeVector32::from_real_imag(&[1.0, 3.0], &[2.0, 4.0]);
    /// vector.get_imag(&mut result).expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 4.0], result.real(0..));
    /// # }
    /// ```
    fn get_imag(&self, destination: &mut Self::RealPartner) -> VoidResult;

    /// Copies the phase of all elements in [rad] into the given vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOps, DataVec, RealIndex};
    /// # fn main() {
    /// let mut result = RealTimeVector32::from_array(&[0.0, 0.0]);
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0]);
    /// vector.get_phase(&mut result).expect("Ignoring error handling in examples");
    /// assert_eq!([0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982], result.real(0..));
    /// # }
    /// ```
    fn get_phase(&self, destination: &mut Self::RealPartner) -> VoidResult;

    /// Gets the real and imaginary parts and stores them in the given vectors.
    /// See also  [`get_phase`](trait.ComplexVectorOps.html#tymethod.get_phase) and
    /// [`get_complex_abs`](trait.ComplexVectorOps.html#tymethod.get_complex_abs) for further
    /// information.
    fn get_real_imag(&self, real: &mut Self::RealPartner, imag: &mut Self::RealPartner) -> VoidResult;

    /// Gets the magnitude and phase and stores them in the given vectors.
    /// See also [`get_real`](trait.ComplexVectorOps.html#tymethod.get_real) and
    /// [`get_imag`](trait.ComplexVectorOps.html#tymethod.get_imag) for further
    /// information.
    fn get_mag_phase(&self, mag: &mut Self::RealPartner, phase: &mut Self::RealPartner) -> VoidResult;

    /// Overrides the `self` vectors data with the real and imaginary data in the given vectors.
    /// `real` and `imag` must have the same size.
    fn set_real_imag(self, real: &Self::RealPartner, imag: &Self::RealPartner) -> TransRes<Self>;

    /// Overrides the `self` vectors data with the magnitude and phase data in the given vectors.
    /// Note that `self` vector will immediately convert the data into a real and imaginary representation
    /// of the complex numbers which is its default format.
    /// `mag` and `phase` must have the same size.
    fn set_mag_phase(self, mag: &Self::RealPartner, phase: &Self::RealPartner) -> TransRes<Self>;
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

impl<S, T, N, D> ComplexToRealTransformsOps<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToRealResult,
          <DspVec<S, T, N, D> as ToRealResult>::RealResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain {
    fn magnitude<B>(mut self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T> {
        assert_complex!(self);
        self.simd_complex_to_real_operation(buffer, |x,_arg| x.complex_abs(), |x,_arg| x.norm(), (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }

    fn magnitude_squared<B>(mut self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T> {
        assert_complex!(self);
        self.simd_complex_to_real_operation(buffer, |x,_arg| x.complex_abs_squared(), |x,_arg| x.re * x.re + x.im * x.im, (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }

    fn to_real<B>(mut self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T> {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x,_arg|x.re, (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }

    fn to_imag<B>(mut self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T> {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x,_arg|x.im, (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }

    fn phase<B>(mut self, buffer: &mut B) -> TransRes<Self::RealResult>
        where B: Buffer<S, T> {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x,_arg|x.arg(), (), Complexity::Small);
        self.number_space.to_real();
        Ok(Self::RealResult::rededicate_from_force(self))
    }
}
