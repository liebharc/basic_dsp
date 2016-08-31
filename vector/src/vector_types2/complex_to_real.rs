use super::{
    TransRes,
    ToRealResult
};

/// Defines all operations which are valid on complex data and result in real data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeComplex` if the vector isn't in the complex number space.
pub trait ComplexToRealTransformsOps : ToRealResult {

    /// Gets the absolute value, magnitude or norm of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{ComplexTimeVector32, ComplexVectorOps, DataVec, RealIndex};
    /// use num::complex::Complex32;
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[3.0, -4.0, -3.0, 4.0]);
    /// let result = vector.magnitude().expect("Ignoring error handling in examples");
    /// assert_eq!([5.0, 5.0], result.real(0..));
    /// # }
    /// ```
    fn magnitude(self) -> TransRes<Self::RealResult>;

    /// Gets the square root of the absolute value of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{ComplexTimeVector32, ComplexVectorOps, DataVec, RealIndex};
    /// use num::complex::Complex32;
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[3.0, -4.0, -3.0, 4.0]);
    /// let result = vector.magnitude_squared().expect("Ignoring error handling in examples");
    /// assert_eq!([25.0, 25.0], result.real(0..));
    /// # }
    /// ```
    fn magnitude_squared(self) -> TransRes<Self::RealResult>;

    /// Gets all real elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{ComplexTimeVector32, ComplexVectorOps, DataVec, RealIndex};
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.to_real().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 3.0], result.real(0..));
    /// # }
    /// ```
    fn to_real(self) -> TransRes<Self::RealResult>;

    /// Gets all imag elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{ComplexTimeVector32, ComplexVectorOps, DataVec, RealIndex};
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.to_imag().expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 4.0], result.real(0..));
    /// # }
    /// ```
    fn to_imag(self) -> TransRes<Self::RealResult>;

    /// Gets the phase of all elements in [rad].
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{ComplexTimeVector32, ComplexVectorOps, DataVec, RealIndex};
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0]);
    /// let result = vector.phase().expect("Ignoring error handling in examples");
    /// assert_eq!([0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982], result.real(0..));
    /// # }
    /// ```
    fn phase(self) -> TransRes<Self::RealResult>;
}
/*
impl<S, T, N, D> ComplexToRealTransformsOps for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToRealResult,
          S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain {
    fn magnitude(self) -> TransRes<Self::RealResult> {
        panic!("Panic")
    }
}*/
