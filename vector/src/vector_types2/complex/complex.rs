pub trait ComplexOps {
	/// Multiplies each vector element with `exp(j*(a*idx*self.delta() + b))`
    /// where `a` and `b` are arguments and `idx` is the index of the data points
    /// in the vector ranging from `0 to self.points() - 1`. `j` is the imaginary number and
    /// `exp` the exponential function.
    ///
    /// This method can be used to perform a frequency shift in time domain.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{ComplexTimeVector32, ComplexVectorOps, DataVec, InterleavedIndex};
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.multiply_complex_exponential(2.0, 3.0).expect("Ignoring error handling in examples");
    /// let expected = [-1.2722325, -1.838865, 4.6866837, -1.7421241];
    /// let result = result.interleaved(0..);
    /// for i in 0..expected.len() {
    ///     assert!((result[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn multiply_complex_exponential(mut self, a: T, b: T) -> TransRes<Self>;

	/// Calculates the complex conjugate of the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::{ComplexTimeVector32, ComplexVectorOps, DataVec, InterleavedIndex};
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.conj().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, -2.0, 3.0, -4.0], result.interleaved(0..));
    /// # }
    /// ```
    fn conj(self) -> TransRes<Self>;
}
