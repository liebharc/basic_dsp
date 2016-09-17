pub trait RealToComplex {
	/// Converts the real vector into a complex vector.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{RealTimeVector32, RealVectorOps, DataVec, InterleavedIndex};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector.to_complex().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], result.interleaved(0..));
    /// ```
    fn to_complex(self) -> TransRes<Self::ComplexPartner>;
}
