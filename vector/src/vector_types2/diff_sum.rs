pub trait DiffSumOps {
	/// Calculates the delta of each elements to its previous element. This will decrease the vector length by one point.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{RealTimeVector32, GenericVectorOps, DataVec, RealIndex};
    /// let vector = RealTimeVector32::from_array(&[2.0, 3.0, 2.0, 6.0]);
    /// let result = vector.diff().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, -1.0, 4.0], result.real(0..));
    /// ```
    fn diff(self) -> TransRes<Self>;

    /// Calculates the delta of each elements to its previous element. The first element
    /// will remain unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{RealTimeVector32, GenericVectorOps, DataVec, RealIndex};
    /// let vector = RealTimeVector32::from_array(&[2.0, 3.0, 2.0, 6.0]);
    /// let result = vector.diff_with_start().expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 1.0, -1.0, 4.0], result.real(0..));
    /// ```
    fn diff_with_start(self) -> TransRes<Self>;

    /// Calculates the cumulative sum of all elements. This operation undoes the `diff_with_start`operation.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{RealTimeVector32, GenericVectorOps, DataVec, RealIndex};
    /// let vector = RealTimeVector32::from_array(&[2.0, 1.0, -1.0, 4.0]);
    /// let result = vector.cum_sum().expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 3.0, 2.0, 6.0], result.real(0..));
    /// ```
    fn cum_sum(self) -> TransRes<Self>;
}
