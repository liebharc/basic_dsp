pub trait StatisticsOps<T> : Sized
    where T: Sized {
    /// Calculates the statistics of the data contained in the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex32;
    /// use basic_dsp_vector::{ComplexTimeVector32, StatisticsOps};
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let result = vector.statistics();
    /// assert_eq!(result.sum, Complex32::new(9.0, 12.0));
    /// assert_eq!(result.count, 3);
    /// assert_eq!(result.average, Complex32::new(3.0, 4.0));
    /// assert!((result.rms - Complex32::new(3.4027193, 4.3102784)).norm() < 1e-4);
    /// assert_eq!(result.min, Complex32::new(1.0, 2.0));
    /// assert_eq!(result.min_index, 0);
    /// assert_eq!(result.max, Complex32::new(5.0, 6.0));
    /// assert_eq!(result.max_index, 2);
    /// }
    /// ```
    fn statistics(&self) -> Statistics<T>;

    /// Calculates the statistics of the data contained in the vector as if the vector would
    /// have been split into `len` pieces. `self.len` should be dividable by `len` without a remainder,
    /// but this isn't enforced by the implementation.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex32;
    /// use basic_dsp_vector::{ComplexTimeVector32, StatisticsOps};
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// let result = vector.statistics_splitted(2);
    /// assert_eq!(result[0].sum, Complex32::new(6.0, 8.0));
    /// assert_eq!(result[1].sum, Complex32::new(10.0, 12.0));
    /// }
    /// ```
    fn statistics_splitted(&self, len: usize) -> Vec<Statistics<T>>;

    /// Calculates the sum of the data contained in the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex32;
    /// use basic_dsp_vector::{ComplexTimeVector32, StatisticsOps};
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let result = vector.sum();
    /// assert_eq!(result, Complex32::new(9.0, 12.0));
    /// }
    /// ```
    fn sum(&self) -> T;

    /// Calculates the sum of the squared data contained in the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex32;
    /// use basic_dsp_vector::{ComplexTimeVector32, StatisticsOps};
    /// # fn main() {
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let result = vector.sum_sq();
    /// assert_eq!(result, Complex32::new(-21.0, 88.0));
    /// }
    /// ```
    fn sum_sq(&self) -> T;
}
