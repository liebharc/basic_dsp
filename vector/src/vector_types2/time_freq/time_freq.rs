/*
/// Defines all operations which are valid on `DataVecs` containing time domain data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeInTimeDomain` if the vector isn't in time domain.
pub trait TimeDomainOperations<T> : DataVec<T>
    where T : RealNumber {
    type FreqResult;

    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    ///
    /// This version of the FFT neither applies a window nor does it scale the
    /// vector.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{ComplexTimeVector32, TimeDomainOperations, DataVec, InterleavedIndex};
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254]);
    /// let result = vector.plain_fft().expect("Ignoring error handling in examples");
    /// let actual = result.interleaved(0..);
    /// let expected = &[0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn plain_fft(self) -> TransRes<Self::FreqResult>;

    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    /// # Unstable
    /// FFTs of real vectors are unstable.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{ComplexTimeVector32, TimeDomainOperations, DataVec, InterleavedIndex};
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254]);
    /// let result = vector.fft().expect("Ignoring error handling in examples");
    /// let actual = result.interleaved(0..);
    /// let expected = &[0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn fft(self) -> TransRes<Self::FreqResult>;

    /// Applies a FFT window and performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    fn windowed_fft(self, window: &WindowFunction<T>) -> TransRes<Self::FreqResult>;

    /// Applies a window to the data vector.
    fn apply_window(self, window: &WindowFunction<T>) -> TransRes<Self>;

    /// Removes a window from the data vector.
    fn unapply_window(self, window: &WindowFunction<T>) -> TransRes<Self>;
}

/// Defines all operations which are valid on `DataVecs` containing real time domain data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeInTimeDomain` if the vector isn't in time domain or
/// with `VectorMustHaveAnOddLength` if `self.points()` isn't and odd number.
pub trait SymmetricTimeDomainOperations<T> : DataVec<T>
    where T : RealNumber {
    type FreqResult;

    /// Performs a Symmetric Fast Fourier Transformation under the assumption that `self`
    /// is symmetric around the center. This assumption
    /// isn't verified and no error is raised if the vector isn't symmetric.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the
    /// vector.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeReal`: if `self` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    ///
    /// # Unstable
    /// Symmetric IFFTs are unstable and may only work under certain conditions.
    fn plain_sfft(self) -> TransRes<Self::FreqResult>;

    /// Performs a Symmetric Fast Fourier Transformation under the assumption that `self`
    /// is symmetric around the center. This assumption
    /// isn't verified and no error is raised if the vector isn't symmetric.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeReal`: if `self` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    ///
    /// # Unstable
    /// Symmetric IFFTs are unstable and may only work under certain conditions.
    fn sfft(self) -> TransRes<Self::FreqResult>;

    /// Performs a Symmetric Fast Fourier Transformation under the assumption that `self`
    /// is symmetric around the center. This assumption
    /// isn't verified and no error is raised if the vector isn't symmetric.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeReal`: if `self` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    ///
    /// # Unstable
    /// Symmetric IFFTs are unstable and may only work under certain conditions.
    fn windowed_sfft(self, window: &WindowFunction<T>) -> TransRes<Self::FreqResult>;
}

/// Defines all operations which are valid on `DataVecs` containing frequency domain data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeInFrquencyDomain` or `VectorMustBeComplex`
/// if the vector isn't in frequency domain and complex number space.
pub trait FrequencyDomainOperations<T> : DataVec<T>
    where T : RealNumber {
    type ComplexTimeResult;

    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the
    /// vector.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{ComplexFreqVector32, FrequencyDomainOperations, DataVec, InterleavedIndex};
    /// let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    /// let result = vector.plain_ifft().expect("Ignoring error handling in examples");
    /// let actual = result.interleaved(0..);
    /// let expected = &[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn plain_ifft(self) -> TransRes<Self::ComplexTimeResult>;

    /// This function mirrors the spectrum vector to transform a symmetric spectrum
    /// into a full spectrum with the DC element at index 0 (no FFT shift/swap halves).
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{ComplexFreqVector32, FrequencyDomainOperations, DataVec, InterleavedIndex};
    /// let vector = ComplexFreqVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let result = vector.mirror().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, -6.0, 3.0, -4.0], result.interleaved(0..));
    /// ```
    fn mirror(self) -> TransRes<Self>;

    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{ComplexFreqVector32, FrequencyDomainOperations, DataVec, InterleavedIndex};
    /// let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 0.0, 0.0, 3.0, 0.0]);
    /// let result = vector.ifft().expect("Ignoring error handling in examples");
    /// let actual = result.interleaved(0..);
    /// let expected = &[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn ifft(self) -> TransRes<Self::ComplexTimeResult>;

    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector and removes the FFT window.
    fn windowed_ifft(self, window: &WindowFunction<T>) -> TransRes<Self::ComplexTimeResult>;

    /// Swaps vector halves after a Fourier Transformation.
    fn fft_shift(self) -> TransRes<Self>;

    /// Swaps vector halves before an Inverse Fourier Transformation.
    fn ifft_shift(self) -> TransRes<Self>;
}

/// Defines all operations which are valid on `DataVecs` containing frequency domain data and
/// the data is assumed to half of complex conjugate symmetric spectrum round 0 Hz where
/// the 0 Hz element itself is real.
/// # Failures
/// All operations in this trait fail with `VectorMustBeInFrquencyDomain`
/// if the vector isn't in frequency domain or with `VectorMustBeConjSymmetric` if the first element (0Hz)
/// isn't real.
pub trait SymmetricFrequencyDomainOperations<T> : DataVec<T>
    where T : RealNumber {
    type RealTimeResult;
    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the
    /// vector.
    fn plain_sifft(self) -> TransRes<Self::RealTimeResult>;

    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    fn sifft(self) -> TransRes<Self::RealTimeResult>;

    /// Performs a Symmetric Inverse Fast Fourier Transformation (SIFFT) and removes the FFT window.
    /// The SIFFT is performed under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    fn windowed_sifft(self, window: &WindowFunction<T>) -> TransRes<Self::RealTimeResult>;
}
*/
