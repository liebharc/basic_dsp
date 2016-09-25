use RealNumber;
use super::super::{
    ToTimeResult, ToRealTimeResult, TransRes, VoidResult,
    Buffer, ToSliceMut
};
use window_functions::*;

/// Defines all operations which are valid on `DataVecs` containing frequency domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0`
/// if the vector isn't in frequency domain and complex number space.
pub trait FrequencyDomainOperations<S, T> : ToTimeResult
    where S: ToSliceMut<T>,
          T: RealNumber {

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
    fn plain_ifft<B>(self, buffer: &mut B) -> TransRes<Self::TimeResult>
        where B: Buffer<S, T>;

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
    fn mirror<B>(&mut self, buffer: &mut B) -> VoidResult
        where B: Buffer<S, T>;

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
    fn ifft<B>(self, buffer: &mut B) -> TransRes<Self::TimeResult>
        where B: Buffer<S, T>;

    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector and removes the FFT window.
    fn windowed_ifft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> TransRes<Self::TimeResult>
        where B: Buffer<S, T>;

    /// Swaps vector halves after a Fourier Transformation.
    fn fft_shift<B>(&mut self, buffer: &mut B)
        where B: Buffer<S, T>;

    /// Swaps vector halves before an Inverse Fourier Transformation.
    fn ifft_shift<B>(&mut self, buffer: &mut B)
        where B: Buffer<S, T>;
}

/// Defines all operations which are valid on `DataVecs` containing frequency domain data and
/// the data is assumed to half of complex conjugate symmetric spectrum round 0 Hz where
/// the 0 Hz element itself is real.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the first element (0Hz)
/// isn't real.
pub trait SymmetricFrequencyDomainOperations<S, T> : ToRealTimeResult
    where S: ToSliceMut<T>,
          T: RealNumber {
    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the
    /// vector.
    fn plain_sifft<B>(self, buffer: &mut B) -> TransRes<Self::RealTimeResult>
		where B: Buffer<S, T>;

    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    fn sifft<B>(self, buffer: &mut B) -> TransRes<Self::RealTimeResult>
		where B: Buffer<S, T>;

    /// Performs a Symmetric Inverse Fast Fourier Transformation (SIFFT) and removes the FFT window.
    /// The SIFFT is performed under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    fn windowed_sifft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> TransRes<Self::RealTimeResult>
		where B: Buffer<S, T>;
}
