use RealNumber;
use super::super::{
    Owner, ToFreqResult, TransRes, VoidResult,
    Buffer, Vector, InsertZerosOps,
    DspVec, ToSliceMut,
    Domain, RealNumberSpace, RededicateForceOps, ErrorReason
};
use window_functions::*;

/// Defines all operations which are valid on `DataVecs` containing time domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the vector isn't in time domain.
pub trait TimeDomainOperations<S, T> : ToFreqResult
    where S: ToSliceMut<T>,
          T: RealNumber {
    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    ///
    /// This version of the FFT neither applies a window nor does it scale the
    /// vector.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let vector = vec!(1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.plain_fft(&mut buffer).expect("Ignoring error handling in examples");
    /// let actual = &result[..];
    /// let expected = &[0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn plain_fft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T>;

    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    /// # Unstable
    /// FFTs of real vectors are unstable.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let vector = vec!(1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.fft(&mut buffer).expect("Ignoring error handling in examples");
    /// let actual = &result[..];
    /// let expected = &[0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn fft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T>;

    /// Applies a FFT window and performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    fn windowed_fft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T>;

    /// Applies a window to the data vector.
    fn apply_window(&mut self, window: &WindowFunction<T>) -> VoidResult;

    /// Removes a window from the data vector.
    fn unapply_window(&mut self, window: &WindowFunction<T>) -> VoidResult;
}

/// Defines all operations which are valid on `DataVecs` containing real time domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the vector isn't in time domain or
/// with `VectorMustHaveAnOddLength` if `self.points()` isn't and odd number.
pub trait SymmetricTimeDomainOperations<S, T> : ToFreqResult
    where S: ToSliceMut<T>,
          T: RealNumber {

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
    fn plain_sfft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T>;

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
    fn sfft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T>;

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
    fn windowed_sfft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T>;
}
