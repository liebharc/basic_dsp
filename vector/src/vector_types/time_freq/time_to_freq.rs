use super::super::{
    Buffer, DataDomain, DspVec, ErrorReason, FrequencyDomainOperations, InsertZerosOpsBuffered,
    MetaData, NumberSpace, RealNumberSpace, RededicateForceOps, ResizeOps, TimeDomain,
    TimeDomainOperations, ToFreqResult, ToSliceMut, TransRes, Vector,
};
use super::fft;
use crate::numbers::*;
use crate::window_functions::*;

/// Defines all operations which are valid on `DataVecs` containing time domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the vector isn't in time domain.
pub trait TimeToFrequencyDomainOperations<S, T>: ToFreqResult
where
    S: ToSliceMut<T>,
    T: RealNumber,
{
    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    ///
    /// This version of the FFT neither applies a window nor does it scale the
    /// vector.
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::*;
    /// # use num_complex::Complex;
    /// let vector = vec!(Complex::new(1.0, 0.0), Complex::new(-0.5, 0.8660254), Complex::new(-0.5, -0.8660254)).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.plain_fft(&mut buffer);
    /// let actual = &result[..];
    /// let expected = &[Complex::new(0.0, 0.0), Complex::new(3.0, 0.0), Complex::new(0.0, 0.0)];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).norm() < 1e-4);
    /// }
    /// ```
    fn plain_fft<B>(self, buffer: &mut B) -> Self::FreqResult
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::*;
    /// # use num_complex::Complex;
    /// let vector = vec!(Complex::new(1.0, 0.0), Complex::new(-0.5, 0.8660254), Complex::new(-0.5, -0.8660254)).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.fft(&mut buffer);
    /// let actual = &result[..];
    /// let expected = &[Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(3.0, 0.0)];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).norm() < 1e-4);
    /// }
    /// ```
    fn fft<B>(self, buffer: &mut B) -> Self::FreqResult
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Applies a FFT window and performs a Fast Fourier Transformation transforming a time
    /// domain vector into a frequency domain vector.
    fn windowed_fft<B>(self, buffer: &mut B, window: &dyn WindowFunction<T>) -> Self::FreqResult
    where
        B: for<'a> Buffer<'a, S, T>;
}

/// Defines all operations which are valid on `DataVecs` containing real time domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the vector isn't in time domain or
/// with `VectorMustHaveAnOddLength` if `self.points()` isn't and odd number.
pub trait SymmetricTimeToFrequencyDomainOperations<S, T>: ToFreqResult
where
    S: ToSliceMut<T>,
    T: RealNumber,
{
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
    fn plain_sfft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Performs a Symmetric Fast Fourier Transformation under the assumption that `self`
    /// is symmetric around the center. This assumption
    /// isn't verified and no error is raised if the vector isn't symmetric.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeReal`: if `self` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    fn sfft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Performs a Symmetric Fast Fourier Transformation under the assumption that `self`
    /// is symmetric around the center. This assumption
    /// isn't verified and no error is raised if the vector isn't symmetric.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeReal`: if `self` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    fn windowed_sfft<B>(
        self,
        buffer: &mut B,
        window: &dyn WindowFunction<T>,
    ) -> TransRes<Self::FreqResult>
    where
        B: for<'a> Buffer<'a, S, T>;
}

impl<S, T, N, D> TimeToFrequencyDomainOperations<S, T> for DspVec<S, T, N, D>
where
    DspVec<S, T, N, D>: ToFreqResult,
    <DspVec<S, T, N, D> as ToFreqResult>::FreqResult:
        RededicateForceOps<DspVec<S, T, N, D>> + FrequencyDomainOperations<S, T>,
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: TimeDomain,
{
    fn plain_fft<B>(mut self, buffer: &mut B) -> Self::FreqResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        if self.domain() != DataDomain::Time {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Self::FreqResult::rededicate_from_force(self);
        }

        if !self.is_complex() {
            self.zero_interleave_b(buffer, 2);
            self.number_space.to_complex();
        }

        fft(&mut self, buffer, false);

        self.domain.to_freq();
        Self::FreqResult::rededicate_from_force(self)
    }

    fn fft<B>(self, buffer: &mut B) -> Self::FreqResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        let mut result = self.plain_fft(buffer);
        result.fft_shift();
        result
    }

    fn windowed_fft<B>(mut self, buffer: &mut B, window: &dyn WindowFunction<T>) -> Self::FreqResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        self.apply_window(window);
        let mut result = self.plain_fft(buffer);
        result.fft_shift();
        result
    }
}

macro_rules! unmirror {
    ($self_: ident, $points: ident) => {
        let len = $points;
        let len = len / 2 + 1;
        $self_
            .resize(len)
            .expect("Shrinking a vector should always succeed");
    };
}

impl<S, T, N, D> SymmetricTimeToFrequencyDomainOperations<S, T> for DspVec<S, T, N, D>
where
    DspVec<S, T, N, D>: ToFreqResult,
    <DspVec<S, T, N, D> as ToFreqResult>::FreqResult:
        RededicateForceOps<DspVec<S, T, N, D>> + FrequencyDomainOperations<S, T> + ResizeOps,
    S: ToSliceMut<T>,
    T: RealNumber,
    N: RealNumberSpace,
    D: TimeDomain,
{
    fn plain_sfft<B>(mut self, buffer: &mut B) -> TransRes<Self::FreqResult>
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        if self.domain() != DataDomain::Time || self.is_complex() {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((
                ErrorReason::InputMustBeInTimeDomain,
                Self::FreqResult::rededicate_from_force(self),
            ));
        }

        if self.points() % 2 == 0 {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((
                ErrorReason::InputMustHaveAnOddLength,
                Self::FreqResult::rededicate_from_force(self),
            ));
        }

        self.zero_interleave_b(buffer, 2);
        self.number_space.to_complex();
        let points = self.points();
        let mut result = self.plain_fft(buffer);
        unmirror!(result, points);
        Ok(result)
    }

    fn sfft<B>(mut self, buffer: &mut B) -> TransRes<Self::FreqResult>
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        if self.domain() != DataDomain::Time || self.is_complex() {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((
                ErrorReason::InputMustBeInTimeDomain,
                Self::FreqResult::rededicate_from_force(self),
            ));
        }

        if self.points() % 2 == 0 {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((
                ErrorReason::InputMustHaveAnOddLength,
                Self::FreqResult::rededicate_from_force(self),
            ));
        }

        self.zero_interleave_b(buffer, 2);
        self.number_space.to_complex();
        let points = self.points();
        let mut result = self.fft(buffer);
        unmirror!(result, points);
        Ok(result)
    }

    fn windowed_sfft<B>(
        mut self,
        buffer: &mut B,
        window: &dyn WindowFunction<T>,
    ) -> TransRes<Self::FreqResult>
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        if self.domain() != DataDomain::Time || self.is_complex() {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((
                ErrorReason::InputMustBeInTimeDomain,
                Self::FreqResult::rededicate_from_force(self),
            ));
        }

        if self.points() % 2 == 0 {
            self.mark_vector_as_invalid();
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((
                ErrorReason::InputMustHaveAnOddLength,
                Self::FreqResult::rededicate_from_force(self),
            ));
        }

        self.zero_interleave_b(buffer, 2);
        self.number_space.to_complex();
        self.apply_window(window);
        let points = self.points();
        let mut result = self.fft(buffer);
        unmirror!(result, points);
        Ok(result)
    }
}
