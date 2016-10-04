use RealNumber;
use super::super::{
    Owner, ToFreqResult, TransRes, TimeDomain,
    Buffer, Vector, DataDomain,
    DspVec, ToSliceMut, RealNumberSpace,
    NumberSpace, RededicateForceOps, ErrorReason,
    InsertZerosOpsBuffered, FrequencyDomainOperations, TimeDomainOperations
};
use super::fft;
use window_functions::*;

/// Defines all operations which are valid on `DataVecs` containing time domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the vector isn't in time domain.
pub trait TimeToFrequencyDomainOperations<S, T> : ToFreqResult
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
    /// use std::f32;
    /// use basic_dsp_vector::vector_types2::*;
    /// let vector = vec!(1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.plain_fft(&mut buffer);
    /// let actual = &result[..];
    /// let expected = &[0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f32::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// ```
    fn plain_fft<B>(self, buffer: &mut B) -> Self::FreqResult
        where B: Buffer<S, T>;

    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    /// # Unstable
    /// FFTs of real vectors are unstable.
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::vector_types2::*;
    /// let vector = vec!(1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.fft(&mut buffer);
    /// let actual = &result[..];
    /// let expected = &[0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f32::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// ```
    fn fft<B>(self, buffer: &mut B) -> Self::FreqResult
        where B: Buffer<S, T>;

    /// Applies a FFT window and performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    fn windowed_fft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> Self::FreqResult
        where B: Buffer<S, T>;
}

/// Defines all operations which are valid on `DataVecs` containing real time domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the vector isn't in time domain or
/// with `VectorMustHaveAnOddLength` if `self.points()` isn't and odd number.
pub trait SymmetricTimeToFrequencyDomainOperations<S, T> : ToFreqResult
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


impl<S, T, N, D> TimeToFrequencyDomainOperations<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToFreqResult,
          <DspVec<S, T, N, D> as ToFreqResult>::FreqResult: RededicateForceOps<DspVec<S, T, N, D>> + FrequencyDomainOperations<S, T>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: TimeDomain {
    fn plain_fft<B>(mut self, buffer: &mut B) -> Self::FreqResult
        where B: Buffer<S, T> {
        if self.domain() != DataDomain::Time {
            self.valid_len = 0;
            self.number_space.to_complex();
            self.domain.to_freq();
            return Self::FreqResult::rededicate_from_force(self)
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
        where B: Buffer<S, T> {
        let mut result = self.plain_fft(buffer);
        result.fft_shift(buffer);
        result
    }

    fn windowed_fft<B>(mut self, buffer: &mut B, window: &WindowFunction<T>) -> Self::FreqResult
        where B: Buffer<S, T> {
        self.apply_window(window);
        let mut result = self.plain_fft(buffer);
        result.fft_shift(buffer);
        result
    }
}

macro_rules! unmirror {
    ($self_: ident) => {
        let len = $self_.points();
        let len = len / 2 + 1;
        $self_.resize(len).expect("Shrinking a vector should always succeed");
    }
}

impl<S, T, N, D> SymmetricTimeToFrequencyDomainOperations<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToFreqResult,
          <DspVec<S, T, N, D> as ToFreqResult>::FreqResult: RededicateForceOps<DspVec<S, T, N, D>> + FrequencyDomainOperations<S, T> + Vector<T>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: RealNumberSpace,
          D: TimeDomain {
    fn plain_sfft<B>(mut self, buffer: &mut B) -> TransRes<Self::FreqResult>
      where B: Buffer<S, T> {
      if self.domain() != DataDomain::Time ||
         self.is_complex() {
          self.valid_len = 0;
          self.number_space.to_complex();
          self.domain.to_freq();
          return Err((ErrorReason::InputMustBeInTimeDomain, Self::FreqResult::rededicate_from_force(self)));
      }

      if self.points() % 2 == 0 {
          self.valid_len = 0;
          self.number_space.to_complex();
          self.domain.to_freq();
          return Err((ErrorReason::InputMustHaveAnOddLength, Self::FreqResult::rededicate_from_force(self)));
      }

      self.zero_interleave_b(buffer, 2);
      self.number_space.to_complex();
      let mut result = self.plain_fft(buffer);
      unmirror!(result);
      Ok(result)
    }

    fn sfft<B>(mut self, buffer: &mut B) -> TransRes<Self::FreqResult>
      where B: Buffer<S, T> {
      if self.domain() != DataDomain::Time ||
         self.is_complex() {
          self.valid_len = 0;
          self.number_space.to_complex();
          self.domain.to_freq();
          return Err((ErrorReason::InputMustBeInTimeDomain, Self::FreqResult::rededicate_from_force(self)));
      }

      if self.points() % 2 == 0 {
          self.valid_len = 0;
          self.number_space.to_complex();
          self.domain.to_freq();
          return Err((ErrorReason::InputMustHaveAnOddLength, Self::FreqResult::rededicate_from_force(self)));
      }

      self.zero_interleave_b(buffer, 2);
      self.number_space.to_complex();
      let mut result = self.fft(buffer);
      unmirror!(result);
      Ok(result)
    }

    fn windowed_sfft<B>(mut self, buffer: &mut B, window: &WindowFunction<T>) -> TransRes<Self::FreqResult>
      where B: Buffer<S, T> {
      if self.domain() != DataDomain::Time ||
         self.is_complex() {
          self.valid_len = 0;
          self.number_space.to_complex();
          self.domain.to_freq();
          return Err((ErrorReason::InputMustBeInTimeDomain, Self::FreqResult::rededicate_from_force(self)));
      }

      if self.points() % 2 == 0 {
          self.valid_len = 0;
          self.number_space.to_complex();
          self.domain.to_freq();
          return Err((ErrorReason::InputMustHaveAnOddLength, Self::FreqResult::rededicate_from_force(self)));
      }

      self.zero_interleave_b(buffer, 2);
      self.number_space.to_complex();
      self.apply_window(window);
      let mut result = self.fft(buffer);
      unmirror!(result);
      Ok(result)
    }
}
