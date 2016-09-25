use RealNumber;
use std::mem;
use num::Complex;
use rustfft::FFT;
use super::super::{
    array_to_complex, array_to_complex_mut,
    Owner, ToFreqResult, TransRes, TimeDomain,
    Buffer, Vector, DataDomain,
    DspVec, ToSliceMut,
    NumberSpace, RededicateForceOps, ErrorReason,
    InsertZerosOps
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
    /// use std::f32;
    /// use basic_dsp_vector::vector_types2::*;
    /// let vector = vec!(1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.plain_fft(&mut buffer).expect("Ignoring error handling in examples");
    /// let actual = &result[..];
    /// let expected = &[0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f32::abs(actual[i] - expected[i]) < 1e-4);
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
    /// use std::f32;
    /// use basic_dsp_vector::vector_types2::*;
    /// let vector = vec!(1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.fft(&mut buffer).expect("Ignoring error handling in examples");
    /// let actual = &result[..];
    /// let expected = &[0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f32::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// ```
    fn fft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T>;

    /// Applies a FFT window and performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector.
    fn windowed_fft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T>;

    /// Applies a window to the data vector.
    fn apply_window(&mut self, window: &WindowFunction<T>);

    /// Removes a window from the data vector.
    fn unapply_window(&mut self, window: &WindowFunction<T>);
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


impl<S, T, N, D> TimeDomainOperations<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToFreqResult,
          <DspVec<S, T, N, D> as ToFreqResult>::FreqResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: TimeDomain {
    fn plain_fft<B>(mut self, buffer: &mut B) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T> {
        if self.domain() != DataDomain::Time {
            self.valid_len = 0;
            self.number_space.to_complex();
            self.domain.to_freq();
            return Err((ErrorReason::InputMustBeInTimeDomain, Self::FreqResult::rededicate_from_force(self)));
        }

        if !self.is_complex() {
            self.zero_interleave_b(buffer, 2);
            self.number_space.to_complex();
        }

        let len = self.len();
        let mut temp = buffer.get(len);
        {
            let temp = temp.to_slice_mut();
            let points = self.points();
            let rbw = (T::from(points).unwrap()) * self.delta;
            self.delta = rbw;
            let mut fft = FFT::new(points, false);
            let signal = self.data.to_slice();
            let spectrum = &mut temp[0..len];
            let signal = array_to_complex(&signal[0..len]);
            let spectrum = array_to_complex_mut(spectrum);
            fft.process(signal, spectrum);
        }

        mem::swap(&mut self.data, &mut temp);
        buffer.free(temp);

        self.domain.to_freq();
        Ok(Self::FreqResult::rededicate_from_force(self))
    }

    fn fft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T> {
            /* self.plain_fft()
             .and_then(|v|v.fft_shift())*/
        panic!("Panic")
    }

    fn windowed_fft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> TransRes<Self::FreqResult>
        where B: Buffer<S, T> {
            /*   self.apply_window(window)
              .and_then(|v|v.plain_fft())
              .and_then(|v|v.fft_shift())*/
        panic!("Panic")
    }

    fn apply_window(&mut self, window: &WindowFunction<T>) {
        if self.is_complex() {
           self.multiply_window_priv(
                window.is_symmetric(),
                |array|array_to_complex_mut(array),
                window,
                |f,i,p|Complex::<T>::new(f.window(i, p), T::zero()));
        } else {
            self.multiply_window_priv(
                window.is_symmetric(),
                |array|array,
                window,
                |f,i,p|f.window(i, p));
        }
    }

    fn unapply_window(&mut self, window: &WindowFunction<T>) {
        if self.is_complex() {
           self.multiply_window_priv(
                window.is_symmetric(),
                |array|array_to_complex_mut(array),
                window,
                |f,i,p|Complex::<T>::new(T::one() / f.window(i, p), T::zero()));
        } else {
            self.multiply_window_priv(
                window.is_symmetric(),
                |array|array,
                window,
                |f,i,p|T::one() / f.window(i, p));
        }
    }
}
