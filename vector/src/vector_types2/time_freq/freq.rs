use RealNumber;
use std::mem;
use std::ptr;
use super::super::{
    ToTimeResult,
    DspVec, Vector, Buffer, ToSliceMut,
    RededicateForceOps, MetaData,
    ComplexNumberSpace, Owner, FrequencyDomain, DataDomain
};

/// Defines all operations which are valid on `DataVecs` containing frequency domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0`
/// if the vector isn't in frequency domain and complex number space.
pub trait FrequencyDomainOperations<S, T> : ToTimeResult
    where S: ToSliceMut<T>,
          T: RealNumber {

    /// This function mirrors the spectrum vector to transform a symmetric spectrum
    /// into a full spectrum with the DC element at index 0 (no FFT shift/swap halves).
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_freq_vec();
    /// let mut buffer = SingleBuffer::new();
    /// vector.mirror(&mut buffer);
    /// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, -6.0, 3.0, -4.0], &vector[..]);
    /// ```
    fn mirror<B>(&mut self, buffer: &mut B)
        where B: Buffer<S, T>;

    /// Swaps vector halves after a Fourier Transformation.
    fn fft_shift<B>(&mut self, buffer: &mut B)
        where B: Buffer<S, T>;

    /// Swaps vector halves before an Inverse Fourier Transformation.
    fn ifft_shift<B>(&mut self, buffer: &mut B)
        where B: Buffer<S, T>;
}

impl<S, T, N, D> FrequencyDomainOperations<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToTimeResult,
          <DspVec<S, T, N, D> as ToTimeResult>::TimeResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: FrequencyDomain {

    fn mirror<B>(&mut self, buffer: &mut B)
      where B: Buffer<S, T> {
          if self.domain() != DataDomain::Frequency &&
             !self.is_complex() {
              self.valid_len = 0;
              return;
          }

         let len = self.len();
         let step = 2;
         let temp_len = 2 * len - step;
         let mut temp = buffer.get(temp_len);
         {
              let data = self.data.to_slice();
              let mut temp = temp.to_slice_mut();
              {
                  let data = &data[step] as *const T;
                  let target = &mut temp[step] as *mut T;
                  unsafe {
                      ptr::copy(data, target, len - step);
                  }
              }
              {
                  let data = &data[0] as *const T;
                  let target = &mut temp[0] as *mut T;
                  unsafe {
                      ptr::copy(data, target, step);
                      }
                  }
                  let mut j = step + 1;
                  let mut i = temp_len - 1;
                  while i >= len {
                      temp[i] = -data[j];
                      temp[i - 1] = data[j - 1];
                      i -= 2;
                      j += 2;
                  }

                  self.valid_len = temp_len;
          }

          mem::swap(&mut self.data, &mut temp);
          buffer.free(temp);
    }

    fn fft_shift<B>(&mut self, buffer: &mut B)
      where B: Buffer<S, T> {
        self.swap_halves_priv(buffer, true)
    }

    fn ifft_shift<B>(&mut self, buffer: &mut B)
      where B: Buffer<S, T> {
        self.swap_halves_priv(buffer, false)
    }
 }
