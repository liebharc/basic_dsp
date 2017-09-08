use numbers::*;
use super::super::{ToTimeResult, DspVec, Vector, Buffer, BufferBorrow, ToSliceMut,
                   RededicateForceOps, MetaData, ComplexNumberSpace, FrequencyDomain, DataDomain};

/// Defines all operations which are valid on `DataVecs` containing frequency domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0`
/// if the vector isn't in frequency domain and complex number space.
pub trait FrequencyDomainOperations<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// This function mirrors the spectrum vector to transform a symmetric spectrum
    /// into a full spectrum with the DC element at index 0 (no FFT shift/swap halves).
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1`
    /// points.
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
    fn mirror<B>(&mut self, buffer: &mut B) where B: for<'a> Buffer<'a, S, T>;

    /// Swaps vector halves after a Fourier Transformation.
    fn fft_shift(&mut self);

    /// Swaps vector halves before an Inverse Fourier Transformation.
    fn ifft_shift(&mut self);
}

impl<S, T, N, D> FrequencyDomainOperations<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToTimeResult,
          <DspVec<S, T, N, D> as ToTimeResult>::TimeResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: FrequencyDomain
{
    fn mirror<B>(&mut self, buffer: &mut B)
        where B: for<'a> Buffer<'a, S, T>
    {
        if self.domain() != DataDomain::Frequency && !self.is_complex() {
            self.valid_len = 0;
            return;
        }

        let len = self.len();
        let step = 2;
        let temp_len = 2 * len - step;
        let mut temp = buffer.borrow(temp_len);
        {
            let data = self.data.to_slice();
            let temp = temp.to_slice_mut();
            &mut temp[step..len].clone_from_slice(&data[step..len]);
            &mut temp[0..step].clone_from_slice(&data[0..step]);
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

        temp.trade(&mut self.data);
    }

    fn fft_shift(&mut self) {
        self.swap_halves_priv(true)
    }

    fn ifft_shift(&mut self) {
        self.swap_halves_priv(false)
    }
}
