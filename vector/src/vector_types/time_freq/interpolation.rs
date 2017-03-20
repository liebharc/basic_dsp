use {RealNumber, array_to_complex, array_to_complex_mut};
use conv_types::{RealImpulseResponse, RealFrequencyResponse};
use num::traits::Zero;
use std::ops::{Add, Mul};
use super::WrappingIterator;
use simd_extensions::*;
use multicore_support::*;
use super::super::{VoidResult, DspVec, Domain, ToComplexVector, ComplexOps, PaddingOption, Owner,
                   NumberSpace, ToSliceMut, GenDspVec, ToDspVector, DataDomain, Buffer, Vector,
                   RealNumberSpace, ErrorReason, InsertZerosOpsBuffered, ScaleOps, MetaData,
                   ResizeOps};
use super::fft;
use std::mem;
use num::complex::Complex;
use InlineVector;

/// Provides interpolation operations for real and complex data vectors.
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions
/// are consistent. However the actual implementation is lacking tests.
pub trait InterpolationOps<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Interpolates `self` with the convolution function `function` by the real value
    /// `interpolation_factor`. InterpolationOps is done in time domain and the argument
    /// `conv_len` can be used to balance accuracy and computational performance.
    /// A `delay` can be used to delay or phase shift the vector.
    /// The `delay` considers `self.delta()`.
    ///
    /// The complexity of this `interpolatef` is `O(self.points() * conv_len)`,
    /// while for `interpolatei` it's `O(self.points() * log(self.points()))`. If computational
    /// performance is important you should therefore decide how large `conv_len` needs to be
    /// to yield the desired accuracy. If you compare `conv_len` to `log(self.points)` you should
    /// get a feeling for the expected performance difference. More important is however to do a
    /// test run to compare the speed of `interpolatef` and `interpolatei`.
    /// Together with the information that changing the vectors size change `log(self.points()`
    /// but not `conv_len` gives the indication that `interpolatef` performs faster for larger
    /// vectors while `interpolatei` performs faster for smaller vectors.
    fn interpolatef<B>(&mut self,
                       buffer: &mut B,
                       function: &RealImpulseResponse<T>,
                       interpolation_factor: T,
                       delay: T,
                       conv_len: usize)
        where B: Buffer<S, T>;

    /// Interpolates `self` with the convolution function `function` by the interger value
    /// `interpolation_factor`. InterpolationOps is done in in frequency domain.
    ///
    /// See the description of `interpolatef` for some basic performance considerations.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `ArgumentFunctionMustBeSymmetric`: if `!self.is_complex() && !function.is_symmetric()`
    ///    or in words if `self` is a real vector and `function` is asymmetric.
    ///    Converting the vector into a complex vector before the interpolation is one way
    ///    to resolve this error.
    fn interpolatei<B>(&mut self,
                       buffer: &mut B,
                       function: &RealFrequencyResponse<T>,
                       interpolation_factor: u32)
                       -> VoidResult
        where B: Buffer<S, T>;

    /// Interpolates the signal in frequency domain by padding it with zeros.
    /// It is required that: `target_points > self.len()`
    fn interpolate<B>(&mut self,
                   buffer: &mut B,
                   function: &RealFrequencyResponse<T>,
                   target_points: usize,
                   delay: T)
                   -> VoidResult
            where B: Buffer<S, T>;

    /// Decimates or downsamples `self`. `decimatei` is the inverse function to `interpolatei`.
    fn decimatei(&mut self, decimation_factor: u32, delay: u32);
}

/// Provides interpolation operations which are only applicable for real data vectors.
/// # Failures
/// All operations in this trait fail with `VectorMustBeReal` if the vector isn't in the
/// real number space.
pub trait RealInterpolationOps<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Piecewise cubic hermite interpolation between samples.
    /// # Unstable
    /// Algorithm might need to be revised.
    /// This operation and `interpolate_lin` might be merged into one function with an
    /// additional argument in future.
    fn interpolate_hermite<B>(&mut self, buffer: &mut B, interpolation_factor: T, delay: T)
        where B: Buffer<S, T>;

    /// Linear interpolation between samples.
    /// # Unstable
    /// This operation and `interpolate_hermite` might be merged into one function with an
    /// additional argument in future.
    fn interpolate_lin<B>(&mut self, buffer: &mut B, interpolation_factor: T, delay: T)
        where B: Buffer<S, T>;
}

fn interpolate_priv_scalar<T, TT>(temp: &mut [TT],
                                  data: &[TT],
                                  function: &RealImpulseResponse<T>,
                                  interpolation_factor: T,
                                  delay: T,
                                  conv_len: usize,
                                  multicore_settings: &MultiCoreSettings)
    where T: RealNumber,
          TT: Zero + Mul<Output = TT> + Copy + Send + Sync + From<T>
{
    Chunk::execute_with_range(Complexity::Large,
                              &multicore_settings,
                              temp,
                              1,
                              data,
                              move |dest_range, range, data| {
          let mut i = range.start as isize;
          for num in dest_range {
              let center = T::from(i).unwrap() / interpolation_factor;
              let rounded = center.floor();
              let iter = WrappingIterator::new(data,
                                               rounded.to_isize().unwrap() - conv_len as isize - 1,
                                               2 * conv_len + 1);
              let mut sum = TT::zero();
              let mut j = -T::from(conv_len).unwrap() - (center - rounded) + delay;
              for c in iter {
                  sum = sum + c * TT::from(function.calc(j));
                  j = j + T::one();
              }
              (*num) = sum;
              i += 1;
          }
    });
}

fn function_to_vectors<T>(function: &RealImpulseResponse<T>,
                          conv_len: usize,
                          complex_result: bool,
                          interpolation_factor: usize,
                          delay: T)
                          -> InlineVector<GenDspVec<Vec<T>, T>>
    where T: RealNumber
{
    let mut result = InlineVector::with_capacity(interpolation_factor);
    for shift in 0..interpolation_factor {
        let offset = T::from(shift).unwrap() / T::from(interpolation_factor).unwrap();
        result.push(function_to_vector(function, conv_len, complex_result, offset, delay));
    }

    result
}

fn function_to_vector<T>(function: &RealImpulseResponse<T>,
                         conv_len: usize,
                         complex_result: bool,
                         offset: T,
                         delay: T)
                         -> GenDspVec<Vec<T>, T>
    where T: RealNumber
{
    let step = if complex_result { 2 } else { 1 };
    let data_len = step * (2 * conv_len + 1);
    let mut imp_resp = vec!(T::zero(); data_len).to_gen_dsp_vec(complex_result, DataDomain::Time);
    let mut i = 0;
    let mut j = -(T::from(conv_len).unwrap() - T::one()) + delay;
    while i < data_len {
        let value = function.calc(j - offset);
        imp_resp[i] = value;
        i += step;
        j = j + T::one();
    }
    imp_resp
}

impl<S, T, N, D> DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn interpolate_priv_simd<TT, C, CMut, RMul, RSum, B>(&mut self,
                                                         buffer: &mut B,
                                                         function: &RealImpulseResponse<T>,
                                                         interpolation_factor: usize,
                                                         delay: T,
                                                         conv_len: usize,
                                                         new_len: usize,
                                                         convert: C,
                                                         convert_mut: CMut,
                                                         simd_mul: RMul,
                                                         simd_sum: RSum)
        where TT: Zero + Clone + From<T> + Copy + Add<Output = TT> + Mul<Output = TT> + Send + Sync,
              C: Fn(&[T]) -> &[TT],
              CMut: Fn(&mut [T]) -> &mut [TT],
              RMul: Fn(T::Reg, T::Reg) -> T::Reg + Sync,
              RSum: Fn(T::Reg) -> TT + Sync,
              B: Buffer<S, T>
    {
        let data_len = self.len();
        let mut temp = buffer.get(new_len);
        {
            let step = if self.is_complex() { 2 } else { 1 };
            let number_of_shifts = T::Reg::len() / step;
            let vectors = function_to_vectors(function,
                                              conv_len,
                                              self.is_complex(),
                                              interpolation_factor,
                                              delay);
            let mut shifts = Vec::with_capacity(vectors.len() * number_of_shifts);
            for vector in &vectors[..] {
                let shifted_copies = DspVec::create_shifted_copies(vector);
                for shift in shifted_copies {
                    shifts.push(shift);
                }
            }

            let data = self.data.to_slice();
            let mut temp = temp.to_slice_mut();
            let dest = convert_mut(&mut temp[0..new_len]);
            let len = dest.len();
            let scalar_len = vectors[0].points() * interpolation_factor;
            let mut i = 0;
            {
                let data = convert(&data[0..data_len]);
                for num in &mut dest[0..scalar_len] {
                    (*num) = Self::interpolate_priv_simd_step(i,
                                                              interpolation_factor,
                                                              conv_len,
                                                              data,
                                                              &vectors[..]);
                    i += 1;
                }
            }

            let (scalar_left, _, vectorization_length) =
                T::Reg::calc_data_alignment_reqs(&data[0..data_len]);
            let simd = T::Reg::array_to_regs(&data[scalar_left..vectorization_length]);
            // Length of a SIMD reg relative to the length of type T
            // which is 1 for real numbers or 2 for complex numbers
            let simd_len_in_t = T::Reg::len() / step;
            Chunk::execute_with_range(Complexity::Large,
                                      &self.multicore_settings,
                                      &mut dest[scalar_len..len - scalar_len],
                                      1,
                                      simd,
                                      move |dest_range, range, simd| {
                  let mut i = range.start + scalar_len;
                  for num in dest_range {
                      let rounded = (i + interpolation_factor - 1) / interpolation_factor;
                      let end = rounded + conv_len;
                      let simd_end = (end + simd_len_in_t - 1) / simd_len_in_t;
                      let simd_shift = end % simd_len_in_t;
                      let factor_shift = i % interpolation_factor;
                      // The reasoning for the next match is analog to the explanation in the
                      // create_shifted_copies function.
                      // We need the inverse of the mod unless we start with zero
                      let factor_shift = match factor_shift {
                          0 => 0,
                          x => interpolation_factor - x,
                      };
                      let selection = factor_shift * simd_len_in_t + simd_shift;
                      let shifted = &shifts[selection];
                      let mut sum = T::Reg::splat(T::zero());
                      let simd_iter = simd[simd_end - shifted.len()..simd_end].iter();
                      let iteration = simd_iter.zip(shifted);
                      for (this, other) in iteration {
                          sum = sum + simd_mul(*this, *other);
                      }
                      (*num) = simd_sum(sum);
                      i += 1;
                  }
            });

            i = len - scalar_len;
            {
                let data = convert(&data[0..data_len]);
                for num in &mut dest[len - scalar_len..len] {
                    (*num) = Self::interpolate_priv_simd_step(i,
                                                              interpolation_factor,
                                                              conv_len,
                                                              data,
                                                              &vectors[..]);
                    i += 1;
                }
            }
        }
        self.valid_len = new_len;
        mem::swap(&mut temp, &mut self.data);
        buffer.free(temp);
    }

    #[inline]
    fn interpolate_priv_simd_step<TT>(i: usize,
                                      interpolation_factor: usize,
                                      conv_len: usize,
                                      data: &[TT],
                                      vectors: &[GenDspVec<Vec<T>, T>])
                                      -> TT
        where TT: Zero + Clone + From<T> + Copy + Add<Output = TT> + Mul<Output = TT>
    {
        let rounded = i / interpolation_factor;
        let iter = WrappingIterator::new(data,
                                         rounded as isize - conv_len as isize,
                                         2 * conv_len + 1);
        let vector = &vectors[i % interpolation_factor];
        let step = if vector.is_complex() { 2 } else { 1 };
        let mut sum = TT::zero();
        let mut j = 0;
        for c in iter {
            sum = sum + c * TT::from(vector[j]);
            j += step;
        }
        sum
    }
}

impl<S, T, N, D> InterpolationOps<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: InsertZerosOpsBuffered<S, T> + ScaleOps<T>,
          S: ToSliceMut<T> + ToComplexVector<S, T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn interpolatef<B>(&mut self,
                       buffer: &mut B,
                       function: &RealImpulseResponse<T>,
                       interpolation_factor: T,
                       delay: T,
                       conv_len: usize)
        where B: Buffer<S, T>
    {
        let delay = delay / self.delta;
        let len = self.len();
        let points_half = self.points() / 2;
        let conv_len = if conv_len > points_half {
            points_half
        } else {
            conv_len
        };
        let is_complex = self.is_complex();
        let new_len = (T::from(len).unwrap() * interpolation_factor).round().to_usize().unwrap();
        let new_len = new_len + new_len % 2;
        if conv_len <= 202 && new_len >= 2000 &&
           (interpolation_factor.round() - interpolation_factor).abs() < T::from(1e-6).unwrap() {
            let interpolation_factor = interpolation_factor.round().to_usize().unwrap();
            if self.is_complex() {
                return self.interpolate_priv_simd(buffer,
                                                  function,
                                                  interpolation_factor,
                                                  delay,
                                                  conv_len,
                                                  new_len,
                                                  |x| array_to_complex(x),
                                                  |x| array_to_complex_mut(x),
                                                  |x, y| x.mul_complex(y),
                                                  |x| x.sum_complex());
            } else {
                return self.interpolate_priv_simd(buffer,
                                                  function,
                                                  interpolation_factor,
                                                  delay,
                                                  conv_len,
                                                  new_len,
                                                  |x| x,
                                                  |x| x,
                                                  |x, y| x * y,
                                                  |x| x.sum_real());
            }
        } else if is_complex {
            let mut temp = buffer.get(new_len);
            {
                let data = self.data.to_slice();
                let data = &data[0..len];
                let temp = temp.to_slice_mut();
                let mut temp = array_to_complex_mut(&mut temp[0..new_len]);
                let data = array_to_complex(data);
                interpolate_priv_scalar(temp,
                                        data,
                                        function,
                                        interpolation_factor,
                                        delay,
                                        conv_len,
                                        &self.multicore_settings);
            }
            mem::swap(&mut temp, &mut self.data);
            buffer.free(temp);
        } else {
            let mut temp = buffer.get(new_len);
            {
                let data = self.data.to_slice();
                let data = &data[0..len];
                let temp = temp.to_slice_mut();
                interpolate_priv_scalar(temp,
                                        data,
                                        function,
                                        interpolation_factor,
                                        delay,
                                        conv_len,
                                        &self.multicore_settings);
            }
            mem::swap(&mut temp, &mut self.data);
            buffer.free(temp);
        }

        self.valid_len = new_len;
    }

    fn interpolatei<B>(&mut self,
                       buffer: &mut B,
                       function: &RealFrequencyResponse<T>,
                       interpolation_factor: u32)
                       -> VoidResult
        where B: Buffer<S, T>
    {
        if interpolation_factor <= 1 {
            return Ok(());
        }

        if !function.is_symmetric() && !self.is_complex() {
            return Err(ErrorReason::ArgumentFunctionMustBeSymmetric);
        }

        let is_complex = self.is_complex();

        if !self.is_complex() {
            self.zero_interleave_b(buffer, interpolation_factor * 2);
        } else {
            self.zero_interleave_b(buffer, interpolation_factor);
        }
        // Vector is always complex from here on

        fft(self, buffer, false); // fft
        let points = self.len() / 2;
        let interpolation_factorf = T::from(interpolation_factor).unwrap();
        self.multiply_function_priv(function.is_symmetric(),
                                    true,
                                    interpolation_factorf,
                                    |array| array_to_complex_mut(array),
                                    function,
                                    |f, x| Complex::<T>::new(f.calc(x), T::zero()));
        fft(self, buffer, true); // ifft
        self.scale(T::one() / T::from(points).unwrap());
        if !is_complex {
            // Convert vector back into real number space
            self.pure_complex_to_real_operation_inplace(|x, _arg| x.re, ());
        }

        Ok(())
    }

    fn interpolate<B>(&mut self,
                   buffer: &mut B,
                   function: &RealFrequencyResponse<T>,
                   dest_points: usize,
                   delay: T)
                   -> VoidResult
            where B: Buffer<S, T> {
        if !function.is_symmetric() && !self.is_complex() {
            return Err(ErrorReason::ArgumentFunctionMustBeSymmetric);
        }

        let delta_t = self.delta();
        let is_complex = self.is_complex();
        let orig_len = self.len();
        let dest_len = if is_complex { 2 * dest_points } else { dest_points };
        let interpolation_factorf = T::from(dest_points).unwrap() / T::from(self.points()).unwrap();

        if !self.is_complex() {
            // Vector is always complex from here on (however the complex flag inside the vector
            // may not be set)
            self.zero_interleave_b(buffer, 2);
        }

        fft(self, buffer, false); // fft

        let two = T::one() + T::one();
        let pi = T::PI();
        // Add the delay, which is a linear phase in frequency domain
        if delay != T::zero()
        {
            let points = self.len() / 2;
            let pos_points = points / 2;
            let neg_points = points - pos_points;
            let phase_inc = -two * pi * -delay / delta_t / delta_t
                            / T::from(points).unwrap();
            {
                let len = self.len();
                let mut freq = (&mut self[2*pos_points..len]).to_complex_freq_vec();
                // Negative frequencies
                let start = -T::from(neg_points).unwrap() * phase_inc;
                freq.multiply_complex_exponential(phase_inc, start);
            }
            {
                let mut freq = (&mut self[0..2*pos_points]).to_complex_freq_vec();
                // Zero and psoitive frequencies
                let start = T::zero();
                freq.multiply_complex_exponential(phase_inc, start);
            }
        }

        if dest_len > orig_len {
            {
                let mut data = buffer.construct_new(0);
                let len = self.len();
                mem::swap(&mut self.data, &mut data); // Take ownership
                let mut complex = data.to_complex_freq_vec();
                complex.resize(len)
                    .expect("Resize should succeed since the size has been checked before");
                complex.zero_pad_b(buffer, dest_points, PaddingOption::Center);
                mem::swap(&mut self.data, &mut complex.data);
                self.resize(2 * dest_points)
                    .expect("Resize should success, since we just increased the storage size")
            }
            // Apply the window function
            self.multiply_function_priv(
                function.is_symmetric(),
                true,
                interpolation_factorf,
                |array| array_to_complex_mut(array),
                function,
                |f, x| Complex::<T>::new(f.calc(x), T::zero()));
        }
        else if dest_len < orig_len {
            return Err(ErrorReason::InvalidArgumentLength);
        }
        
    	fft(self, buffer, true); // ifft
        let points = self.len() / 2;
        self.scale(T::one() / T::from(points).unwrap());
        if !is_complex {
            // Convert vector back into real number space
            self.pure_complex_to_real_operation_inplace(|x, _arg| x.re, ());
        }

        Ok(())
    }

    fn decimatei(&mut self, decimation_factor: u32, delay: u32) {
        let mut i = delay as usize;
        let mut j = 0;
        let len = self.len();
        let points = self.points();
        let is_complex = self.is_complex();
        let data = self.data.to_slice_mut();
        if is_complex {
            let mut data = array_to_complex_mut(&mut data[0..len]);
            let decimation_factor = decimation_factor as usize;
            while i < points {
                data[j] = data[i];
                i += decimation_factor;
                j += 1;
            }
            self.valid_len = j * 2;
        } else {
            let mut data = &mut data[0..len];
            let decimation_factor = decimation_factor as usize;
            while i < len {
                data[j] = data[i];
                i += decimation_factor;
                j += 1;
            }
            self.valid_len = j;
        }
    }
}

impl<S, T, N, D> RealInterpolationOps<S, T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    fn interpolate_lin<B>(&mut self, buffer: &mut B, interpolation_factor: T, delay: T)
        where B: Buffer<S, T>
    {
        let data_len = self.len();
        let dest_len = (T::from(data_len - 1).unwrap() * interpolation_factor)
            .round()
            .to_usize()
            .unwrap() + 1;
        let mut temp = buffer.get(dest_len);
        {
            if self.is_complex() {
                self.valid_len = 0;
                return;
            }
            let data = self.data.to_slice();
            let mut temp = temp.to_slice_mut();
            let data = &data[0..data_len];
            let dest = &mut temp[0..dest_len];
            let mut i = T::zero();

            for num in &mut dest[0..dest_len - 1] {
                let rounded = i / interpolation_factor + delay;
                let beforef = rounded.floor();
                let before = beforef.to_usize().unwrap();
                let next = before + 1;
                let x0 = beforef;
                let y0 = data[before];
                let y1 = data[next];
                let x = rounded;
                (*num) = y0 + (y1 - y0) * (x - x0);
                i = i + T::one();
            }
            dest[dest_len - 1] = data[data_len - 1];
            self.valid_len = dest_len;
        }
        mem::swap(&mut self.data, &mut temp);
        buffer.free(temp);
    }

    fn interpolate_hermite<B>(&mut self, buffer: &mut B, interpolation_factor: T, delay: T)
        where B: Buffer<S, T>
    {
        let data_len = self.len();
        let dest_len = (T::from(data_len - 1).unwrap() * interpolation_factor)
            .round()
            .to_usize()
            .unwrap() + 1;
        let mut temp = buffer.get(dest_len);
        {
            if self.is_complex() {
                self.valid_len = 0;
                return;
            }
            let data = self.data.to_slice();
            let mut temp = temp.to_slice_mut();
            // Literature: http://paulbourke.net/miscellaneous/interpolation/
            let data = &data[0..data_len];
            let dest = &mut temp[0..dest_len];
            let mut i = T::zero();
            let start = ((T::one() - delay) * interpolation_factor).ceil().to_usize().unwrap();
            let end = start + 1;
            let half = T::from(0.5).unwrap();
            let one_point_five = T::from(1.5).unwrap();
            let two = T::from(2.0).unwrap();
            let two_point_five = T::from(2.5).unwrap();
            for num in &mut dest[0..start] {
                let rounded = i / interpolation_factor + delay;
                let beforef = rounded.floor();
                let before = beforef.to_usize().unwrap();
                assert_eq!(before, 0);
                let next = before + 1;
                let next_next = next + 1;
                let y1 = data[before];
                let y2 = data[next];
                let y3 = data[next_next];
                let x = rounded - beforef;
                let y0 = y1 - (y2 - y1);
                let x2 = x * x;
                let a0 = -half * y0 + one_point_five * y1 - one_point_five * y2 + half * y3;
                let a1 = y0 - two_point_five * y1 + two * y2 - half * y3;
                let a2 = -half * y0 + half * y2;
                let a3 = y1;

                (*num) = (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                i = i + T::one();
            }

            for num in &mut dest[start..dest_len - end] {
                let rounded = i / interpolation_factor + delay;
                let beforef = rounded.floor();
                let before = beforef.to_usize().unwrap();
                let before_before = before - 1;
                let next = before + 1;
                let next_next = next + 1;
                let y0 = data[before_before];
                let y1 = data[before];
                let y2 = data[next];
                let y3 = data[next_next];
                let x = rounded - beforef;
                let x2 = x * x;
                let a0 = -half * y0 + one_point_five * y1 - one_point_five * y2 + half * y3;
                let a1 = y0 - two_point_five * y1 + two * y2 - half * y3;
                let a2 = -half * y0 + half * y2;
                let a3 = y1;

                (*num) = (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                i = i + T::one();
            }

            for num in &mut dest[dest_len - end..dest_len] {
                let rounded = i / interpolation_factor + delay;
                let beforef = rounded.floor();
                let before = beforef.to_usize().unwrap();
                assert!(before + 2 >= data_len);
                let before_before = before - 1;
                let y0 = data[before_before];
                let y1 = data[before];
                let y2 = if before < data_len - 1 {
                    data[before + 1]
                } else {
                    y1 + (y1 - y0)
                };
                let y3 = if before < data_len - 2 {
                    data[before + 2]
                } else {
                    y2 + (y2 - y1)
                };
                let x = rounded - beforef;
                let x2 = x * x;
                let a0 = -half * y0 + one_point_five * y1 - one_point_five * y2 + half * y3;
                let a1 = y0 - two_point_five * y1 + two * y2 - half * y3;
                let a2 = -half * y0 + half * y2;
                let a3 = y1;

                (*num) = (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                i = i + T::one();
            }
        }
        self.valid_len = dest_len;
        mem::swap(&mut self.data, &mut temp);
        buffer.free(temp);
    }
}

#[cfg(test)]
mod tests {
    use conv_types::*;
    use super::super::super::*;
    use RealNumber;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
        where T: RealNumber
    {
        assert_eq!(left.len(), right.len());
        for i in 0..left.len() {
            if (left[i] - right[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?} at index {}", left, right, i);
            }
        }
    }

    #[test]
    fn interpolatei_sinc_test() {
        let len = 6;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatei(&mut buffer, &sinc as &RealFrequencyResponse<f32>, 2).unwrap();
        let result = time.magnitude();
        let expected = [0.16666667,
                        0.044658206,
                        0.16666667,
                        0.16666667,
                        0.16666667,
                        0.6220085,
                        1.1666667,
                        0.6220085,
                        0.16666667,
                        0.16666667,
                        0.16666667,
                        0.044658206];
        assert_eq_tol(&result[..], &expected, 1e-4);
    }

    #[test]
    fn interpolate_sinc_even_test() {
        let len = 6;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(&mut buffer, &sinc as &RealFrequencyResponse<f32>, 2 * len, 0.0).unwrap();
        let result = time.to_real();
        let expected = [0.00000, 0.04466, 0.00000, -0.16667, 0.00000, 0.62201, 1.00000, 0.62201,
                        0.00000, -0.16667, 0.00000, 0.04466];
        assert_eq_tol(&result[..], &expected, 1e-4);
    }

    #[test]
    fn interpolate_sinc_odd_test() {
        let len = 7;
        let mut time = vec!(0.0; len).to_real_time_vec();
        time[len / 2] = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(&mut buffer, &sinc as &RealFrequencyResponse<f32>, 2 * len, 0.0).unwrap();
        let result = time.to_real();
        let expected = [0.00000, 0.15856, 0.00000, -0.22913, 0.00000, 0.64199, 1.00000, 0.64199,
                        0.00000, -0.22913, -0.00000, 0.15856, 0.00000, -0.14286];
        assert_eq_tol(&result[..], &expected, 1e-4);
    }

    #[test]
    fn interpolatei_rc_test() {
        let len = 6;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        time[len] = 1.0;
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.4);
        let mut buffer = SingleBuffer::new();
        time.interpolatei(&mut buffer, &rc as &RealFrequencyResponse<f32>, 2).unwrap();
        let result = time.magnitude();
        let expected = [0.0,
                        0.038979173,
                        0.0000000062572014,
                        0.15530863,
                        0.000000015884869,
                        0.6163295,
                        1.0,
                        0.61632943,
                        0.0000000142918966,
                        0.15530863,
                        0.000000048099658,
                        0.038979173];
        assert_eq_tol(&result[..], &expected, 1e-4);
    }

    #[test]
    fn interpolatef_by_integer_sinc_even_test() {
        let len = 6;
        let mut time = vec!(0.0; len).to_real_time_vec();
        time[len / 2] = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatef(&mut buffer,
                          &sinc as &RealImpulseResponse<f32>,
                          2.0,
                          0.0,
                          len);
        let result = time.to_real();
        let expected = [0.00000, 0.04466, 0.00000, -0.16667, 0.00000, 0.62201, 1.00000, 0.62201,
                        0.00000, -0.16667, 0.00000, 0.04466];
        assert_eq_tol(&result[..], &expected, 0.1);
    }

    #[test]
    fn interpolatef_by_integer_sinc_odd_test() {
        let len = 7;
        let mut time = vec!(0.0; len).to_real_time_vec();
        time[len / 2] = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatef(&mut buffer,
                          &sinc as &RealImpulseResponse<f32>,
                          2.0,
                          0.0,
                          len);
        let result = time.to_real();
        let expected = [0.00000, 0.15856, 0.00000, -0.22913, 0.00000, 0.64199, 1.00000, 0.64199,
                        0.00000, -0.22913, -0.00000, 0.15856, 0.00000, -0.14286];
        assert_eq_tol(&result[..], &expected, 0.1);
    }

    #[test]
    fn interpolatef_by_fractional_sinc_test() {
        let len = 6;
        let mut time = vec!(0.0; len).to_real_time_vec();
        time[len / 2] = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatef(&mut buffer,
                          &sinc as &RealImpulseResponse<f32>,
                          13.0 / 6.0,
                          0.0,
                          len);
        let result = time.to_real();
        // Expected has been obtained with Octave: a = zeros(6,1); a(4) = 1; interpft(a, 13)
        let expected = [-2.7756e-17, 4.0780e-02, 2.0934e-02, -1.3806e-01, -1.1221e-01, 3.6167e-01,
                        9.1022e-01, 9.1022e-01, 3.6167e-01, -1.1221e-01, -1.3806e-01, 2.0934e-02,
                        4.0780e-02];
        assert_eq_tol(&result[..], &expected, 0.1);
    }

    #[test]
    fn interpolate_by_fractional_sinc_test() {
        let len = 6;
        let mut time = vec!(0.0; len).to_real_time_vec();
        time[len / 2] = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(&mut buffer,
                          &sinc as &RealFrequencyResponse<f32>,
                          13,
                          0.0).unwrap();
        let result = time.to_real();
        let expected = [-2.7756e-17, 4.0780e-02, 2.0934e-02, -1.3806e-01, -1.1221e-01, 3.6167e-01,
                        9.1022e-01, 9.1022e-01, 3.6167e-01, -1.1221e-01, -1.3806e-01, 2.0934e-02,
                        4.0780e-02];
        assert_eq_tol(&result[..], &expected, 0.1);
    }

    #[test]
    fn interpolate_by_fractional_sinc_real_data_test() {
        let len = 6;
        let mut time = vec!(0.0; len).to_real_time_vec();
        time[len / 2] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(&mut buffer,
                          &sinc as &RealFrequencyResponse<f32>,
                          13,
                          0.0).unwrap();
        let expected = [-2.7756e-17, 4.0780e-02, 2.0934e-02, -1.3806e-01, -1.1221e-01, 3.6167e-01,
                        9.1022e-01, 9.1022e-01, 3.6167e-01, -1.1221e-01, -1.3806e-01, 2.0934e-02,
                        4.0780e-02];
        assert_eq_tol(&time[..], &expected, 0.1);
    }

    #[test]
    fn interpolatef_delayed_sinc_test() {
        let len = 6;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatef(&mut buffer,
                          &sinc as &RealImpulseResponse<f32>,
                          2.0,
                          1.0,
                          len);
        let result = time.magnitude();
        let expected = [0.00000, 0.00000, 0.00000, 0.04466, 0.00000, 0.16667, 0.00000, 0.62201,
                        1.00000, 0.62201, 0.00000, 0.16667];
        assert_eq_tol(&result[..], &expected, 0.1);
    }


    #[test]
    fn interpolate_delayed_sinc_test() {
    	// We use different test data for `interpolate` then for `interpolatef`
    	// since the dirac impulse used in `interpolatef` does not work well
    	// with a FFT since it violates the Nyquist–Shannon sampling theorem.   
    	
        // time data in Octave: [fir1(5, 0.2)];
        let time = vec!(0.019827, 0.132513, 0.347660, 0.347660, 0.132513, 0.019827).to_real_time_vec();
        let len = time.len();
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(&mut buffer,
                          &sinc as &RealFrequencyResponse<f32>,
                          2 * len,
                          1.0).unwrap();
        let result = time.magnitude();
        // expected in Octave: interpft([time(2:end) 0], 12);
        let expected = [0.132513, 0.244227, 0.347660, 0.390094, 0.347660, 0.244227,
				        0.132513, 0.054953, 0.019827, 0.011546, 0.019827, 0.054953];
        assert_eq_tol(&result[..], &expected, 0.1);
    }

    #[test]
    fn decimatei_test() {
        let mut time = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
            .to_complex_time_vec();
        time.decimatei(2, 1);
        let expected = [2.0, 3.0, 6.0, 7.0, 10.0, 11.0];
        assert_eq_tol(&time[..], &expected, 0.1);
    }

    #[test]
    fn hermit_spline_test() {
        let mut time = vec![-1.0, -2.0, -1.0, 0.0, 1.0, 3.0, 4.0].to_real_freq_vec();
        let mut buffer = SingleBuffer::new();
        time.interpolate_hermite(&mut buffer, 4.0, 0.0);
        let expected = [-1.0000, -1.4375, -1.7500, -1.9375, -2.0000, -1.8906, -1.6250, -1.2969,
                        -1.0000, -0.7500, -0.5000, -0.2500, 0.0, 0.2344, 0.4583, 0.7031, 1.0000,
                        1.4375, 2.0000, 2.5625, 3.0000, 3.3203, 3.6042, 3.8359, 4.0];
        assert_eq_tol(&time[4..expected.len() - 4],
                      &expected[4..expected.len() - 4],
                      6e-2);
    }

    #[test]
    fn hermit_spline_test_linear_increment() {
        let mut time = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0].to_real_freq_vec();
        let mut buffer = SingleBuffer::new();
        time.interpolate_hermite(&mut buffer, 3.0, 0.0);
        let expected = [-3.0, -2.666, -2.333, -2.0, -1.666, -1.333, -1.0, -0.666, -0.333, 0.0,
                        0.333, 0.666, 1.0, 1.333, 1.666, 2.0, 2.333, 2.666, 3.0];
        assert_eq_tol(&time[..], &expected, 5e-3);
    }

    #[test]
    fn linear_test() {
        let mut time = vec![-1.0, -2.0, -1.0, 0.0, 1.0, 3.0, 4.0].to_real_freq_vec();
        let mut buffer = SingleBuffer::new();
        time.interpolate_lin(&mut buffer, 4.0, 0.0);
        let expected = [-1.0000, -1.2500, -1.5000, -1.7500, -2.0000, -1.7500, -1.5000, -1.2500,
                        -1.0000, -0.7500, -0.5000, -0.2500, 0.0, 0.2500, 0.5000, 0.7500, 1.0000,
                        1.5000, 2.0000, 2.5000, 3.0000, 3.2500, 3.5000, 3.7500, 4.0];
        assert_eq_tol(&time[..], &expected, 0.1);
    }
}
