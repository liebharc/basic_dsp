use super::super::{
    Buffer, BufferBorrow, ComplexOps, DataDomain, Domain, DspVec, ErrorReason, FloatIndex,
    FloatIndexMut, FrequencyToTimeDomainOperations, GenDspVec, InsertZerosOpsBuffered, MetaData,
    NoTradeBuffer, NumberSpace, PaddingOption, ResizeBufferedOps, ResizeOps, ScaleOps,
    TimeToFrequencyDomainOperations, ToComplexVector, ToDspVector, ToSliceMut, Vector, VoidResult,
};
use super::{create_shifted_copies, WrappingIterator};
use crate::conv_types::{RealFrequencyResponse, RealImpulseResponse};
use crate::inline_vector::InlineVector;
use crate::multicore_support::*;
use crate::numbers::*;
use crate::simd_extensions::*;
use crate::{array_to_complex, array_to_complex_mut, memcpy, Zero};
use std;
use std::ops::{Add, Mul};

/// Provides interpolation operations for real and complex data vectors.
pub trait InterpolationOps<S, T>
where
    S: ToSliceMut<T>,
    T: RealNumber,
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
    fn interpolatef<B>(
        &mut self,
        buffer: &mut B,
        function: &RealImpulseResponse<T>,
        interpolation_factor: T,
        delay: T,
        conv_len: usize,
    ) where
        B: for<'a> Buffer<'a, S, T>;

    /// Interpolates `self` with the convolution function `function` by the integer value
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
    fn interpolatei<B>(
        &mut self,
        buffer: &mut B,
        function: &RealFrequencyResponse<T>,
        interpolation_factor: u32,
    ) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Interpolates the signal in frequency domain by padding it with zeros.
    fn interpolate<B>(
        &mut self,
        buffer: &mut B,
        function: Option<&RealFrequencyResponse<T>>,
        target_points: usize,
        delay: T,
    ) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Interpolates the signal in frequency domain by padding it with zeros.
    /// This function preserves the shape of the signal in frequency domain.
    ///
    /// Calling this function is the same as calling `interpolate` with `None` as
    /// function and `0.0` as delay.
    fn interpft<B>(&mut self, buffer: &mut B, target_points: usize)
    where
        B: for<'a> Buffer<'a, S, T>;

    /// Decimates or downsamples `self`. `decimatei` is the inverse function to `interpolatei`.
    fn decimatei(&mut self, decimation_factor: u32, delay: u32);
}

fn interpolate_priv_scalar<T, TT>(
    temp: &mut [TT],
    data: &[TT],
    function: &RealImpulseResponse<T>,
    interpolation_factor: T,
    delay: T,
    conv_len: usize,
    multicore_settings: MultiCoreSettings,
) where
    T: RealNumber,
    TT: Zero + Mul<Output = TT> + Copy + Send + Sync + From<T> + Add<Output = TT>,
{
    Chunk::execute_with_range(
        Complexity::Large,
        &multicore_settings,
        temp,
        1,
        data,
        move |dest_range, range, data| {
            let mut i = range.start as isize;
            for num in dest_range {
                let center = T::from(i).unwrap() / interpolation_factor;
                let rounded = center.floor();
                let iter = WrappingIterator::new(
                    data,
                    rounded.to_isize().unwrap() - conv_len as isize - 1,
                    2 * conv_len + 1,
                );
                let mut sum = TT::zero();
                let mut j = -T::from(conv_len).unwrap() - (center - rounded) + delay;
                for c in iter {
                    sum = sum + c * TT::from(function.calc(j));
                    j = j + T::one();
                }
                (*num) = sum;
                i += 1;
            }
        },
    );
}

fn function_to_vectors<T>(
    function: &RealImpulseResponse<T>,
    conv_len: usize,
    complex_result: bool,
    interpolation_factor: usize,
    delay: T,
) -> InlineVector<GenDspVec<InlineVector<T>, T>>
where
    T: RealNumber,
{
    let mut result = InlineVector::with_capacity(interpolation_factor);
    for shift in 0..interpolation_factor {
        let offset = T::from(shift).unwrap() / T::from(interpolation_factor).unwrap();
        result.push(function_to_vector(
            function,
            conv_len,
            complex_result,
            offset,
            delay,
        ));
    }

    result
}

fn function_to_vector<T>(
    function: &RealImpulseResponse<T>,
    conv_len: usize,
    complex_result: bool,
    offset: T,
    delay: T,
) -> GenDspVec<InlineVector<T>, T>
where
    T: RealNumber,
{
    let step = if complex_result { 2 } else { 1 };
    let data_len = step * (2 * conv_len + 1);
    let mut imp_resp =
        InlineVector::of_size(T::zero(), data_len).to_gen_dsp_vec(complex_result, DataDomain::Time);
    let mut i = 0;
    let mut j = -(T::from(conv_len).unwrap() - T::one()) + delay;
    while i < data_len {
        let value = function.calc(j - offset);
        *imp_resp.data_mut(i) = value;
        i += step;
        j = j + T::one();
    }
    imp_resp
}

impl<S, T, N, D> DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    fn interpolate_priv_simd<Reg: SimdGeneric<T>, TT, C, CMut, RMul, RSum, B>(
        &mut self,
        _: RegType<Reg>,
        buffer: &mut B,
        function: &RealImpulseResponse<T>,
        interpolation_factor: usize,
        delay: T,
        conv_len: usize,
        new_len: usize,
        convert: C,
        convert_mut: CMut,
        simd_mul: RMul,
        simd_sum: RSum,
    ) where
        TT: Zero + Clone + From<T> + Copy + Add<Output = TT> + Mul<Output = TT> + Send + Sync,
        C: Fn(&[T]) -> &[TT],
        CMut: Fn(&mut [T]) -> &mut [TT],
        RMul: Fn(Reg, Reg) -> Reg + Sync,
        RSum: Fn(Reg) -> TT + Sync,
        B: for<'a> Buffer<'a, S, T>,
    {
        let data_len = self.len();
        let mut temp = buffer.borrow(new_len);
        {
            let step = if self.is_complex() { 2 } else { 1 };
            let number_of_shifts = Reg::LEN / step;
            let vectors = function_to_vectors(
                function,
                conv_len,
                self.is_complex(),
                interpolation_factor,
                delay,
            );
            let mut shifts = InlineVector::with_capacity(vectors.len() * number_of_shifts);
            for vector in &vectors[..] {
                let shifted_copies = create_shifted_copies(vector);
                for shift in shifted_copies.iter() {
                    shifts.push(shift.clone());
                }
            }

            let data = self.data.to_slice();
            let temp = temp.to_slice_mut();
            let dest = convert_mut(&mut temp[0..new_len]);
            let len = dest.len();
            let scalar_len = vectors[0].points() * interpolation_factor;

            let partition = Reg::calc_data_alignment_reqs(&data[0..data_len]);
            let left_points = partition.left / step;
            let simd = Reg::array_to_regs(partition.center(data));
            // Length of a SIMD reg relative to the length of type T
            // which is 1 for real numbers or 2 for complex numbers
            let simd_len_in_t = Reg::LEN / step;
            Chunk::execute_with_range(
                Complexity::Large,
                &self.multicore_settings,
                &mut dest[scalar_len..len - scalar_len],
                1,
                simd,
                move |dest_range, range, simd| {
                    let mut i = range.start + scalar_len;
                    for num in dest_range {
                        let rounded = (i + interpolation_factor - 1) / interpolation_factor;
                        let end = rounded + conv_len - left_points;
                        let simd_end = (end + simd_len_in_t - 1) / simd_len_in_t;
                        let simd_shift = end % simd_len_in_t;
                        let factor_shift = i % interpolation_factor;
                        // The reasoning for the next match is analog to the explanation in the
                        // create_shifted_copies function.
                        // We need the inverse of the mod unless we start with zero
                        let factor_shift =
                            (interpolation_factor - factor_shift) % interpolation_factor;
                        let selection = factor_shift * simd_len_in_t + simd_shift;
                        let shifted = &shifts[selection];
                        let mut sum = Reg::splat(T::zero());
                        let simd_iter = simd[simd_end - shifted.len()..simd_end].iter();
                        let iteration = simd_iter.zip(shifted.iter());
                        for (this, other) in iteration {
                            sum = sum + simd_mul(*this, *other);
                        }
                        (*num) = simd_sum(sum);
                        i += 1;
                    }
                },
            );

            let data = convert(&data[0..data_len]);
            for (i, num) in IndexedEdgeIteratorMut::new(dest, scalar_len, scalar_len) {
                (*num) = Self::interpolate_priv_simd_step(
                    i as usize,
                    interpolation_factor,
                    conv_len,
                    data,
                    &vectors[..],
                );
            }
        }
        self.valid_len = new_len;
        temp.trade(&mut self.data);
    }

    #[inline]
    fn interpolate_priv_simd_step<TT>(
        i: usize,
        interpolation_factor: usize,
        conv_len: usize,
        data: &[TT],
        vectors: &[GenDspVec<InlineVector<T>, T>],
    ) -> TT
    where
        TT: Zero + Clone + From<T> + Copy + Add<Output = TT> + Mul<Output = TT>,
    {
        let rounded = i / interpolation_factor;
        let iter =
            WrappingIterator::new(data, rounded as isize - conv_len as isize, 2 * conv_len + 1);
        let vector = &vectors[i % interpolation_factor];
        let step = if vector.is_complex() { 2 } else { 1 };
        let mut sum = TT::zero();
        let mut j = 0;
        for c in iter {
            sum = sum + c * TT::from(*vector.data(j));
            j += step;
        }
        sum
    }

    /// Applies a linear phase to a vector. Vector is assumed to be in frequency domain.
    /// This operation is equivalent with adding a dealy in time domain.
    fn apply_linear_phase(&mut self, delay: T) {
        let pi = T::PI();
        let two = T::one() + T::one();
        let points = self.len() / 2;
        let pos_points = points / 2;
        let neg_points = points - pos_points;
        let phase_inc = two * pi * delay / T::from(points).unwrap();
        {
            let len = self.len();
            let mut freq = (self.data_mut(2 * pos_points..len)).to_complex_freq_vec();
            // Negative frequencies
            let start = -T::from(neg_points).unwrap() * phase_inc;
            freq.multiply_complex_exponential(phase_inc, start);
        }
        {
            let mut freq = (self.data_mut(0..2 * pos_points)).to_complex_freq_vec();
            // Zero and psoitive frequencies
            let start = T::zero();
            freq.multiply_complex_exponential(phase_inc, start);
        }
    }

    fn interpolate_upsample(
        &mut self,
        function: Option<&RealFrequencyResponse<T>>,
        interpolation_factorf: T,
    ) {
        match function {
            None => {
                self.scale(interpolation_factorf);
            }
            Some(func) => {
                // Apply the window function
                self.multiply_function_priv(
                    func.is_symmetric(),
                    true,
                    interpolation_factorf,
                    |array| array_to_complex_mut(array),
                    func,
                    |f, x| Complex::<T>::new(f.calc(x), T::zero()),
                );
            }
        };
    }

    fn interpolate_downsample(&mut self, dest_points: usize) {
        let orig_len = self.len();
        let neg_points = dest_points / 2;
        let pos_points = dest_points - neg_points;
        let step = 2;
        {
            let copyrange = orig_len - step * neg_points..orig_len;
            memcpy(self.data.to_slice_mut(), copyrange, step * pos_points);
        }
        self.resize(step * (neg_points + pos_points))
            .expect("Shrinking should always succeed");
        self.scale(T::from(step * (neg_points + pos_points)).unwrap() / T::from(orig_len).unwrap());
    }
}

impl<S, T, N, D> InterpolationOps<S, T> for DspVec<S, T, N, D>
where
    DspVec<S, T, N, D>: InsertZerosOpsBuffered<S, T> + ScaleOps<T> + ResizeBufferedOps<S, T>,
    S: ToSliceMut<T> + ToComplexVector<S, T> + ToDspVector<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn interpolatef<B>(
        &mut self,
        buffer: &mut B,
        function: &RealImpulseResponse<T>,
        interpolation_factor: T,
        delay: T,
        conv_len: usize,
    ) where
        B: for<'a> Buffer<'a, S, T>,
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
        let new_len = (T::from(len).unwrap() * interpolation_factor)
            .round()
            .to_usize()
            .unwrap();
        let new_len = new_len + new_len % 2;
        if conv_len <= 202
            && new_len >= 2000
            && (interpolation_factor.round() - interpolation_factor).abs() < T::from(1e-6).unwrap()
        {
            let interpolation_factor = interpolation_factor.round().to_usize().unwrap();
            if self.is_complex() {
                return sel_reg!(self.interpolate_priv_simd::<T>(
                    buffer,
                    function,
                    interpolation_factor,
                    delay,
                    conv_len,
                    new_len,
                    |x| array_to_complex(x),
                    |x| array_to_complex_mut(x),
                    |x, y| x.mul_complex(y),
                    |x| x.sum_complex()
                ));
            } else {
                return sel_reg!(self.interpolate_priv_simd::<T>(
                    buffer,
                    function,
                    interpolation_factor,
                    delay,
                    conv_len,
                    new_len,
                    |x| x,
                    |x| x,
                    |x, y| x * y,
                    |x| x.sum_real()
                ));
            }
        } else if is_complex {
            let mut temp = buffer.borrow(new_len);
            {
                let data = self.data.to_slice();
                let data = &data[0..len];
                let temp = temp.to_slice_mut();
                let temp = array_to_complex_mut(&mut temp[0..new_len]);
                let data = array_to_complex(data);
                interpolate_priv_scalar(
                    temp,
                    data,
                    function,
                    interpolation_factor,
                    delay,
                    conv_len,
                    self.multicore_settings,
                );
            }
            temp.trade(&mut self.data);
        } else {
            let mut temp = buffer.borrow(new_len);
            {
                let data = self.data.to_slice();
                let data = &data[0..len];
                let temp = temp.to_slice_mut();
                interpolate_priv_scalar(
                    temp,
                    data,
                    function,
                    interpolation_factor,
                    delay,
                    conv_len,
                    self.multicore_settings,
                );
            }
            temp.trade(&mut self.data);
        }

        self.valid_len = new_len;
    }

    fn interpolatei<B>(
        &mut self,
        buffer: &mut B,
        function: &RealFrequencyResponse<T>,
        interpolation_factor: u32,
    ) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>,
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

        let len = self.len();
        let mut temp = buffer.borrow(len);
        // The next steps: fft, mul, ifft
        // However to keep the impl definition simpler and to avoid uncessary copies
        // we have to to be creative with where we store our signal and what's the buffer.
        {
            let complex = (self.data_mut(..)).to_complex_time_vec();
            let mut buffer = NoTradeBuffer::new(&mut temp[..]);
            complex.plain_fft(&mut buffer); // after this operation, our result is in `temp`. See also definition of `NoTradeBuffer`.
        }
        {
            let mut complex = (&mut temp[..]).to_complex_freq_vec();
            let interpolation_factorf = T::from(interpolation_factor).unwrap();
            complex.multiply_function_priv(
                function.is_symmetric(),
                true,
                interpolation_factorf,
                |array| array_to_complex_mut(array),
                function,
                |f, x| Complex::<T>::new(f.calc(x), T::zero()),
            );
            let mut buffer = NoTradeBuffer::new(self.data_mut(..));
            complex.plain_ifft(&mut buffer); // the result is now back in `self`.
        }
        let points = len / 2;
        self.scale(T::one() / T::from(points).unwrap());
        if !is_complex {
            // Convert vector back into real number space
            self.pure_complex_to_real_operation_inplace(|x, _arg| x.re, ());
        }

        Ok(())
    }

    fn interpft<B>(&mut self, buffer: &mut B, dest_points: usize)
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        self.interpolate(buffer, None, dest_points, T::zero())
            .expect("interpolate with no frequency response should never fail");
    }

    fn interpolate<B>(
        &mut self,
        buffer: &mut B,
        function: Option<&RealFrequencyResponse<T>>,
        dest_points: usize,
        delay: T,
    ) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        if function.is_some() && !function.unwrap().is_symmetric() && !self.is_complex() {
            return Err(ErrorReason::ArgumentFunctionMustBeSymmetric);
        }

        let delta_t = self.delta();
        let is_complex = self.is_complex();
        let orig_len = self.len();
        let dest_len = if is_complex {
            2 * dest_points
        } else {
            dest_points
        };
        let interpolation_factorf = T::from(dest_points).unwrap() / T::from(self.points()).unwrap();

        if !self.is_complex() {
            // Vector is always complex from here on (however the complex flag inside the vector
            // may not be set)
            self.zero_interleave_b(buffer, 2);
        }

        let complex_orig_len = self.len();
        let complex_dest_len = 2 * dest_points;
        let temp_len = std::cmp::max(complex_orig_len, complex_dest_len);
        self.resize_b(buffer, temp_len)?; // allocate space
        let mut temp = buffer.borrow(temp_len);
        {
            let complex = (self.data_mut(0..complex_orig_len)).to_complex_time_vec();
            let mut buffer = NoTradeBuffer::new(&mut temp[0..temp_len]);
            complex.plain_fft(&mut buffer); // after this operation, our result is in `temp`. See also definition of `NoTradeBuffer`.
        }
        {
            let mut complex = (&mut temp[0..temp_len]).to_complex_freq_vec();
            complex
                .resize(complex_orig_len)
                .expect("Shrinking should always succeed");
            let mut buffer = NoTradeBuffer::new(self.data_mut(0..temp_len));
            // Add the delay, which is a linear phase in frequency domain
            if delay != T::zero() {
                complex.apply_linear_phase(delay / delta_t);
            }

            if dest_len > orig_len {
                complex.zero_pad_b(&mut buffer, dest_points, PaddingOption::Center)?;
                // data is in `self` now, so we have to copy it back into `temp` so that
                // it finally ends up at the correct destination after `plain_ifft`
                complex
                    .data_mut(0..complex_dest_len)
                    .copy_from_slice(&buffer.borrow(complex_dest_len)[..]);
                complex.interpolate_upsample(function, interpolation_factorf);
            } else if dest_len < orig_len {
                complex.interpolate_downsample(dest_points);
            }

            complex.plain_ifft(&mut buffer); // the result is now back in `self`.
        }
        self.resize(complex_dest_len)
            .expect("Shrinking should always succeed");
        self.scale(T::one() / T::from(dest_points).unwrap());
        self.delta = delta_t / interpolation_factorf;
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
            let data = array_to_complex_mut(&mut data[0..len]);
            let decimation_factor = decimation_factor as usize;
            while i < points {
                data[j] = data[i];
                i += decimation_factor;
                j += 1;
            }
            self.valid_len = j * 2;
        } else {
            let data = &mut data[0..len];
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

#[cfg(test)]
mod tests {
    use super::super::super::*;
    use crate::conv_types::*;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
    where
        T: RealNumber,
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
        let mut time = vec![0.0; 2 * len].to_complex_time_vec();
        *time.data_mut(len) = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatei(&mut buffer, &sinc as &RealFrequencyResponse<f32>, 2)
            .unwrap();
        let result = time.magnitude();
        let expected = [
            0.16666667,
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
            0.044658206,
        ];
        assert_eq_tol(result.data(..), &expected, 1e-4);
    }

    #[test]
    fn interpolate_sinc_even_test() {
        let len = 6;
        let mut time = vec![0.0; 2 * len].to_complex_time_vec();
        *time.data_mut(len) = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(
            &mut buffer,
            Some(&sinc as &RealFrequencyResponse<f32>),
            2 * len,
            0.0,
        )
        .unwrap();
        let result = time.to_real();
        let expected = [
            0.00000, 0.04466, 0.00000, -0.16667, 0.00000, 0.62201, 1.00000, 0.62201, 0.00000,
            -0.16667, 0.00000, 0.04466,
        ];
        assert_eq_tol(result.data(..), &expected, 1e-4);
    }

    #[test]
    fn interpolate_sinc_odd_test() {
        let len = 7;
        let mut time = vec![0.0; len].to_real_time_vec();
        *time.data_mut(len / 2) = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(
            &mut buffer,
            Some(&sinc as &RealFrequencyResponse<f32>),
            2 * len,
            0.0,
        )
        .unwrap();
        let result = time.to_real();
        let expected = [
            0.00000, 0.15856, 0.00000, -0.22913, 0.00000, 0.64199, 1.00000, 0.64199, 0.00000,
            -0.22913, -0.00000, 0.15856, 0.00000, -0.14286,
        ];
        assert_eq_tol(result.data(..), &expected, 1e-4);
    }

    #[test]
    fn interpolatei_rc_test() {
        let len = 6;
        let mut time = vec![0.0; 2 * len].to_complex_time_vec();
        *time.data_mut(len) = 1.0;
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.4);
        let mut buffer = SingleBuffer::new();
        time.interpolatei(&mut buffer, &rc as &RealFrequencyResponse<f32>, 2)
            .unwrap();
        let result = time.magnitude();
        let expected = [
            0.0,
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
            0.038979173,
        ];
        assert_eq_tol(result.data(..), &expected, 1e-4);
    }

    #[test]
    fn interpolatef_by_integer_sinc_even_test() {
        let len = 6;
        let mut time = vec![0.0; len].to_real_time_vec();
        *time.data_mut(len / 2) = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatef(
            &mut buffer,
            &sinc as &RealImpulseResponse<f32>,
            2.0,
            0.0,
            len,
        );
        let result = time.to_real();
        let expected = [
            0.00000, 0.04466, 0.00000, -0.16667, 0.00000, 0.62201, 1.00000, 0.62201, 0.00000,
            -0.16667, 0.00000, 0.04466,
        ];
        assert_eq_tol(result.data(..), &expected, 0.1);
    }

    #[test]
    fn interpolatef_by_integer_sinc_odd_test() {
        let len = 7;
        let mut time = vec![0.0; len].to_real_time_vec();
        *time.data_mut(len / 2) = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatef(
            &mut buffer,
            &sinc as &RealImpulseResponse<f32>,
            2.0,
            0.0,
            len,
        );
        let result = time.to_real();
        let expected = [
            0.00000, 0.15856, 0.00000, -0.22913, 0.00000, 0.64199, 1.00000, 0.64199, 0.00000,
            -0.22913, -0.00000, 0.15856, 0.00000, -0.14286,
        ];
        assert_eq_tol(result.data(..), &expected, 0.1);
    }

    #[test]
    fn interpolatef_by_fractional_sinc_test() {
        let len = 6;
        let mut time = vec![0.0; len].to_real_time_vec();
        *time.data_mut(len / 2) = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatef(
            &mut buffer,
            &sinc as &RealImpulseResponse<f32>,
            13.0 / 6.0,
            0.0,
            len,
        );
        let result = time.to_real();
        // Expected has been obtained with Octave: a = zeros(6,1); a(4) = 1; interpft(a, 13)
        let expected = [
            -2.7756e-17,
            4.0780e-02,
            2.0934e-02,
            -1.3806e-01,
            -1.1221e-01,
            3.6167e-01,
            9.1022e-01,
            9.1022e-01,
            3.6167e-01,
            -1.1221e-01,
            -1.3806e-01,
            2.0934e-02,
            4.0780e-02,
        ];
        assert_eq_tol(result.data(..), &expected, 0.1);
    }

    #[test]
    fn interpolate_by_fractional_sinc_test() {
        let len = 6;
        let mut time = vec![0.0; len].to_real_time_vec();
        *time.data_mut(len / 2) = 1.0;
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(
            &mut buffer,
            Some(&sinc as &RealFrequencyResponse<f32>),
            13,
            0.0,
        )
        .unwrap();
        let result = time.to_real();
        let expected = [
            -2.7756e-17,
            4.0780e-02,
            2.0934e-02,
            -1.3806e-01,
            -1.1221e-01,
            3.6167e-01,
            9.1022e-01,
            9.1022e-01,
            3.6167e-01,
            -1.1221e-01,
            -1.3806e-01,
            2.0934e-02,
            4.0780e-02,
        ];
        assert_eq_tol(result.data(..), &expected, 0.1);
    }

    #[test]
    fn interpolate_by_fractional_sinc_real_data_test() {
        let len = 6;
        let mut time = vec![0.0; len].to_real_time_vec();
        *time.data_mut(len / 2) = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(
            &mut buffer,
            Some(&sinc as &RealFrequencyResponse<f32>),
            13,
            0.0,
        )
        .unwrap();
        let expected = [
            -2.7756e-17,
            4.0780e-02,
            2.0934e-02,
            -1.3806e-01,
            -1.1221e-01,
            3.6167e-01,
            9.1022e-01,
            9.1022e-01,
            3.6167e-01,
            -1.1221e-01,
            -1.3806e-01,
            2.0934e-02,
            4.0780e-02,
        ];
        assert_eq_tol(time.data(..), &expected, 0.1);
    }

    #[test]
    fn interpolatef_delayed_sinc_test() {
        let len = 6;
        let mut time = vec![0.0; 2 * len].to_complex_time_vec();
        *time.data_mut(len) = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolatef(
            &mut buffer,
            &sinc as &RealImpulseResponse<f32>,
            2.0,
            1.0,
            len,
        );
        let result = time.magnitude();
        let expected = [
            0.00000, 0.00000, 0.00000, 0.04466, 0.00000, 0.16667, 0.00000, 0.62201, 1.00000,
            0.62201, 0.00000, 0.16667,
        ];
        assert_eq_tol(result.data(..), &expected, 0.1);
    }

    #[test]
    fn interpolate_delayed_sinc_test() {
        // We use different test data for `interpolate` then for `interpolatef`
        // since the dirac impulse used in `interpolatef` does not work well
        // with a FFT since it violates the Nyquist–Shannon sampling theorem.

        // time data in Octave: [fir1(5, 0.2)];
        let time =
            vec![0.019827, 0.132513, 0.347660, 0.347660, 0.132513, 0.019827].to_real_time_vec();
        let len = time.len();
        let mut time = time.to_complex().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut buffer = SingleBuffer::new();
        time.interpolate(
            &mut buffer,
            Some(&sinc as &RealFrequencyResponse<f32>),
            2 * len,
            1.0,
        )
        .unwrap();
        let result = time.magnitude();
        // expected in Octave: interpft([time(2:end) 0], 12);
        let expected = [
            0.132513, 0.244227, 0.347660, 0.390094, 0.347660, 0.244227, 0.132513, 0.054953,
            0.019827, 0.011546, 0.019827, 0.054953,
        ];
        assert_eq_tol(result.data(..), &expected, 0.1);
    }

    #[test]
    fn interpolate_identity() {
        let mut time =
            vec![0.019827, 0.132513, 0.347660, 0.347660, 0.132513, 0.019827].to_real_time_vec();
        let len = time.len();
        let mut buffer = SingleBuffer::new();
        time.interpft(&mut buffer, len);
        // expected in Octave: interpft([time(2:end) 0], 12);
        let expected = [0.019827, 0.132513, 0.347660, 0.347660, 0.132513, 0.019827];
        assert_eq_tol(time.data(..), &expected, 0.1);
    }

    #[test]
    fn decimatei_test() {
        let mut time = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
            .to_complex_time_vec();
        time.decimatei(2, 1);
        let expected = [2.0, 3.0, 6.0, 7.0, 10.0, 11.0];
        assert_eq_tol(time.data(..), &expected, 0.1);
    }

    #[test]
    fn decimate_with_interpolate_test() {
        // Octave: fir1(12, 0.2)
        let time = vec![
            -2.6551e-03,
            1.5106e-04,
            1.6104e-02,
            5.9695e-02,
            1.2705e-01,
            1.9096e-01,
            2.1739e-01,
            1.9096e-01,
            1.2705e-01,
            5.9695e-02,
            1.6104e-02,
            1.5106e-04,
            -2.6551e-03,
        ]
        .to_real_time_vec();
        let len = time.len();
        let mut time = time.to_complex().unwrap();
        let mut buffer = SingleBuffer::new();
        let sinc: SincFunction<f32> = SincFunction::new();
        time.interpolate(
            &mut buffer,
            Some(&sinc as &RealFrequencyResponse<f32>),
            len / 2,
            0.0,
        )
        .unwrap();
        let result = time.magnitude();
        // Octave: interpft(time, 6)
        let expected = [
            2.0600e-03, 2.1088e-02, 1.5072e-01, 2.1024e-01, 8.0868e-02, 7.5036e-04,
        ];
        assert_eq_tol(result.data(..), &expected, 1e-4);
    }
}
