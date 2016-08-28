use super::definitions::{
    DataVec,
    TransRes};
use RealNumber;
use conv_types::{
    RealImpulseResponse,
    RealFrequencyResponse};
use super::{
    GenericDataVec,
    DataVecDomain,
    RealVectorOps,
    ComplexVectorOps,
    GenericVectorOps,
    RealTimeVector,
    RealFreqVector,
    TimeDomainOperations,
    FrequencyDomainOperations,
    ComplexTimeVector,
    ComplexFreqVector,
    ErrorReason,
    array_to_complex,
    array_to_complex_mut,
    round_len};
use num::complex::Complex;
use num::traits::Zero;
use super::convolution_impl::WrappingIterator;
use simd_extensions::*;
use std::ops::{Add, Mul};
use std::fmt::{Display, Debug};

/// Provides interpolation operations for real and complex data vectors.
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions are consistent.
/// However the actual implementation is lacking tests.
pub trait InterpolationOps<T> : DataVec<T>
    where T : RealNumber {
    /// Interpolates `self` with the convolution function `function` by the real value `interpolation_factor`.
    /// InterpolationOps is done in time domain and the argument `conv_len` can be used to balance accuracy
    /// and computational performance.
    /// A `delay` can be used to delay or phase shift the vector. The `delay` considers `self.delta()`.
    ///
    /// The complexity of this `interpolatef` is `O(self.points() * conv_len)`, while for `interpolatei` it's
    /// `O(self.points() * log(self.points()))`. If computational performance is important you should therefore decide
    /// how large `conv_len` needs to be to yield the desired accuracy. If you compare `conv_len` to `log(self.points)` you should
    /// get a feeling for the expected performance difference. More important is however to do a test
    /// run to compare the speed of `interpolatef` and `interpolatei`. Together with the information that
    /// changing the vectors size change `log(self.points()` but not `conv_len` gives the indication that `interpolatef`
    /// performs faster for larger vectors while `interpolatei` performs faster for smaller vectors.
    fn interpolatef(self, function: &RealImpulseResponse<T>, interpolation_factor: T, delay: T, conv_len: usize) -> TransRes<Self>;

    /// Interpolates `self` with the convolution function `function` by the interger value `interpolation_factor`.
    /// InterpolationOps is done in in frequency domain.
    ///
    /// See the description of `interpolatef` for some basic performance considerations.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `ArgumentFunctionMustBeSymmetric`: if `!self.is_complex() && !function.is_symmetric()` or in words if `self` is a real
    ///    vector and `function` is asymmetric. Converting the vector into a complex vector before the interpolation is one way
    ///    to resolve this error.
    fn interpolatei(self, function: &RealFrequencyResponse<T>, interpolation_factor: u32) -> TransRes<Self>;

    /// Decimates or downsamples `self`. `decimatei` is the inverse function to `interpolatei`.
    fn decimatei(self, decimation_factor: u32, delay: u32) -> TransRes<Self>;
}

/// Provides interpolation operations which are only applicable for real data vectors.
/// # Failures
/// All operations in this trait fail with `VectorMustBeReal` if the vector isn't in the real number space.
pub trait RealInterpolationOps<T> : DataVec<T>
    where T: RealNumber {

    /// Piecewise cubic hermite interpolation between samples.
    /// # Unstable
    /// Algorithm might need to be revised.
    /// This operation and `interpolate_lin` might be merged into one function with an additional argument in future.
    fn interpolate_hermite(self, interpolation_factor: T, delay: T) -> TransRes<Self>;

    /// Linear interpolation between samples.
    /// # Unstable
    /// This operation and `interpolate_hermite` might be merged into one function with an additional argument in future.
    fn interpolate_lin(self, interpolation_factor: T, delay: T) -> TransRes<Self>;
}

macro_rules! define_interpolation_impl {
    ($($data_type:ident,$reg:ident);*) => {
        $(
            impl GenericDataVec<$data_type> {
                fn interpolate_priv_scalar<T>(
                    temp: &mut [T], data: &[T],
                    function: &RealImpulseResponse<$data_type>,
                    interpolation_factor: $data_type, delay: $data_type,
                    conv_len: usize)
                        where T: Zero + Mul<Output=T> + Copy + Display + Debug + Send + Sync + From<$data_type> {
                    let mut i = 0;
                    for num in temp {
                        let center = i as $data_type / interpolation_factor;
                        let rounded = (center).floor();
                        let iter = WrappingIterator::new(&data, rounded as isize - conv_len as isize -1, 2 * conv_len + 1);
                        let mut sum = T::zero();
                        let mut j = -(conv_len as $data_type) - (center - rounded) + delay;
                        for c in iter {
                            sum = sum + c * T::from(function.calc(j));
                            j += 1.0;
                        }
                        (*num) = sum;
                        i += 1;
                    }
                }

                fn function_to_vectors(
                    function: &RealImpulseResponse<$data_type>,
                    conv_len: usize,
                    complex_result: bool,
                    interpolation_factor: usize,
                    delay: $data_type) -> Vec<GenericDataVec<$data_type>> {
                    let mut result = Vec::with_capacity(interpolation_factor);
                    for shift in 0..interpolation_factor {
                        let offset = shift as $data_type / interpolation_factor as $data_type;
                        result.push(Self::function_to_vector(
                            function,
                            conv_len,
                            complex_result,
                            offset,
                            delay));
                    }

                    result
                }

                fn function_to_vector(
                    function: &RealImpulseResponse<$data_type>,
                    conv_len: usize,
                    complex_result: bool,
                    offset: $data_type,
                    delay: $data_type) -> GenericDataVec<$data_type> {
                    let step = if complex_result { 2 } else { 1 };
                    let data_len = step * (2 * conv_len + 1);
                    let mut imp_resp = GenericDataVec::<$data_type>::new(
                        complex_result,
                        DataVecDomain::Time,
                        0.0,
                        data_len,
                        1.0);
                    let mut i = 0;
                    let mut j = -(conv_len as $data_type - 1.0) + delay;
                    while i < data_len {
                        let value = function.calc(j - offset);
                        imp_resp[i] = value;
                        i += step;
                        j += 1.0;
                    }
                    imp_resp
                }

                fn interpolate_priv_simd<T, C, CMut, RMul, RSum>(
                    mut self,
                    function: &RealImpulseResponse<$data_type>,
                    interpolation_factor: usize,
                    delay: $data_type,
                    conv_len: usize,
                    new_len: usize,
                    convert: C,
                    convert_mut: CMut,
                    simd_mul: RMul,
                    simd_sum: RSum) -> TransRes<Self>
                        where
                            T: Zero + Clone + From<$data_type> + Copy + Add<Output=T> + Mul<Output=T>,
                            C: Fn(&[$data_type]) -> &[T],
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            RMul: Fn($reg, $reg) -> $reg,
                            RSum: Fn($reg) -> T {
                    {
                        let step = if self.is_complex { 2 } else { 1 };
                        let number_of_shifts = $reg::len() / step;
                        let vectors = Self::function_to_vectors(
                            function,
                            conv_len,
                            self.is_complex,
                            interpolation_factor,
                            delay);
                        let mut shifts = Vec::with_capacity(vectors.len() * number_of_shifts);
                        for vector in &vectors {
                            let shifted_copies = Self::create_shifted_copies(&vector);
                            for shift in shifted_copies {
                                shifts.push(shift);
                            }
                        }

                        let len = self.len();
                        let data = convert(&self.data[0..len]);
                        let mut temp = temp_mut!(self, new_len);
                        let dest = convert_mut(&mut temp[0..new_len]);

                        let len = dest.len();
                        let scalar_len = vectors[0].points() * interpolation_factor;

                        let mut i = 0;
                        for num in &mut dest[0..scalar_len] {
                            (*num) =
                                Self::interpolate_priv_simd_step(
                                    i, interpolation_factor, conv_len,
                                    data, &vectors);
                            i += 1;
                        }

                        let (scalar_left, _, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..self.data.len()]);
                        let simd = $reg::array_to_regs(&self.data[scalar_left..vectorization_length]);
                        // Length of a SIMD reg relative to the length of type T
                        // which is 1 for real numbers or 2 for complex numbers
                        let simd_len_in_t = $reg::len() / step;
                        for num in &mut dest[scalar_len .. len - scalar_len] {
                            let rounded = (i + interpolation_factor - 1) / interpolation_factor;
                            let end = (rounded + conv_len) as usize;
                            let simd_end = (end + simd_len_in_t - 1) / simd_len_in_t;
                            let simd_shift = end % simd_len_in_t;
                            let factor_shift = i % interpolation_factor;
                            // The reasoning for the next match is analog to the explanation in the
                            // create_shifted_copies function.
                            // We need the inverse of the mod unless we start with zero
                            let factor_shift = match factor_shift {
                                0 => 0,
                                x => interpolation_factor - x
                            };
                            let selection = factor_shift * simd_len_in_t + simd_shift;
                            let shifted = &shifts[selection];
                            let mut sum = $reg::splat(0.0);
                            let simd_iter = simd[simd_end - shifted.len() .. simd_end].iter();
                            let iteration =
                                simd_iter
                                .zip(shifted);
                            for (this, other) in iteration {
                                sum = sum + simd_mul(*this, *other);
                            }
                            (*num) = simd_sum(sum);
                            i += 1;
                        }

                        i = len - scalar_len;
                        for num in &mut dest[len-scalar_len..len] {
                            (*num) =
                                Self::interpolate_priv_simd_step(
                                    i, interpolation_factor, conv_len,
                                    data, &vectors);
                            i += 1;
                        }
                    }
                    self.valid_len = new_len;
                    Ok(self.swap_data_temp())
                }

                #[inline]
                fn interpolate_priv_simd_step<T>(
                    i: usize,
                    interpolation_factor: usize,
                    conv_len: usize,
                    data: &[T],
                    vectors: &Vec<GenericDataVec<$data_type>>) -> T
                        where
                            T: Zero + Clone + From<$data_type> + Copy + Add<Output=T> + Mul<Output=T> {
                    let rounded = i / interpolation_factor;
                    let iter = WrappingIterator::new(&data, rounded as isize - conv_len as isize, 2 * conv_len + 1);
                    let vector = &vectors[i % interpolation_factor];
                    let step = if vector.is_complex() { 2 } else { 1 };
                    let mut sum = T::zero();
                    let mut j = 0;
                    for c in iter {
                        sum = sum + c * T::from(vector[j]);
                        j += step;
                    }
                    sum
                }
            }

            impl InterpolationOps<$data_type> for GenericDataVec<$data_type> {
                fn interpolatef(mut self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, conv_len: usize) -> TransRes<Self> {
                    {
                        let delay = delay / self.delta;
                        let len = self.len();
                        let points_half = self.points() / 2;
                        let conv_len =
                            if conv_len > points_half {
                                points_half
                            } else {
                                conv_len
                            };
                        let is_complex = self.is_complex();
                        let new_len = (len as $data_type * interpolation_factor).round() as usize;
                        let new_len = new_len + new_len % 2;
                        if conv_len <= 202 && new_len >= 2000 &&
                            (interpolation_factor.round() - interpolation_factor).abs() < 1e-6 {
                            let interpolation_factor = interpolation_factor.round() as usize;
                            if self.is_complex {
                                return self.interpolate_priv_simd(
                                    function,
                                    interpolation_factor,
                                    delay,
                                    conv_len,
                                    new_len,
                                    |x| array_to_complex(x),
                                    |x| array_to_complex_mut(x),
                                    |x,y| x.mul_complex(y),
                                    |x| x.sum_complex())
                            } else {
                                return self.interpolate_priv_simd(
                                    function,
                                    interpolation_factor,
                                    delay,
                                    conv_len,
                                    new_len,
                                    |x| x,
                                    |x| x,
                                    |x,y| x * y,
                                    |x| x.sum_real())
                            }
                        }
                        else if is_complex {
                            let data = &self.data[0..len];
                            let temp = temp_mut!(self, new_len);
                            let temp = array_to_complex_mut(temp);
                            let data = array_to_complex(data);
                            Self::interpolate_priv_scalar(
                                temp, data,
                                function,
                                interpolation_factor, delay, conv_len);
                        }
                        else {
                            let data = &self.data[0..len];
                            let temp = temp_mut!(self, new_len);
                            Self::interpolate_priv_scalar(
                                temp, data,
                                function,
                                interpolation_factor, delay, conv_len);
                        }

                        self.valid_len = new_len;
                    }
                    Ok(self.swap_data_temp())
                }

                fn interpolatei(self, function: &RealFrequencyResponse<$data_type>, interpolation_factor: u32) -> TransRes<Self> {
                    if interpolation_factor <= 1 {
                        return Ok(self);
                    }
                    reject_if!(self, !function.is_symmetric() &&!self.is_complex , ErrorReason::ArgumentFunctionMustBeSymmetric);
                    let is_complex = self.is_complex;
                    let freq = try! {
                        Ok(self)
                        .and_then(|v|v.zero_interleave(interpolation_factor))
                        .and_then(|v|v.fft())
                    };
                    let points = freq.points();
                    let interpolation_factorf = interpolation_factor as $data_type;
                    Ok(freq)
                    .and_then(|v| {
                        Ok(v.multiply_function_priv(
                                        function.is_symmetric(),
                                        interpolation_factorf,
                                        |array|array_to_complex_mut(array),
                                        function,
                                        |f,x|Complex::<$data_type>::new(f.calc(x), 0.0)))
                    })
                    .and_then(|v|v.ifft_shift())
                    .and_then(|v|v.plain_ifft())
                    .and_then(|v|v.real_scale(1.0 / points as $data_type))
                    .and_then(|v| {
                        if is_complex { Ok(v) } else { v.to_real() }
                    })
                }

                fn decimatei(mut self, decimation_factor: u32, delay: u32) -> TransRes<Self> {
                    {
                        let mut i = delay as usize;
                        let mut j = 0;
                        let len = self.points();
                        let is_complex = self.is_complex();
                        if is_complex {
                            let mut data = array_to_complex_mut(&mut self.data);
                            let decimation_factor = decimation_factor as usize;
                            while i < len {
                                data[j] = data[i];
                                i += decimation_factor;
                                j += 1;
                            }
                            self.valid_len = j * 2;
                        }
                        else {
                            let mut data = &mut self.data;
                            let decimation_factor = decimation_factor as usize;
                            while i < len {
                                data[j] = data[i];
                                i += decimation_factor;
                                j += 1;
                            }
                            self.valid_len = j;
                        }
                    }

                    Ok(self)
                }
            }

            impl RealInterpolationOps<$data_type> for GenericDataVec<$data_type> {
                fn interpolate_lin(mut self, interpolation_factor: $data_type, delay: $data_type) -> TransRes<Self> {
                    {
                        assert_real!(self);
                        let data_len = self.len();
                        let dest_len = ((data_len - 1) as $data_type * interpolation_factor).round() as usize + 1;
                        let data = &self.data[0..data_len];
                        let dest = temp_mut!(self, dest_len);
                        let mut i = 0.0;

                        for num in &mut dest[0..dest_len-1] {
                            let rounded = i / interpolation_factor + delay;
                            let beforef = rounded.floor();
                            let before = beforef as usize;
                            let next = before + 1;
                            let x0 = beforef;
                            let y0 = data[before];
                            let y1 = data[next];
                            let x = rounded;
                            (*num) =  y0 + (y1 - y0) * (x - x0);
                            i += 1.0;
                        }
                        dest[dest_len-1] = data[data_len - 1];
                        self.valid_len = dest_len;
                    }
                    Ok(self.swap_data_temp())
                }

                fn interpolate_hermite(mut self, interpolation_factor: $data_type, delay: $data_type) -> TransRes<Self> {
                    {
                        assert_real!(self);
                        // Literature: http://paulbourke.net/miscellaneous/interpolation/
                        let data_len = self.len();
                        let dest_len = ((data_len - 1) as $data_type * interpolation_factor).round() as usize + 1;
                        let data = &self.data[0..data_len];
                        let dest = temp_mut!(self, dest_len);
                        let mut i = 0.0;
                        let start = ((1.0 - delay) * interpolation_factor).ceil() as usize;
                        let end = start + 1;
                        for num in &mut dest[0..start] {
                            let rounded = i / interpolation_factor + delay;
                            let beforef = rounded.floor();
                            let before = beforef as usize;
                            assert_eq!(before, 0);
                            let next = before + 1;
                            let next_next = next + 1;
                            let y1 = data[before];
                            let y2 = data[next];
                            let y3 = data[next_next];
                            let x =  rounded - beforef;
                            let y0 = y1 - (y2 - y1);
                            let x2 = x * x;
                            let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                            let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                            let a2 = -0.5 * y0 + 0.5 * y2;
                            let a3 = y1;

                            (*num) =  (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                            i += 1.0;
                        }

                        for num in &mut dest[start..dest_len-end] {
                            let rounded = i / interpolation_factor + delay;
                            let beforef = rounded.floor();
                            let before = beforef as usize;
                            let before_before = before - 1;
                            let next = before + 1;
                            let next_next = next + 1;
                            let y0 = data[before_before];
                            let y1 = data[before];
                            let y2 = data[next];
                            let y3 = data[next_next];
                            let x =  rounded - beforef;
                            let x2 = x * x;
                            let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                            let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                            let a2 = -0.5 * y0 + 0.5 * y2;
                            let a3 = y1;

                            (*num) =  (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                            i += 1.0;
                        }

                        for num in &mut dest[dest_len-end..dest_len] {
                            let rounded = i / interpolation_factor + delay;
                            let beforef = rounded.floor();
                            let before = beforef as usize;
                            assert!(before + 2 >= data_len);
                            let before_before = before - 1;
                            let y0 = data[before_before];
                            let y1 = data[before];
                            let y2 = if before < data_len - 1 { data[before + 1] } else { y1 + (y1 - y0) };
                            let y3 = if before < data_len - 2 { data[before + 2] } else { y2 + (y2 - y1) };
                            let x =  rounded - beforef;
                            let x2 = x * x;
                            let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                            let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                            let a2 = -0.5 * y0 + 0.5 * y2;
                            let a3 = y1;

                            (*num) =  (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                            i += 1.0;
                        }
                        self.valid_len = dest_len;
                    }
                    Ok(self.swap_data_temp())
                }
            }
        )*
    }
}
define_interpolation_impl!(f32, Reg32; f64, Reg64);

macro_rules! define_interpolation_forward {
    ($($name:ident, $data_type:ident);*) => {
        $(
            impl InterpolationOps<$data_type> for $name<$data_type> {
                fn interpolatef(self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, len: usize) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().interpolatef(function, interpolation_factor, delay, len))
                }

                fn interpolatei(self, function: &RealFrequencyResponse<$data_type>, interpolation_factor: u32) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().interpolatei(function, interpolation_factor))
                }

                fn decimatei(self, decimation_factor: u32, delay: u32) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().decimatei(decimation_factor, delay))
                }
            }
        )*
    }
}

macro_rules! define_real_only_interpolation_forward {
    ($($name:ident, $data_type:ident);*) => {
        $(
            impl RealInterpolationOps<$data_type> for $name<$data_type> {
                fn interpolate_lin(self, interpolation_factor: $data_type, delay: $data_type) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().interpolate_lin(interpolation_factor, delay))
                }

                fn interpolate_hermite(self, interpolation_factor: $data_type, delay: $data_type) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().interpolate_hermite(interpolation_factor, delay))
                }
            }
        )*
    }
}

define_interpolation_forward!(
    RealTimeVector, f32; RealTimeVector, f64;
    ComplexTimeVector, f32; ComplexTimeVector, f64;
    RealFreqVector, f32; RealFreqVector, f64;
    ComplexFreqVector, f32; ComplexFreqVector, f64
);

define_real_only_interpolation_forward!(
    RealTimeVector, f32; RealTimeVector, f64;
    RealFreqVector, f32; RealFreqVector, f64
);

#[cfg(test)]
mod tests {
    use vector_types::*;
    use num::complex::Complex32;
    use conv_types::*;
    use RealNumber;
    use std::fmt::Debug;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
        where T: RealNumber + Debug {
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
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let result = time.interpolatei(&sinc as &RealFrequencyResponse<f32>, 2).unwrap();
        let result = result.magnitude().unwrap();
        let expected =
            [0.16666667, 0.044658206, 0.16666667, 0.16666667, 0.16666667, 0.6220085,
             1.1666667, 0.6220085, 0.16666667, 0.16666667, 0.16666667, 0.044658206];
        assert_eq_tol(result.real(0..), &expected, 1e-4);
    }

    #[test]
    fn interpolatei_rc_test() {
        let len = 6;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        time[len] = 1.0;
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.4);
        let result = time.interpolatei(&rc as &RealFrequencyResponse<f32>, 2).unwrap();
        let result = result.magnitude().unwrap();
        let expected =
            [0.0, 0.038979173, 0.0000000062572014, 0.15530863, 0.000000015884869, 0.6163295,
             1.0, 0.61632943, 0.0000000142918966, 0.15530863, 0.000000048099658, 0.038979173];
        assert_eq_tol(result.real(0..), &expected, 1e-4);
    }

    #[test]
    fn interpolatef_sinc_test() {
        let len = 6;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let result = time.interpolatef(&sinc as &RealImpulseResponse<f32>, 2.0, 0.0, len).unwrap();
        let result = result.magnitude().unwrap();
        let expected =
            [0.00000, 0.04466, 0.00000, 0.16667, 0.00000, 0.62201,
             1.00000, 0.62201, 0.00000, 0.16667, 0.00000, 0.04466];
        assert_eq_tol(result.real(0..), &expected, 0.1);
    }

    #[test]
    fn interpolatef_delayed_sinc_test() {
        let len = 6;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let result = time.interpolatef(&sinc as &RealImpulseResponse<f32>, 2.0, 1.0, len).unwrap();
        let result = result.magnitude().unwrap();
        let expected =
            [0.00000, 0.00000, 0.00000, 0.04466, 0.00000, 0.16667,
             0.00000, 0.62201, 1.00000, 0.62201, 0.00000, 0.16667];
        assert_eq_tol(result.real(0..), &expected, 0.1);
    }

    #[test]
    fn decimatei_test() {
        let time = ComplexTimeVector32::from_interleaved(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        let result = time.decimatei(2, 1).unwrap();
        let expected = [2.0, 3.0, 6.0, 7.0, 10.0, 11.0];
        assert_eq_tol(result.interleaved(0..), &expected, 0.1);
    }

    #[test]
    fn hermit_spline_test() {
        let time = RealFreqVector32::from_array(&[-1.0, -2.0, -1.0, 0.0, 1.0, 3.0, 4.0]);
        let result = time.interpolate_hermite(4.0, 0.0).unwrap();
        let expected = [
            -1.0000, -1.4375, -1.7500, -1.9375, -2.0000, -1.8906, -1.6250, -1.2969,
            -1.0000, -0.7500, -0.5000, -0.2500, 0.0, 0.2344, 0.4583, 0.7031,
            1.0000, 1.4375, 2.0000, 2.5625, 3.0000, 3.3203, 3.6042, 3.8359, 4.0];
        assert_eq_tol(
            &result.real(0..)[4..expected.len()-4],
            &expected[4..expected.len()-4],
            6e-2);
    }

    #[test]
    fn hermit_spline_test_linear_increment() {
        let time = RealFreqVector32::from_array(&[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let result = time.interpolate_hermite(3.0, 0.0).unwrap();
        let expected = [
            -3.0, -2.666, -2.333, -2.0, -1.666, -1.333, -1.0, -0.666, -0.333, 0.0,
            0.333, 0.666, 1.0, 1.333, 1.666, 2.0, 2.333, 2.666, 3.0];
        assert_eq_tol(result.real(0..), &expected, 5e-3);
    }

    #[test]
    fn linear_test() {
        let time = RealFreqVector32::from_array(&[-1.0, -2.0, -1.0, 0.0, 1.0, 3.0, 4.0]);
        let result = time.interpolate_lin(4.0, 0.0).unwrap();
        let expected = [
            -1.0000, -1.2500, -1.5000, -1.7500, -2.0000, -1.7500, -1.5000, -1.2500,
            -1.0000, -0.7500, -0.5000, -0.2500, 0.0, 0.2500, 0.5000, 0.7500,
             1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.2500, 3.5000, 3.7500, 4.0];
        assert_eq_tol(result.real(0..), &expected, 0.1);
    }
}
