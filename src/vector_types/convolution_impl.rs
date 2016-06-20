use super::definitions::{
    DataVector,
    VecResult,
    ErrorReason,
    DataVectorDomain};
use conv_types::*;
use RealNumber;
use num::traits::Zero;
use std::ops::{Add, Mul};
use std::fmt::Display;
use num::complex::Complex;
use simd_extensions::*;
use super::{
    GenericDataVector,
    RealFreqVector,
    RealTimeVector,
    ComplexFreqVector,
    ComplexTimeVector,
    round_len};

/// Provides a convolution operation for data vectors. 
pub trait Convolution<T, C> : DataVector<T> 
    where T : RealNumber {
    /// Convolves `self` with the convolution function `impulse_response`. For performance consider to 
    /// to use `FrequencyMultiplication` instead of this operation depending on `len`.
    ///
    /// An optimized convolution algorithm is used if  `1.0 / ratio` is an integer (inside a `1e-6` tolerance) 
    /// and `len` is smaller than a threshold (`202` right now).
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMustBeComplex`: if `self` is in real number space but `impulse_response` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    fn convolve(self, impulse_response: C, ratio: T, len: usize) -> VecResult<Self>;
}

/// Provides a convolution operation for data vectors with data vectors.
pub trait VectorConvolution<T> : DataVector<T> 
    where T : RealNumber {
    /// Convolves `self` with the convolution function `impulse_response`. For performance it's recommended 
    /// to use multiply both vectors in frequency domain instead of this operation.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    /// 2. `VectorMetaDataMustAgree`: in case `self` and `impulse_response` are not in the same number space and same domain.
    /// 3. `InvalidArgumentLength`: if `self.points() < impulse_response.points()`.
    fn convolve_vector(self, impulse_response: &Self) -> VecResult<Self>;
}

/// Provides a frequency response multiplication operation for data vectors.
pub trait FrequencyMultiplication<T, C> : DataVector<T> 
    where T : RealNumber {
    /// Mutiplies `self` with the frequency response function `frequency_response`.
    /// 
    /// In order to multiply a vector with another vector in frequency response use `multiply_vector`.
    /// # Assumptions
    /// The operation assumes that the vector contains a full spectrum centered at 0 Hz. If half a spectrum
    /// or a fft shifted spectrum is provided the operation will come back with invalid results.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMustBeComplex`: if `self` is in real number space but `frequency_response` is in complex number space.
    /// 2. `VectorMustBeInFreqDomain`: if `self` is in time domain.
    fn multiply_frequency_response(self, frequency_response: C, ratio: T) -> VecResult<Self>;
}

macro_rules! add_conv_impl{
    ($($data_type:ident, $reg:ident);*) => {
        $(
            impl<'a> Convolution<$data_type, &'a RealImpulseResponse<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &RealImpulseResponse<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    assert_time!(self);
                    if !self.is_complex {
                        let ratio_inv = 1.0 / ratio;
                        if len <= 202 && self.len() > 2000 && (ratio_inv.round() - ratio_inv).abs() < 1e-6 && ratio > 0.5 {
                            let mut imp_resp = ComplexTimeVector::<$data_type>::from_constant_with_delta(
                                Complex::<$data_type>::zero(), 
                                (2 * len + 1) * ratio as usize, 
                                self.delta());
                            let mut i = 0;
                            let mut j = -(len as $data_type);
                            while i < imp_resp.len() {
                                let value = function.calc(j * ratio_inv);
                                imp_resp[i] = value;
                                i += 1;
                                j += 1.0;
                            }
                            
                            return self.convolve_vector(&imp_resp.to_gen_borrow());
                        }
                        
                        Ok(self.convolve_function_priv(
                                ratio,
                                len,
                                |data|data,
                                |temp|temp,
                                |x|function.calc(x)
                            ))
                    } else {
                        let ratio_inv = 1.0 / ratio;
                        if len <= 202 && self.len() > 2000 && (ratio_inv.round() - ratio_inv).abs() < 1e-6 && ratio > 0.5 {
                            let mut imp_resp = ComplexTimeVector::<$data_type>::from_constant_with_delta(Complex::<$data_type>::zero(), (2 * len + 1) * ratio as usize, self.delta());
                            let mut i = 0;
                            let mut j = -(len as $data_type);
                            while i < imp_resp.len() {
                                let value = function.calc(j * ratio_inv);
                                imp_resp[i] = value;
                                i += 2;
                                j += 1.0;
                            }
                            
                            return self.convolve_vector(&imp_resp.to_gen_borrow());
                        }
                        
                        Ok(self.convolve_function_priv(
                            ratio,
                            len,
                            |data|Self::array_to_complex(data),
                            |temp|Self::array_to_complex_mut(temp),
                            |x|Complex::<$data_type>::new(function.calc(x), 0.0)
                        ))
                    }
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexImpulseResponse<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &ComplexImpulseResponse<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    assert_complex!(self);
                    assert_time!(self);
                    
                    let ratio_inv = 1.0 / ratio;
                    if len <= 202 && self.len() > 2000 && (ratio_inv.round() - ratio_inv).abs() < 1e-6 && ratio > 0.5 {
                        let mut imp_resp = ComplexTimeVector::<$data_type>::from_constant_with_delta(Complex::<$data_type>::zero(), (2 * len + 1) * ratio as usize, self.delta());
                        let mut i = 0;
                        let mut j = -(len as $data_type);
                        while i < imp_resp.len() {
                            let value = function.calc(j * ratio_inv);
                            imp_resp[i] = value.re;
                            i += 2;
                            imp_resp[i] = value.im;
                            i += 1;
                            j += 1.0;
                        }
                        
                        return self.convolve_vector(&imp_resp.to_gen_borrow());
                    }
                    
                    Ok(self.convolve_function_priv(
                            ratio,
                            len,
                            |data|Self::array_to_complex(data),
                            |temp|Self::array_to_complex_mut(temp),
                            |x|function.calc(x)
                        ))
                }
            }
            
            impl GenericDataVector<$data_type> {
                fn convolve_function_priv<T,C,CMut,F>(
                    mut self, 
                    ratio: $data_type,
                    conv_len: usize,
                    convert: C,
                    convert_mut: CMut,
                    fun: F) -> Self
                        where 
                            C: Fn(&[$data_type]) -> &[T],
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            F: Fn($data_type)->T,
                            T: Zero + Mul<Output=T> + Copy + Display
                {
                    {
                        let len = self.len();
                        let mut temp = temp_mut!(self, len);
                        let complex = convert(&self.data[0..len]);
                        let dest = convert_mut(&mut temp[0..len]);
                        let len = complex.len();
                        let mut i = 0;
                        let conv_len =
                            if conv_len > len {
                                len
                            } else {
                                conv_len
                            };
                        let sconv_len = conv_len as isize;
                        for num in dest {
                            let iter = WrappingIterator::new(complex, i - sconv_len - 1, 2 * conv_len + 1);
                            let mut sum = T::zero();
                            let mut j = -(conv_len as $data_type);
                            for c in iter {
                                sum = sum + c * fun(-j * ratio);
                                j += 1.0;
                            }
                            (*num) = sum;
                            i += 1;
                        }
                    }
                    self.swap_data_temp()
                }
            }

            impl<'a> FrequencyMultiplication<$data_type, &'a ComplexFrequencyResponse<$data_type>> for GenericDataVector<$data_type> {
                fn multiply_frequency_response(self, function: &ComplexFrequencyResponse<$data_type>, ratio: $data_type) -> VecResult<Self> {
                    assert_complex!(self);
                    assert_freq!(self);
                    Ok(self.multiply_function_priv(
                                    function.is_symmetric(),
                                    ratio,
                                    |array|Self::array_to_complex_mut(array),
                                    function,
                                    |f,x|f.calc(x)))
                }
            }
            
            impl<'a> FrequencyMultiplication<$data_type, &'a RealFrequencyResponse<$data_type>> for GenericDataVector<$data_type> {
                fn multiply_frequency_response(self, function: &RealFrequencyResponse<$data_type>, ratio: $data_type) -> VecResult<Self> {
                    assert_freq!(self);
                    if self.is_complex {
                        Ok(self.multiply_function_priv(
                                        function.is_symmetric(),
                                        ratio,
                                        |array|Self::array_to_complex_mut(array),
                                        function,
                                        |f,x|Complex::<$data_type>::new(f.calc(x), 0.0)))
                    }
                    else {
                        Ok(self.multiply_function_priv(
                                        function.is_symmetric(),
                                        ratio,
                                        |array|array,
                                        function,
                                        |f,x|f.calc(x)))
                    }
                }
            }
            
            impl VectorConvolution<$data_type> for GenericDataVector<$data_type> {
                fn convolve_vector(self, vector: &Self) -> VecResult<Self> {
                    assert_meta_data!(self, vector);
                    assert_time!(self);
                    reject_if!(self, self.points() < vector.points(), ErrorReason::InvalidArgumentLength);
                    // The values in this condition are nothing more than a 
                    // ... guess. The reasoning is basically this:
                    // For the SIMD operation we need to clone `vector` several
                    // times and this only is worthwhile if `vector.len() << self.len()`
                    // where `<<` means "significant smaller".
                    if self.len() > 1000 && vector.len() <= 202 {
                        self.convolve_vector_simd(vector)
                    }
                    else {
                        self.convolve_vector_scalar(vector)
                    }
                }
            }
            
            impl GenericDataVector<$data_type> {
                fn convolve_vector_scalar(mut self, vector: &Self) -> VecResult<Self> {
                    let points = self.points();
                    let other_points = vector.points();
                    let (other_start, other_end, full_conv_len, conv_len) =
                            if other_points > points {
                                let center = other_points / 2;
                                let conv_len = points / 2;
                                (center - conv_len, center + conv_len, points, conv_len)
                            } else {
                                (0, other_points, other_points, other_points - other_points / 2)
                            };
                    if self.is_complex {
                        {
                            let len = self.len();
                            let other = Self::array_to_complex(&vector.data[0..vector.len()]);
                            let temp = temp_mut!(self, len);
                            let complex = Self::array_to_complex(&self.data[0..len]);
                            let dest = Self::array_to_complex_mut(&mut temp[0..len]);
                            let other_iter = &other[other_start .. other_end];
                            let conv_len = conv_len as isize;
                            let mut i = 0;
                            for num in dest {
                                *num = Self::convolve_iteration(complex, other_iter, i, conv_len, full_conv_len);
                                i += 1;
                            }
                        }
                        Ok(self.swap_data_temp())
                    } else {
                        {
                            let len = self.len();
                            let other = &vector.data[0..vector.len()];
                            let data = &self.data[0..len];
                            let temp = temp_mut!(self, len);
                            let dest = &mut temp[0..len];
                            let other_iter = &other[other_start .. other_end];
                            let conv_len = conv_len as isize;
                            let mut i = 0;
                            for num in dest {
                                *num = Self::convolve_iteration(data, other_iter, i, conv_len, full_conv_len);
                                i += 1;
                            }
                        }
                        Ok(self.swap_data_temp())
                    }
                }
                
                #[inline]
                fn convolve_iteration<T>(data: &[T], other_iter: &[T], i: isize, conv_len: isize, full_conv_len: usize) -> T 
                    where T: Zero + Clone + Copy + Add<Output=T> + Mul<Output=T> {
                    let data_iter = ReverseWrappingIterator::new(data, i + conv_len, full_conv_len);
                    let mut sum = T::zero();
                    let iteration = 
                        data_iter
                        .zip(other_iter);
                    for (this, other) in iteration {
                        sum = sum + this * (*other);
                    }
                    sum
                }
                
                fn convolve_vector_simd(self, vector: &Self) -> VecResult<Self> {
                    if self.is_complex {
                        self.convolve_vector_simd_impl(
                            vector,
                            |x| Self::array_to_complex(x),
                            |x| Self::array_to_complex_mut(x),
                            |x,y| x.mul_complex(y),
                            |x| x.sum_complex())
                    } else {
                        self.convolve_vector_simd_impl(
                            vector,
                            |x| x,
                            |x| x,
                            |x,y| x * y,
                            |x| x.sum_real())
                    }
                }
                
                fn convolve_vector_simd_impl<T, C, CMut, RMul, RSum>(
                    mut self, 
                    vector: &Self,
                    convert: C,
                    convert_mut: CMut,
                    simd_mul: RMul,
                    simd_sum: RSum) -> VecResult<Self> 
                        where 
                            T: Zero + Clone + Copy + Add<Output=T> + Mul<Output=T>,
                            C: Fn(&[$data_type]) -> &[T],
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            RMul: Fn($reg, $reg) -> $reg,
                            RSum: Fn($reg) -> T {
                    let points = self.points();
                    let other_points = vector.points();
                    assert!(other_points < points);
                    let (other_start, other_end, full_conv_len, conv_len) =
                                (0, other_points, other_points, other_points - other_points / 2);
                    {
                        let len = self.len();
                        let points = self.points();
                        let other = convert(&vector.data[0..vector.len()]);
                        let temp = temp_mut!(self, len);
                        let complex = convert(&self.data[0..len]);
                        let dest = convert_mut(&mut temp[0..len]);
                        let other_iter = &other[other_start .. other_end];

                        let shifted_copies = Self::create_shifted_copies(vector);
                        let mut shifts = Vec::with_capacity(shifted_copies.len());
                        for shift in 0..shifted_copies.len() {
                            let simd = $reg::array_to_regs(&shifted_copies[shift]);
                            shifts.push(simd);
                        }

                        let scalar_len = conv_len + $reg::len(); // + $reg::len() due to rounding of odd numbers
                        let conv_len = conv_len as isize;
                        let mut i = 0;
                        for num in &mut dest[0..scalar_len] {
                            *num = Self::convolve_iteration(complex, other_iter, i, conv_len, full_conv_len);
                            i += 1;
                        }

                        let len_rounded = (len / $reg::len()) * $reg::len(); // The exact value is of no importance here
                        let simd = $reg::array_to_regs(&self.data[0..len_rounded]);
                        for num in &mut dest[scalar_len .. points - scalar_len] {
                            let end = (i + conv_len) as usize;
                            let shift = end % shifts.len();
                            let end = (end + shifts.len() - 1) / shifts.len();
                            let mut sum = $reg::splat(0.0);
                            let shifted = shifts[shift];
                            let simd_iter = simd[end - shifted.len() .. end].iter(); 
                            let iteration = 
                                simd_iter
                                .zip(shifted);
                            for (this, other) in iteration {
                                sum = sum + simd_mul(*this, *other);
                            }
                            (*num) = simd_sum(sum);
                            i += 1;
                        }

                        for num in &mut dest[points - scalar_len .. points] {
                            *num = Self::convolve_iteration(complex, other_iter, i, conv_len, full_conv_len);
                            i += 1;
                        }
                    }
                    Ok(self.swap_data_temp())
                }
            }
        )*
    }
}
add_conv_impl!(f32, Reg32; f64, Reg64);

macro_rules! add_conv_forw{
    ($($data_type:ident),*) => {
        $(
            impl<'a> Convolution<$data_type, &'a RealImpulseResponse<$data_type>> for RealTimeVector<$data_type> {
                fn convolve(self, function: &RealImpulseResponse<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }
            
            impl<'a> Convolution<$data_type, &'a RealImpulseResponse<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &RealImpulseResponse<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexImpulseResponse<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &ComplexImpulseResponse<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }

            impl<'a> FrequencyMultiplication<$data_type, &'a ComplexFrequencyResponse<$data_type>> for ComplexFreqVector<$data_type> {
                fn multiply_frequency_response(self, function: &ComplexFrequencyResponse<$data_type>, ratio: $data_type) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().multiply_frequency_response(function, ratio))
                }
            }
            
            impl<'a> FrequencyMultiplication<$data_type, &'a RealFrequencyResponse<$data_type>> for RealFreqVector<$data_type> {
                fn multiply_frequency_response(self, function: &RealFrequencyResponse<$data_type>, ratio: $data_type) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().multiply_frequency_response(function, ratio))
                }
            }
            
            impl<'a> FrequencyMultiplication<$data_type, &'a RealFrequencyResponse<$data_type>> for ComplexFreqVector<$data_type> {
                fn multiply_frequency_response(self, function: &RealFrequencyResponse<$data_type>, ratio: $data_type) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().multiply_frequency_response(function, ratio))
                }
            }
        )*
    }
}
add_conv_forw!(f32, f64);

macro_rules! add_conv_vector_forward{
    ($($name:ident, $($data_type:ident),*);*) => {
        $(
            $(
                impl VectorConvolution<$data_type> for $name<$data_type> {
                    fn convolve_vector(self, other: &Self) -> VecResult<Self> {
                        Self::from_genres(self.to_gen().convolve_vector(other.to_gen_borrow()))
                    }
                }
            )*
        )*
    }
}
add_conv_vector_forward!(
        RealTimeVector, f32, f64;
        ComplexTimeVector, f32, f64;
        RealFreqVector, f32, f64;
        ComplexFreqVector, f32, f64);
        
pub struct WrappingIterator<T>
    where T: Clone {
    start: *const T,
    end: *const T,
    pos: *const T,
    count: usize
}

impl<T> Iterator for WrappingIterator<T> 
    where T: Clone {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.count == 0 {
                return None;
            }
            
            let mut n = self.pos;
            if n < self.end {
                n = n.offset(1);
            } else {
                n = self.start;
            }
            
            self.pos = n;
            self.count -= 1;
            Some((*n).clone())
        }
    }
}

impl<T> WrappingIterator<T>
    where T: Clone {
    pub fn new(slice: &[T], pos: isize, iter_len: usize) -> Self {
        use std::isize;
        
        assert!(slice.len() <= isize::MAX as usize);
        let len = slice.len() as isize;
        let mut pos = pos % len;
        while pos < 0 {
            pos += len;
        }
        
        let start = slice.as_ptr();
        unsafe {
            WrappingIterator {
                start: start,
                end: start.offset(len - 1),
                pos: start.offset(pos),
                count: iter_len
            }
        }
    }
}

pub struct ReverseWrappingIterator<T>
    where T: Clone {
    start: *const T,
    end: *const T,
    pos: *const T,
    count: usize
}

impl<T> Iterator for ReverseWrappingIterator<T> 
    where T: Clone {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.count == 0 {
                return None;
            }
            
            let mut n = self.pos;
            if n > self.start {
                n = n.offset(-1);
            } else {
                n = self.end;
            }
            
            self.pos = n;
            self.count -= 1;
            Some((*n).clone())
        }
    }
}

impl<T> ReverseWrappingIterator<T>
    where T: Clone {
    pub fn new(slice: &[T], pos: isize, iter_len: usize) -> Self {
        use std::isize;
        
        assert!(slice.len() <= isize::MAX as usize);
        let len = slice.len() as isize;
        let mut pos = pos % len;
        while pos < 0 {
            pos += len;
        }
        
        let start = slice.as_ptr();
        unsafe {
            ReverseWrappingIterator {
                start: start,
                end: start.offset(len - 1),
                pos: start.offset(pos),
                count: iter_len
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WrappingIterator, ReverseWrappingIterator};
    use vector_types::*;
    use conv_types::*;
    use RealNumber;
    use std::fmt::Debug; 
    use num::complex::Complex32;
    
    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T) 
        where T: RealNumber + Debug {
        assert_eq!(left.len(), right.len());
        for i in 0..left.len() {
            if (left[i] - right[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?}", left, right);
            }
        }
    }
    
    #[test]
    fn convolve_complex_freq_and_freq32() {
        let vector = ComplexFreqVector32::from_constant(Complex32::new(1.0, 1.0), 5);
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
        let result = vector.multiply_frequency_response(&rc as &RealFrequencyResponse<f32>, 2.0).unwrap();
        let expected = 
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
    
    #[test]
    fn convolve_complex_freq_and_freq_even32() {
        let vector = ComplexFreqVector32::from_constant(Complex32::new(1.0, 1.0), 6);
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
        let result = vector.multiply_frequency_response(&rc as &RealFrequencyResponse<f32>, 2.0).unwrap();
        let expected = 
            [0.0, 0.0, 0.5, 0.5, 1.5, 1.5, 2.0, 2.0, 1.5, 1.5, 0.5, 0.5];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
    
    #[test]
    fn convolve_real_time_and_time32() {
        let vector = RealTimeVector32::from_array(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
        let result = vector.convolve(&rc as &RealImpulseResponse<f32>, 0.2, 5).unwrap();
        let expected = 
            [0.0, 0.2171850639713355, 0.4840621929215732, 0.7430526238101408, 0.9312114164253432, 
             1.0, 0.9312114164253432, 0.7430526238101408, 0.4840621929215732, 0.2171850639713355];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
    
    #[test]
    fn convolve_complex_time_and_time32() {
        let len = 11;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let result = time.convolve(&sinc as &RealImpulseResponse<f32>, 0.5, len / 2).unwrap();
        let result = result.magnitude().unwrap();
        let expected = 
            [0.12732396, 0.000000027827534, 0.21220659, 0.000000027827534, 0.63661975, 
             1.0, 0.63661975, 0.000000027827534, 0.21220659, 0.000000027827534, 0.12732396];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
    
    #[test]
    fn compare_conv_freq_mul() {
        let len = 11;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        time[len] = 1.0;
        let freq = time.clone().fft().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let ratio = 0.5;    
        let freq_res = freq.multiply_frequency_response(&sinc as &RealFrequencyResponse<f32>, 1.0 / ratio).unwrap();
        let time_res = time.convolve(&sinc as &RealImpulseResponse<f32>, 0.5, len).unwrap();
        let ifreq_res = freq_res.ifft().unwrap();
        let time_res = time_res.magnitude().unwrap();
        let ifreq_res = ifreq_res.magnitude().unwrap();
        assert_eq!(ifreq_res.is_complex(), time_res.is_complex());
        assert_eq!(ifreq_res.domain(), time_res.domain());
        assert_eq_tol(time_res.data(), ifreq_res.data(), 0.2);
    }
    
    #[test]
    fn invalid_length_parameter() {
        let len = 20;
        let time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        let sinc: SincFunction<f32> = SincFunction::new();
        let _result = time.convolve(&sinc as &RealImpulseResponse<f32>, 0.5, 10 * len).unwrap();
        // As long as we don't panic we are happy with the error handling here
    }
    
    #[test]
    fn convolve_complex_vectors32() {
        const LEN: usize = 11;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), LEN);
        time[LEN] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut argument_data = [0.0; LEN];
        {
            let mut v = -5.0;
            for a in &mut argument_data {
                *a = (&sinc as &RealImpulseResponse<f32>).calc(v * 0.5);
                v += 1.0;
            }
        }
        let argument = ComplexTimeVector32::from_real_imag(&argument_data, &[0.0; LEN]);
        assert_eq!(time.points(), argument.points());
        let result = time.convolve_vector(&argument).unwrap();
        assert_eq!(result.points(), LEN);
        let result = result.magnitude().unwrap();
        assert_eq!(result.points(), LEN);
        let expected = 
            [0.12732396, 0.000000027827534, 0.21220659, 0.000000027827534, 0.63661975, 
             1.0, 0.63661975, 0.000000027827534, 0.21220659, 0.000000027827534, 0.12732396];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
    
    #[test]
    fn wrapping_iterator() {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut iter = WrappingIterator::new(&array, -3, 8);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 3.0);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 1.0);
    }
    
    #[test]
    fn wrapping_rev_iterator() {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut iter = ReverseWrappingIterator::new(&array, 2, 5);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 3.0);
    }
        
    #[test]
    fn vector_conv_vs_freq_multiplication() {
        let a = ComplexTimeVector32::from_interleaved(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = ComplexTimeVector32::from_interleaved(&[15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0]);
        let conv = a.clone().convolve_vector(&b).unwrap();
        let a = a.fft().unwrap();
        let b = b.fft().unwrap();
        let mul = a.multiply_vector(&b).unwrap();
        let mul = mul.ifft().unwrap();
        let mul = mul.reverse().unwrap();
        let mul = mul.swap_halves().unwrap();
        assert_eq_tol(mul.data(), conv.data(), 1e-4);
    }
    
    #[test]
    fn shift_left_by_1_as_conv() {
        let a = RealTimeVector32::from_array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = RealTimeVector32::from_array(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let conv = a.convolve_vector(&b).unwrap();
        let conv = conv.magnitude().unwrap();
        let exp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq_tol(conv.data(), &exp, 1e-4);
    }
    
    #[test]
    fn shift_left_by_1_as_conv_shorter() {
        let a = RealTimeVector32::from_array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = RealTimeVector32::from_array(&[0.0, 0.0, 1.0]);
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let conv = a.convolve_vector(&b).unwrap();
        let conv = conv.magnitude().unwrap();
        let exp = [9.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq_tol(conv.data(), &exp, 1e-4);
    }
    
    #[test]
    fn vector_conv_vs_freq_multiplication_pure_real_data() {
        let a = RealTimeVector32::from_array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = RealTimeVector32::from_array(&[15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0]);
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let conv = a.clone().convolve_vector(&b).unwrap();
        let a = a.fft().unwrap();
        let b = b.fft().unwrap();
        let mul = a.multiply_vector(&b).unwrap();
        let mul = mul.ifft().unwrap();
        let mul = mul.magnitude().unwrap();
        let mul = mul.reverse().unwrap();
        let mul = mul.swap_halves().unwrap();
        let conv = conv.magnitude().unwrap();
        assert_eq_tol(mul.data(), conv.data(), 1e-4);
    }
    
    #[test]
    fn vector_conv_vs_freq_multiplication_pure_real_data_odd() {
        let a = RealTimeVector32::from_array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = RealTimeVector32::from_array(&[15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0]);
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let conv = a.clone().convolve_vector(&b).unwrap();
        let a = a.fft().unwrap();
        let b = b.fft().unwrap();
        let mul = a.multiply_vector(&b).unwrap();
        let mul = mul.ifft().unwrap();
        let mul = mul.magnitude().unwrap();
        let mul = mul.reverse().unwrap();
        let mul = mul.swap_halves().unwrap();
        let conv = conv.magnitude().unwrap();
        assert_eq_tol(mul.data(), conv.data(), 1e-4);
    }
}
