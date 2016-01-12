use super::definitions::{
    DataVector,
    VecResult,
    ErrorReason,
    DataVectorDomain};
use conv_types::*;
use RealNumber;
use num::traits::Zero;
use std::ops::Mul;
use std::fmt::Display;
use num::complex::Complex;
use super::{
    GenericDataVector,
    RealFreqVector,
    RealTimeVector,
    ComplexFreqVector,
    ComplexTimeVector};
use multicore_support::{Chunk, Complexity};

/// Provides a convolution operation for data vectors. 
pub trait Convolution<T, C> : DataVector<T> 
    where T : RealNumber {
    /// Convolves `self` with the convolution function `impulse_response`. For performance it's recommended 
    /// to use `FrequencyMultiplication` instead of this operation.
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
    ($($data_type:ident),*) => {
        $(
            impl<'a> Convolution<$data_type, &'a RealImpulseResponse<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &RealImpulseResponse<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    assert_time!(self);
                    if !self.is_complex {
                        Ok(self.convolve_function_priv(
                                ratio,
                                len,
                                |data|data,
                                |temp|temp,
                                |x|function.calc(x)
                            ))
                    } else {
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
                        for num in dest {
                            let iter = WrappingIterator::new(complex, i as isize - conv_len as isize -1);
                            let mut sum = T::zero();
                            let mut j = -(conv_len as $data_type);
                            for c in iter.take(2 * conv_len + 1) {
                                sum = sum + c * fun(j * ratio);
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
                                        ratio,
                                        |array|Self::array_to_complex_mut(array),
                                        function,
                                        |f,x|Complex::<$data_type>::new(f.calc(x), 0.0)))
                    }
                    else {
                        Ok(self.multiply_function_priv(
                                        ratio,
                                        |array|array,
                                        function,
                                        |f,x|f.calc(x)))
                    }
                }
            }
            
            impl GenericDataVector<$data_type> {
                fn multiply_function_priv<T,CMut,FA, F>(
                    mut self, 
                    ratio: $data_type,
                    convert_mut: CMut,
                    function_arg: FA, 
                    fun: F) -> Self
                        where 
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            FA: Copy + Sync,
                            F: Fn(FA, $data_type)->T + 'static + Sync,
                            T: Zero + Mul<Output=T> + Copy + Display + Send + Sync + From<$data_type>
                {
                    {
                        let len = self.len();
                        let points = self.points();
                        let complex = convert_mut(&mut self.data[0..len]);
                        Chunk::execute_with_range(
                            Complexity::Medium, &self.multicore_settings,
                            complex, points, 1, function_arg,
                            move |array, range, arg| {
                                let scale = T::from(ratio);
                                let offset = if points % 2 != 0 { 1 } else { 0 };
                                let max = (points - offset) as $data_type / 2.0; 
                                let mut j = -((points - offset + range.start) as $data_type) / 2.0;
                                for num in array {
                                    (*num) = (*num) * scale * fun(arg, j / max * ratio);
                                    j += 1.0;
                                }
                            });
                    }
                    self
                }
            }
            
            impl VectorConvolution<$data_type> for GenericDataVector<$data_type> {
                fn convolve_vector(mut self, vector: &Self) -> VecResult<Self> {
                    assert_meta_data!(self, vector);
                    assert_time!(self);
                    let points = self.points();
                    let other_points = vector.points();
                    let (other_start, other_end, conv_len) =
                            if other_points > points {
                                let center = other_points / 2;
                                let conv_len = points / 2;
                                (center - conv_len, center + conv_len, conv_len)
                            } else {
                                (0, other_points, other_points / 2)
                            };
                    if self.is_complex {
                        {
                            let len = self.len();
                            let other = Self::array_to_complex(&vector.data[0..vector.len()]);
                            let complex = Self::array_to_complex(&self.data[0..len]);
                            let dest = Self::array_to_complex_mut(&mut self.temp[0..len]);
                            print!("{}, {}\n", other_start, other_end);
                            let other_iter = &other[other_start .. other_end];
                            let mut i = 0;
                            for num in dest {
                                let data_iter = WrappingIterator::new(complex, i as isize - conv_len as isize -1); 
                                let mut sum = Complex::<$data_type>::zero();
                                for (this, other) in data_iter.zip(other_iter) {
                                    sum = sum + this * other;
                                }
                                (*num) = sum;
                                i += 1;
                            }
                        }
                        Ok(self.swap_data_temp())
                    } else {
                        {
                            let len = self.len();
                            let other = &vector.data[0..vector.len()];
                            let data = &self.data[0..len];
                            let dest = &mut self.temp[0..len];
                            let other_iter = &other[other_start .. other_end];
                            let mut i = 0;
                            for num in dest {
                                let data_iter = WrappingIterator::new(data, i as isize - conv_len as isize -1);
                                let mut sum = 0.0;
                                for (this, other) in data_iter.zip(other_iter) {
                                    sum = sum + this * other;
                                }
                                (*num) = sum;
                                i += 1;
                            }
                        }
                        Ok(self.swap_data_temp())
                    }
                }
            }
        )*
    }
}
add_conv_impl!(f32, f64);

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
        
struct WrappingIterator<T>
    where T: Clone {
    start: *const T,
    end: *const T,
    pos: *const T
}

impl<T> Iterator for WrappingIterator<T> 
    where T: Clone {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            let mut n = self.pos;
            if n < self.end {
                n = n.offset(1);
            } else {
                n = self.start;
            }
            
            self.pos = n;
            Some((*n).clone())
        }
    }
}

impl<T> WrappingIterator<T>
    where T: Clone {
    fn new(slice: &[T], pos: isize) -> Self {
        use std::isize;
        assert!(slice.len() <= isize::MAX as usize);
        let len = slice.len() as isize;
        let pos = pos % len;
        let pos = 
            if pos >= 0 {
                pos
            } else {
                (len + pos)
            };
        let start = slice.as_ptr();
        unsafe {
            WrappingIterator {
                start: start,
                end: start.offset(len - 1),
                pos: start.offset(pos)
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::WrappingIterator;
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
	fn convolve_real_freq_and_freq32() {
        let vector = ComplexFreqVector32::from_constant(Complex32::new(1.0, 1.0), 5);
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
        let result = vector.multiply_frequency_response(&rc as &RealFrequencyResponse<f32>, 2.0).unwrap();
        let expected = 
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0];
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
        let mut iter = WrappingIterator::new(&array, -3);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 3.0);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 1.0);
    }
}