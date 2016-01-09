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
    fn convolve_vector(self, impulse_response: &Self, len: usize) -> VecResult<Self>;
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
                    if self.domain == DataVectorDomain::Time {
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
                        let complex = convert(&self.data[0..len]);
                        let dest = convert_mut(&mut self.temp[0..len]);
                        let mut i = 0;
                        for num in dest {
                            let start = if i > conv_len { i - conv_len } else { 0 };
                            let end = if i + conv_len < len { i + conv_len } else { len };
                            let mut sum = T::zero();
                            let center = i as $data_type;
                            let mut j = start as $data_type - center;
                            for c in &complex[start..end] {
                                sum = sum + (*c) * fun(j * ratio);
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
                            T: Zero + Mul<Output=T> + Copy + Display + Send + Sync
                {
                    {
                        let len = self.len();
                        let points = self.points();
                        let complex = convert_mut(&mut self.data[0..len]);
                        Chunk::execute_with_range(
                            Complexity::Medium, &self.multicore_settings,
                            complex, points, 1, function_arg,
                            move |array, range, arg| {
                                let max = points as $data_type / 2.0; 
                                let mut j = -((points + range.start) as $data_type) / 2.0;
                                for num in array {
                                    (*num) = (*num) * fun(arg, j / max * ratio);
                                    j += 1.0;
                                }
                            });
                    }
                    self
                }
            }
            
            impl VectorConvolution<$data_type> for GenericDataVector<$data_type> {
                fn convolve_vector(mut self, vector: &Self, conv_len: usize) -> VecResult<Self> {
                    assert_meta_data!(self, vector);
                    assert_time!(self);
                    reject_if!(self, self.points() != vector.points(), ErrorReason::VectorMetaDataMustAgree);
                    if self.is_complex {
                        {
                            let len = self.len();
                            let other = Self::array_to_complex(&vector.data[0..len]);
                            let complex = Self::array_to_complex(&self.data[0..len]);
                            let dest = Self::array_to_complex_mut(&mut self.temp[0..len]);
                            let mut i = 0;
                            for num in dest {
                                let start = if i > conv_len { i - conv_len } else { 0 };
                                let end = if i + conv_len < len { i + conv_len } else { len };
                                let mut sum = Complex::<$data_type>::zero();
                                let center = i;
                                let mut j = start - center;
                                for c in &complex[start..end] {
                                    sum = sum + (*c) * other[j];
                                    j += 1;
                                }
                                (*num) = sum;
                                i += 1;
                            }
                        }
                        Ok(self)
                    } else {
                        {
                            let len = self.len();
                            let other = &vector.data[0..len];
                            let data = &self.data[0..len];
                            let dest = &mut self.temp[0..len];
                            let mut i = 0;
                            for num in dest {
                                let start = if i > conv_len { i - conv_len } else { 0 };
                                let end = if i + conv_len < len { i + conv_len } else { len };
                                let mut sum = 0.0;
                                let center = i;
                                let mut j = start - center;
                                for c in &data[start..end] {
                                    sum = sum + (*c) * other[j];
                                    j += 1;
                                }
                                (*num) = sum;
                                i += 1;
                            }
                        }
                        Ok(self)
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
                    fn convolve_vector(self, other: &Self, len: usize) -> VecResult<Self> {
                        Self::from_genres(self.to_gen().convolve_vector(other.to_gen_borrow(), len))
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

#[cfg(test)]
mod tests {
	use super::*;
    use vector_types::{
        ComplexFreqVector32,
        RealTimeVector32,
        ComplexTimeVector32,
        DataVector};
    use vector_types::time_freq_impl::*;
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
        let rc: RaisedCosineFuncton<f32> = RaisedCosineFuncton::new(1.0);
        let result = vector.multiply_frequency_response(&rc as &RealFrequencyResponse<f32>, 1.0).unwrap();
        let expected = 
            [0.0, 0.0, 0.3454914, 0.3454914, 0.9045085, 0.9045085, 0.9045085, 0.9045085, 0.3454914, 0.3454914];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
    
    #[test]
	fn convolve_real_time_and_time32() {
        let vector = RealTimeVector32::from_array(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let rc: RaisedCosineFuncton<f32> = RaisedCosineFuncton::new(0.35);
        let result = vector.convolve(&rc as &RealImpulseResponse<f32>, 0.2, 10).unwrap();
        let expected = 
            [0.0, 0.2171850639713355, 0.4840621929215732, 0.7430526238101408, 0.9312114164253432, 
             1.0, 0.9312114164253432, 0.7430526238101408, 0.4840621929215732, 0.2171850639713355];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
    /*
    #[test]
    fn compare_conv_freq_mul() {
        let len = 10;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        time[4] = 1.0;
        let freq = time.clone().fft().unwrap();
        let rc: RaisedCosineFuncton<f32> = RaisedCosineFuncton::new(0.2);
        let ratio = 0.5;
        let freq_res = freq;// freq.multiply_frequency_response(&rc as &RealFrequencyResponse<f32>, ratio).unwrap();
        let time_res = time.convolve(&rc as &RealImpulseResponse<f32>, ratio, len).unwrap();
        let ifreq_res = freq_res.ifft().unwrap();
        assert_eq!(ifreq_res.is_complex(), time_res.is_complex());
        assert_eq!(ifreq_res.domain(), time_res.domain());
        assert_eq!(&ifreq_res.data(), &time_res.data());
    }*/
}