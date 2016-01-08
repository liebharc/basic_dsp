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
    RealTimeVector,
    ComplexFreqVector,
    ComplexTimeVector};
use super::time_freq_impl::{
    TimeDomainOperations,
    FrequencyDomainOperations
};
use multicore_support::{Chunk, Complexity};

/// Provides a convolution operation for data vectors. 
/// 
/// In contrast to the convolution definition this library allows for convinience to define a convolution in frequency domain which will then correctly be executed as multiplication.
pub trait Convolution<T, C> : DataVector<T> 
    where T : RealNumber {
    /// Convolves `self` with the convolution function `function`.
    /// The domain in which `function` resides defines in which domain the operation
    /// is performed. For performance reasons it is advised to pass `function` in
    /// frequency domain since this will significantly speed up the calculation in most cases.
    /// # Assumptions
    /// In frequency domain the operation assumes that the vector contains a full spectrum centered at 0 Hz. If half a spectrum
    /// or a fft shifted spectrum is provided the operation will come back with invalid results.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMustBeComplex`: if `self` is in real number space but `function` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain, `function` is in time domain and the vector can't be automatically converted to frequency domain since it is in real number space.
    fn convolve(self, function: C, ratio: T, len: usize) -> VecResult<Self>;
}

macro_rules! add_conv_impl{
    ($($data_type:ident),*) => {
        $(
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &RealTimeConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Ok(self.convolve_function_priv(
                            ratio,
                            len,
                            |data|data,
                            |temp|temp,
                            |x|function.calc(x)
                        ))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexTimeConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &ComplexTimeConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    assert_complex!(self);
                    let was_time = self.domain == DataVectorDomain::Time;
                    let time = 
                        if was_time {
                            self
                        } else {
                            try! { self.ifft() }
                        };
                    
                    
                    let result = time.convolve_function_priv(
                            ratio,
                            len,
                            |data|Self::array_to_complex(data),
                            |temp|Self::array_to_complex_mut(temp),
                            |x|function.calc(x)
                        );
                    
                    if was_time {
                        Ok(result)
                    } else {
                        result.fft()
                    }
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

            impl<'a> Convolution<$data_type, &'a ComplexFrequencyConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &ComplexFrequencyConvFunction<$data_type>, _ratio: $data_type, _len: usize) -> VecResult<Self> {
                    assert_complex!(self);
                    let was_time = self.domain == DataVectorDomain::Time;
                    let freq = 
                        if was_time {
                            try!{ self.fft() }
                        } else {
                            self
                        };
                        
                    let result = freq.multiply_function_priv(
                                    |array|Self::array_to_complex_mut(array),
                                    function,
                                    |f,x|f.calc(x));
                                    
                    if was_time {
                        result.ifft()
                    } else {
                        Ok(result)
                    }
                }
            }
            
            impl<'a> Convolution<$data_type, &'a RealFrequencyConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &RealFrequencyConvFunction<$data_type>, _ratio: $data_type, _len: usize) -> VecResult<Self> {
                    if self.is_complex {
                        let was_time = self.domain == DataVectorDomain::Time;
                        let freq = 
                            if was_time {
                                try!{ self.fft() }
                            } else {
                                self
                            };
                        let result = freq.multiply_function_priv(
                                        |array|Self::array_to_complex_mut(array),
                                        function,
                                        |f,x|Complex::<$data_type>::new(f.calc(x), 0.0));
                        
                        if was_time {
                            result.ifft()
                        } else {
                            Ok(result)
                        }
                    }
                    else {
                        assert_freq!(self);
                        let result = self.multiply_function_priv(
                                        |array|array,
                                        function,
                                        |f,x|f.calc(x));
                        Ok(result)
                    }
                }
            }
            
            impl GenericDataVector<$data_type> {
                fn multiply_function_priv<T,CMut,FA, F>(
                    mut self, 
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
                                    (*num) = (*num) * fun(arg, j / max);
                                    j += 1.0;
                                }
                            });
                    }
                    self
                }
            }
        )*
    }
}
add_conv_impl!(f32, f64);

macro_rules! add_conv_forw{
    ($($data_type:ident),*) => {
        $(
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for RealTimeVector<$data_type> {
                fn convolve(self, function: &RealTimeConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }
            
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &RealTimeConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }
            
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for ComplexFreqVector<$data_type> {
                fn convolve(self, function: &RealTimeConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexTimeConvFunction<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &ComplexTimeConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexFrequencyConvFunction<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &ComplexFrequencyConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexTimeConvFunction<$data_type>> for ComplexFreqVector<$data_type> {
                fn convolve(self, function: &ComplexTimeConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexFrequencyConvFunction<$data_type>> for ComplexFreqVector<$data_type> {
                fn convolve(self, function: &ComplexFrequencyConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }
            
            impl<'a> Convolution<$data_type, &'a RealFrequencyConvFunction<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &RealFrequencyConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }
            
            impl<'a> Convolution<$data_type, &'a RealFrequencyConvFunction<$data_type>> for ComplexFreqVector<$data_type> {
                fn convolve(self, function: &RealFrequencyConvFunction<$data_type>, ratio: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function, ratio, len))
                }
            }
        )*
    }
}
add_conv_forw!(f32, f64);

#[cfg(test)]
mod tests {
	use super::*;
    use super::super::{
        ComplexFreqVector32,
        RealTimeVector32,
        DataVector};
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
    /*
	#[test]
	fn convolve_real_and_time32() {
        let result = {
            let vector = RealTimeVector32::from_constant(1.0, 10);
            let conv = RaiseCosineFuncton::new(0.35);
            vector.convolve(&conv).unwrap()
        };
        
        assert_eq!(result.data(), &[0.0; 10])
    }
   
    #[test]
	fn convolve_complex_time_and_time32() {
        let vector = ComplexTimeVector32::from_constant(Complex32::new(1.0, 0.0), 10);
        let rc = RaiseCosineFuncton::new(0.35);
        let real = RealTimeLinearTableLookup::<f32>::from_conv_function(&rc, 0.4, 10);
        let complex = real.to_complex();
        let result = vector.convolve(&complex as &ComplexTimeConvFunction<f32>).unwrap();
        assert_eq!(result.data(), &[0.0; 10])
    }
    
    #[test]
	fn convolve_complex_time_and_freq32() {
        let vector = ComplexTimeVector32::from_constant(Complex32::new(1.0, 0.0), 10);
        let rc = RaiseCosineFuncton::new(0.35);
        let real = RealTimeLinearTableLookup::<f32>::from_conv_function(&rc, 0.4, 10);
        let freq = real.to_complex().fft();
        let result = vector.convolve(&freq as &ComplexFrequencyConvFunction<f32>).unwrap();
        assert_eq!(result.data(), &[0.0; 10])
    }
    
    #[test]
	fn convolve_complex_freq_and_time32() {
        let vector = ComplexFreqVector32::from_constant(Complex32::new(1.0, 0.0), 10);
        let rc = RaiseCosineFuncton::new(0.35);
        let real = RealTimeLinearTableLookup::<f32>::from_conv_function(&rc, 0.4, 10);
        let complex = real.to_complex();
        let result = vector.convolve(&complex as &ComplexTimeConvFunction<f32>).unwrap();
        assert_eq!(result.data(), &[0.0; 10])
    }*/
    
    #[test]
	fn convolve_real_freq_and_freq32() {
        let vector = ComplexFreqVector32::from_constant(Complex32::new(1.0, 1.0), 10);
        let rc: RaisedCosineFuncton<f32> = RaisedCosineFuncton::new(1.0);
        let result = vector.convolve(&rc as &RealFrequencyConvFunction<f32>, 1.0, 10).unwrap();
        let expected = 
            [0.0, 0.0, 0.3454914, 0.3454914, 0.9045085, 0.9045085, 0.9045085, 0.9045085, 0.3454914, 0.3454914];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
    
    #[test]
	fn convolve_real_time_and_time32() {
        let vector = RealTimeVector32::from_array(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let rc: RaisedCosineFuncton<f32> = RaisedCosineFuncton::new(0.35);
        let result = vector.convolve(&rc as &RealTimeConvFunction<f32>, 0.2, 10).unwrap();
        let expected = 
            [0.0, 0.2171850639713355, 0.4840621929215732, 0.7430526238101408, 0.9312114164253432, 
             1.0, 0.9312114164253432, 0.7430526238101408, 0.4840621929215732, 0.2171850639713355];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
}