use super::definitions::{
    DataVector,
    VecResult,
    ErrorReason,
    DataVectorDomain};
use conv_types::*;
use RealNumber;
use num::traits::Zero;
use std::ops::Mul;
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
    fn convolve(self, function: C) -> VecResult<Self>;
}

macro_rules! add_conv_impl{
    ($($data_type:ident),*) => {
        $(
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(mut self, function: &RealTimeConvFunction<$data_type>) -> VecResult<Self> {
                    if self.domain() == DataVectorDomain::Time {
                        assert_freq!(self);
                    }
                    {
                        self.convolve_priv(
                                |data|data,
                                |temp|temp,
                                |x|function.calc(x)
                            );
                    }
                    Ok(self)
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexTimeConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &ComplexTimeConvFunction<$data_type>) -> VecResult<Self> {
                    assert_complex!(self);
                    let was_time = self.domain == DataVectorDomain::Time;
                    let mut time = 
                        if was_time {
                            self
                        } else {
                            try! { self.ifft() }
                        };
                    {
                        time.convolve_priv(
                            |data|Self::array_to_complex(data),
                            |temp|Self::array_to_complex_mut(temp),
                            |x|function.calc(x)
                        );
                    }
                    
                    if was_time {
                        Ok(time)
                    } else {
                        time.fft()
                    }
                }
            }
            
            impl GenericDataVector<$data_type> {
                fn convolve_priv<T,C,CMut,F>(
                    &mut self, 
                    convert: C,
                    convert_mut: CMut,
                    fun: F)
                        where 
                            C: Fn(&[$data_type]) -> &[T],
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            F: Fn($data_type)->T,
                            T: Zero + Mul<Output=T> + Copy
                {
                    let len = self.len();
                    let delta = self.delta();
                    let complex = convert(&self.data[0..len]);
                    let dest = convert_mut(&mut self.temp[0..len]);
                    let conv_len = 11; // TODO make configurable
                    let mut i = 0;
                    for num in dest {
                        let start = if i > conv_len { i - conv_len } else { 0 };
                        let end = if i + conv_len < len { i + conv_len } else { len };
                        let mut sum = T::zero();
                        let mut j = 0.0;
                        for c in &complex[start..end] {
                            sum = (*c) * fun(j * delta);
                            j += 1.0;
                        }
                        (*num) = sum;
                        i += 1;
                    }
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexFrequencyConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &ComplexFrequencyConvFunction<$data_type>) -> VecResult<Self> {
                    assert_complex!(self);
                    let was_time = self.domain == DataVectorDomain::Time;
                    let mut freq = 
                        if was_time {
                            try!{ self.fft() }
                        } else {
                            self
                        };
                    {
                        let len = freq.len();
                        let points = freq.points();
                        let delta = freq.delta();
                        let complex = Self::array_to_complex_mut(&mut freq.data[0..len]);
                        Chunk::execute_with_range(
                            Complexity::Medium, &freq.multicore_settings,
                            complex, points, 1, function,
                            move |array, range, function| {
                                let mut j = -((points + range.start) as $data_type) / 2.0;
                                for num in array {
                                    (*num) = (*num) * function.calc(j * delta);
                                    j += 1.0;
                                }
                            });
                    }
                    
                    if was_time {
                        freq.ifft()
                    } else {
                        Ok(freq)
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
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for RealTimeVector<$data_type> {
                fn convolve(self, function: &RealTimeConvFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function))
                }
            }
            
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &RealTimeConvFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function))
                }
            }
            
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for ComplexFreqVector<$data_type> {
                fn convolve(self, function: &RealTimeConvFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexTimeConvFunction<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &ComplexTimeConvFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexFrequencyConvFunction<$data_type>> for ComplexTimeVector<$data_type> {
                fn convolve(self, function: &ComplexFrequencyConvFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexTimeConvFunction<$data_type>> for ComplexFreqVector<$data_type> {
                fn convolve(self, function: &ComplexTimeConvFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function))
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexFrequencyConvFunction<$data_type>> for ComplexFreqVector<$data_type> {
                fn convolve(self, function: &ComplexFrequencyConvFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().convolve(function))
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
        ComplexTimeVector32,
        ComplexFreqVector32,
        RealTimeVector32,
        FrequencyDomainOperations,
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
    }
    
    #[test]
	fn convolve_complex_freq_and_freq32() {
        let vector = ComplexFreqVector32::from_constant(Complex32::new(1.0, 0.0), 10);
        let rc = RaiseCosineFuncton::new(0.35);
        let real = RealTimeLinearTableLookup::<f32>::from_conv_function(&rc, 0.4, 10);
        let freq = real.to_complex().fft();
        let result = vector.convolve(&freq as &ComplexFrequencyConvFunction<f32>).unwrap();
        let result = result.ifft().unwrap();
        let expected = 
            [-0.63574266, -0.63574266, 0.14328241, -0.2502644, -0.7839512, 
            -0.14717892, -0.14717895, -0.78395116, -0.2502644, 0.14328243];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
}