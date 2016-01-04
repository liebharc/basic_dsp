use super::definitions::{
    DataVector,
    VecResult,
    ErrorReason,
    DataVectorDomain};
use conv_types::*;
use RealNumber;
use super::{
    GenericDataVector,
    RealTimeVector,
    RealFreqVector,
    ComplexFreqVector,
    ComplexTimeVector};
use super::time_freq_impl::{
    TimeDomainOperations,
    FrequencyDomainOperations
};

pub trait Convolution<T, C> : DataVector<T> 
    where T : RealNumber {
    /// Convolves `self` with the convolution function `function`.
    /// The domain in which `function` resides defines in which domain the operation
    /// is performed. For performance reasons it is advised to pass `function` in
    /// frequency domain since this will significantly speed up the calculation in most cases.
    fn convolve(self, function: C) -> VecResult<Self>;
}

macro_rules! add_conv_impl{
    ($($data_type:ident),*) => {
        $(
            impl<'a> Convolution<$data_type, &'a RealTimeConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &RealTimeConvFunction<$data_type>) -> VecResult<Self> {
                    panic!("TODO: Add convolution with real function for both real and complex vectors")
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexTimeConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &ComplexTimeConvFunction<$data_type>) -> VecResult<Self> {
                    assert_complex!(self);
                    panic!("TODO: Add convolution with complex function for complex vectors")
                }
            }

            impl<'a> Convolution<$data_type, &'a ComplexFrequencyConvFunction<$data_type>> for GenericDataVector<$data_type> {
                fn convolve(self, function: &ComplexFrequencyConvFunction<$data_type>) -> VecResult<Self> {
                    assert_complex!(self);
                    let freq = 
                        if self.domain == DataVectorDomain::Time {
                            self.fft()
                        } else {
                            Ok(self)
                        };
                   panic!("TODO: Add multiplication with complex function for complex vectors")
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
        DataVector};
    use conv_types::*;

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
        let vector = ComplexTimeVector32::from_constant(1.0, 10);
        let rc = RaiseCosineFuncton::new(0.35);
        let real = RealTimeLinearTableLookup::<f32>::from_conv_function(&rc, 0.4, 0.0, 10);
        let complex = real.to_complex();
        let result = vector.convolve(&complex as &ComplexTimeConvFunction<f32>).unwrap();
        assert_eq!(result.data(), &[0.0; 10])
    }
    
    #[test]
	fn convolve_complex_time_and_freq32() {
        let vector = ComplexTimeVector32::from_constant(1.0, 10);
        let rc = RaiseCosineFuncton::new(0.35);
        let real = RealTimeLinearTableLookup::<f32>::from_conv_function(&rc, 0.4, 0.0, 10);
        let freq = real.to_complex().fft();
        let result = vector.convolve(&freq as &ComplexFrequencyConvFunction<f32>).unwrap();
        assert_eq!(result.data(), &[0.0; 10])
    }
    
    #[test]
	fn convolve_complex_freq_and_time32() {
        let vector = ComplexFreqVector32::from_constant(1.0, 10);
        let rc = RaiseCosineFuncton::new(0.35);
        let real = RealTimeLinearTableLookup::<f32>::from_conv_function(&rc, 0.4, 0.0, 10);
        let complex = real.to_complex();
        let result = vector.convolve(&complex as &ComplexTimeConvFunction<f32>).unwrap();
        assert_eq!(result.data(), &[0.0; 10])
    }
    
    #[test]
	fn convolve_complex_freq_and_freq32() {
        let vector = ComplexFreqVector32::from_constant(1.0, 10);
        let rc = RaiseCosineFuncton::new(0.35);
        let real = RealTimeLinearTableLookup::<f32>::from_conv_function(&rc, 0.4, 0.0, 10);
        let freq = real.to_complex().fft();
        let result = vector.convolve(&freq as &ComplexFrequencyConvFunction<f32>).unwrap();
        assert_eq!(result.data(), &[0.0; 10])
    }
}