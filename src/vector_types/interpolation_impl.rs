use super::definitions::{
    DataVector,
    VecResult};
use RealNumber;
use conv_types::RealTimeConvFunction;
use super::{
    GenericDataVector,
    RealTimeVector,
    RealFreqVector,
    ComplexTimeVector,
    ComplexFreqVector};

/// Provides a interpolation operation for data vectors.
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions are consistent.
/// However the actual implementation is lacking tests.
pub trait Interpolation<T> : DataVector<T> 
    where T : RealNumber {
    /// Interpolates `self` with the convolution function `function`.
    fn interpolate(self, function: &RealTimeConvFunction<T>, interpolation_factor: T, delay: T) -> VecResult<Self>;
}

macro_rules! define_interpolation_impl {
    ($($data_type:ident);*) => {
        $( 
            impl Interpolation<$data_type> for GenericDataVector<$data_type> {
                fn interpolate(self, _function: &RealTimeConvFunction<$data_type>, _interpolation_factor: $data_type, _delay: $data_type) -> VecResult<Self> {
                    panic!("Panic")
                }
            }
        )*
    }
}
define_interpolation_impl!(f32; f64);

macro_rules! define_interpolation_forward {
    ($($name:ident, $data_type:ident);*) => {
        $( 
            impl Interpolation<$data_type> for $name<$data_type> {
                fn interpolate(self, function: &RealTimeConvFunction<$data_type>, interpolation_factor: $data_type, delay: $data_type) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().interpolate(function, interpolation_factor, delay))
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