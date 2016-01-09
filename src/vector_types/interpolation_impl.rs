use super::definitions::{
    DataVector,
    VecResult};
use RealNumber;
use conv_types::RealImpulseResponse;
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
    fn interpolate(self, function: &RealImpulseResponse<T>, interpolation_factor: T, delay: T, len: usize) -> VecResult<Self>;
}

macro_rules! define_interpolation_impl {
    ($($data_type:ident);*) => {
        $( 
            impl Interpolation<$data_type> for GenericDataVector<$data_type> {
                fn interpolate(mut self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, conv_len: usize) -> VecResult<Self> {
                    {
                        let len = self.len();
                        let new_len = (len as $data_type * interpolation_factor) as usize;
                        let data = &self.data[0..len];
                        let temp = temp_mut!(self, new_len);
                        let mut i = 0;
                        for num in temp {
                            let k = (i as $data_type / interpolation_factor).round() as usize;
                            let start = if k > conv_len { i - conv_len } else { 0 };
                            let end = if k + conv_len < len { i + conv_len } else { len };
                            let mut sum = 0.0;
                            let center = k as $data_type;
                            let mut j = start as $data_type - center - delay;
                            for c in &data[start..end] {
                                sum = sum + (*c) * function.calc(j * interpolation_factor);
                                j += 1.0;
                            }
                            (*num) = sum;
                            i += 1;
                        }
                    }
                    Ok(self.swap_data_temp())
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
                fn interpolate(self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().interpolate(function, interpolation_factor, delay, len))
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