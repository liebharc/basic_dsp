use super::definitions::{
    DataVector,
    VecResult};
use RealNumber;
use conv_types::{RealImpulseResponse,RealFrequencyResponse};
use super::{
    PaddingOption,
    GenericDataVector,
    RealVectorOperations,
    GenericVectorOperations,
    RealTimeVector,
    RealFreqVector,
    TimeDomainOperations,
    FrequencyDomainOperations,
    ComplexTimeVector,
    ComplexFreqVector};
use num::complex::Complex;

/// Provides a interpolation operation for data vectors.
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions are consistent.
/// However the actual implementation is lacking tests.
pub trait Interpolation<T> : DataVector<T> 
    where T : RealNumber {
    /// Interpolates `self` with the convolution function `function` by the real value `interpolation_factor`.
    fn interpolatef(self, function: &RealImpulseResponse<T>, interpolation_factor: T, delay: T, len: usize) -> VecResult<Self>;
    
    /// Interpolates `self` with the convolution function `function` by the interger value `interpolation_factor`.
    fn interpolatei(self, function: &RealFrequencyResponse<T>, interpolation_factor: usize) -> VecResult<Self>;
}

macro_rules! define_interpolation_impl {
    ($($data_type:ident);*) => {
        $( 
            impl Interpolation<$data_type> for GenericDataVector<$data_type> {
                fn interpolatef(mut self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, conv_len: usize) -> VecResult<Self> {
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
                
                fn interpolatei(self, function: &RealFrequencyResponse<$data_type>, interpolation_factor: usize) -> VecResult<Self> {
                    if interpolation_factor <= 1 {
                        return Ok(self);
                    }
                    let freq = try! { self.fft() };
                    let points = freq.points();
                    let interpolation_factorf = interpolation_factor as $data_type;
                    freq.zero_pad(points * interpolation_factor, PaddingOption::Surround)
                    .and_then(|v| {
                        Ok(v.multiply_function_priv(
                                        interpolation_factorf,
                                        |array|Self::array_to_complex_mut(array),
                                        function,
                                        |f,x|Complex::<$data_type>::new(f.calc(x), 0.0)))
                    })
                    .and_then(|v|v.ifft_shift())
                    .and_then(|v|v.plain_ifft())
                    .and_then(|v|v.real_scale(1.0 / points as $data_type / interpolation_factorf as $data_type))
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
                fn interpolatef(self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().interpolatef(function, interpolation_factor, delay, len))
                }
                
                fn interpolatei(self, function: &RealFrequencyResponse<$data_type>, interpolation_factor: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().interpolatei(function, interpolation_factor))
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
                panic!("assertion failed: {:?} != {:?}", left, right);
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
            [0.0, 0.17254603, 0.0000000227619, 0.23570225, 0.000000004967054, 0.6439505, 
             1.0, 0.6439506, 0.000000017908969, 0.23570228, 0.000000004967054, 0.17254609];
        assert_eq_tol(result.data(), &expected, 1e-4);
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
            [0.07765429, 0.093237594, 0.08049384, 0.1812773, 0.08617285, 0.6247517, 
             0.9109876, 0.6247517, 0.08617285, 0.18127733, 0.0804938, 0.09323766];
        assert_eq_tol(result.data(), &expected, 1e-4);
    }
}