use super::definitions::{
    DataVector,
    VecResult};
use RealNumber;
use conv_types::{
    RealImpulseResponse,
    RealFrequencyResponse};
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
use num::traits::Zero;
use super::convolution_impl::WrappingIterator;

/// Provides a interpolation operation for data vectors.
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions are consistent.
/// However the actual implementation is lacking tests.
pub trait Interpolation<T> : DataVector<T> 
    where T : RealNumber {
    /// Interpolates `self` with the convolution function `function` by the real value `interpolation_factor`.
    /// Interpolation is done in in time domain and the argument `conv_len` can be used to balance accuracy 
    /// and performance. 
    /// A `delay` can be used to delay or phase shift the vector. The `delay` considers `self.delta()`.
    fn interpolatef(self, function: &RealImpulseResponse<T>, interpolation_factor: T, delay: T, conv_len: usize) -> VecResult<Self>;
    
    /// Interpolates `self` with the convolution function `function` by the interger value `interpolation_factor`.
    /// Interpolation is done in in frequency domain.
    fn interpolatei(self, function: &RealFrequencyResponse<T>, interpolation_factor: usize) -> VecResult<Self>;
}

macro_rules! define_interpolation_impl {
    ($($data_type:ident);*) => {
        $( 
            impl Interpolation<$data_type> for GenericDataVector<$data_type> {
                fn interpolatef(mut self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, conv_len: usize) -> VecResult<Self> {
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
                        let new_len = (len as $data_type * interpolation_factor).round() as usize;
                        let data = &self.data[0..len];
                        let temp = temp_mut!(self, new_len);
                        let temp = Self::array_to_complex_mut(temp);
                        let data = Self::array_to_complex(data);
                        let mut i = 0;
                        for num in temp {
                            let center = i as $data_type / interpolation_factor;
                            let rounded = (center).floor();
                            let iter = WrappingIterator::new(&data, rounded as isize - conv_len as isize -1, 2 * conv_len + 1);
                            let mut sum = Complex::<$data_type>::zero();
                            let mut j = -(conv_len as $data_type) - (center - rounded) + delay;
                            for c in iter {
                                sum = sum + c * function.calc(j);
                                j += 1.0;
                            }
                            (*num) = sum;
                            i += 1;
                        }
                        
                        self.valid_len = new_len;
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
                                        function.is_symmetric(),
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
        assert_eq_tol(result.data(), &expected, 0.1);
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
        assert_eq_tol(result.data(), &expected, 0.1);
    }
}