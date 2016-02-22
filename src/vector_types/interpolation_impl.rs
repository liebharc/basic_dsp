use super::definitions::{
    DataVector,
    VecResult};
use RealNumber;
use conv_types::{
    RealImpulseResponse,
    RealFrequencyResponse};
use super::{
    GenericDataVector,
    DataVectorDomain,
    RealVectorOperations,
    ComplexVectorOperations,
    GenericVectorOperations,
    RealTimeVector,
    RealFreqVector,
    TimeDomainOperations,
    FrequencyDomainOperations,
    ComplexTimeVector,
    ComplexFreqVector,
    ErrorReason};
use num::complex::Complex;
use num::traits::Zero;
use super::convolution_impl::WrappingIterator;
use simd_extensions::*;
use std::ops::{Add, Mul};
use std::fmt::{Display, Debug};

/// Provides a interpolation operation for data vectors.
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions are consistent.
/// However the actual implementation is lacking tests.
pub trait Interpolation<T> : DataVector<T> 
    where T : RealNumber {
    /// Interpolates `self` with the convolution function `function` by the real value `interpolation_factor`.
    /// Interpolation is done in in time domain and the argument `conv_len` can be used to balance accuracy 
    /// and computational performance. 
    /// A `delay` can be used to delay or phase shift the vector. The `delay` considers `self.delta()`.
    ///
    /// The complexity of this `interpolatef` is `O(self.points() * conv_len)`, while for `interpolatei` it's
    /// `O(self.points() * log(self.points()))`. If computational performance is important you should therefore decide
    /// how large `conv_len` needs to be to yield the descired accuracy. Then compare `conv_len` and then do a test
    /// run to compare the speed of `interpolatef` and `interpolatei`. Together with the information that 
    /// changing the vectors size change `log(self.points()` but not `conv_len` gives the indication that `interpolatef`
    /// performs faster for larger vectors while `interpolatei` performs faster for smaller vectors.
    fn interpolatef(self, function: &RealImpulseResponse<T>, interpolation_factor: T, delay: T, conv_len: usize) -> VecResult<Self>;
    
    /// Interpolates `self` with the convolution function `function` by the interger value `interpolation_factor`.
    /// Interpolation is done in in frequency domain.
    ///
    /// See the description of `interpolatef` for some basic performance considerations.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `ArgumentFunctionMustBeSymmetric`: if `!self.is_complex() && !function.is_symmetric()` or in words if `self` is a real
    ///    vector and `function` is asymmetric. Converting the vector into a complex vector before the interpolation is one way
    ///    to resolve this error. 
    fn interpolatei(self, function: &RealFrequencyResponse<T>, interpolation_factor: u32) -> VecResult<Self>;
    
    /// Decimates or downsamples `self`. `decimatei` is the inverse function to `interpolatei`.
    fn decimatei(self, decimation_factor: u32, delay: u32) -> VecResult<Self>;
}

macro_rules! define_interpolation_impl {
    ($($data_type:ident,$reg:ident);*) => {
        $( 
            impl GenericDataVector<$data_type> {
                fn interpolate_priv_scalar<T>(
                    temp: &mut [T], data: &[T], 
                    function: &RealImpulseResponse<$data_type>, 
                    interpolation_factor: $data_type, delay: $data_type,
                    conv_len: usize) 
                        where T: Zero + Mul<Output=T> + Copy + Display + Debug + Send + Sync + From<$data_type> {
                    let mut i = 0;
                    for num in temp {
                        let center = i as $data_type / interpolation_factor;
                        let rounded = (center).floor();
                        let iter = WrappingIterator::new(&data, rounded as isize - conv_len as isize -1, 2 * conv_len + 1);
                        let mut sum = T::zero();
                        let mut j = -(conv_len as $data_type) - (center - rounded) + delay;
                        for c in iter {
                            sum = sum + c * T::from(function.calc(j));
                            j += 1.0;
                        }
                        (*num) = sum;
                        i += 1;
                    }
                }
                
                fn function_to_vectors(
                    function: &RealImpulseResponse<$data_type>, 
                    conv_len: usize, 
                    interpolation_factor: usize) -> Vec<GenericDataVector<$data_type>> {
                    let mut result = Vec::with_capacity(interpolation_factor);
                    for shift in 0..interpolation_factor {
                        let offset = shift as $data_type / interpolation_factor as $data_type;
                        result.push(Self::function_to_vector(function, conv_len, offset));
                    }
                    
                    result
                }
                
                fn function_to_vector(
                    function: &RealImpulseResponse<$data_type>, 
                    conv_len: usize, 
                    offset: $data_type) -> GenericDataVector<$data_type> {
                    let mut imp_resp = GenericDataVector::<$data_type>::new(
                        false,
                        DataVectorDomain::Time,
                        0.0, 
                        2 * conv_len + 1,
                        1.0);
                    let mut i = 0;
                    let mut j = -(conv_len as $data_type);
                    while i < imp_resp.len() {
                        let value = function.calc(j - offset);
                        imp_resp[i] = value;
                        i += 1;
                        j += 1.0;
                    }
                    imp_resp
                }
                
                fn interpolate_priv_simd<T, C, CMut, RMul, RSum>(
                    mut self, 
                    function: &RealImpulseResponse<$data_type>, 
                    interpolation_factor: usize,
                    conv_len: usize, 
                    new_len: usize,
                    convert: C,
                    convert_mut: CMut,
                    simd_mul: RMul,
                    simd_sum: RSum) -> VecResult<Self> 
                        where 
                            T: Zero + Clone + From<$data_type> + Copy + Add<Output=T> + Mul<Output=T>,
                            C: Fn(&[$data_type]) -> &[T],
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            RMul: Fn($reg, $reg) -> $reg,
                            RSum: Fn($reg) -> T {
                    {              
                        let vectors = Self::function_to_vectors(function, conv_len, interpolation_factor);
                        /*let shifted_copies = Self::create_shifted_copies(&vector);
                        let mut shifts = Vec::with_capacity(shifted_copies.len());
                        for shift in 0..shifted_copies.len() {
                            let simd = $reg::array_to_regs(&shifted_copies[shift]);
                            shifts.push(simd);
                        }*/
                        
                        let len = self.len();
                        let data = convert(&self.data[0..len]);
                        let mut temp = temp_mut!(self, new_len);
                        let dest = convert_mut(&mut temp[0..new_len]);
                        
                        let len = dest.len();
                        let scalar_len = conv_len + 1; // + 1 due to rounding of odd numbers
                            
                        let mut i = 0;
                        for num in &mut dest[0..len] {
                            let rounded = i / interpolation_factor;
                            let iter = WrappingIterator::new(&data, rounded as isize - conv_len as isize -1, 2 * conv_len + 1);
                            let vector = &vectors[i % interpolation_factor];
                            let mut sum = T::zero();
                            let mut j = 0;
                            for c in iter {
                                sum = sum + c * T::from(vector[j]);
                                j += 1;
                            }
                            (*num) = sum;
                            i += 1;
                        }
                        
                        /*for num in &mut dest[len - scalar_len .. len] {
                            let center = i as $data_type / interpolation_factor as $data_type;
                            let rounded = (center).floor();
                            let iter = WrappingIterator::new(&data, rounded as isize - conv_len as isize -1, 2 * conv_len + 1);
                            let mut sum = T::zero();
                            let mut j = 0;
                            for c in iter {
                                sum = sum + c * T::from(vector[j]);
                                j += 1;
                            }
                            (*num) = sum;
                            i += 1;
                        }*/
                    }
                    self.valid_len = new_len;
                    Ok(self.swap_data_temp())
                }
            }
            
            impl Interpolation<$data_type> for GenericDataVector<$data_type> {
                fn interpolatef(mut self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, conv_len: usize) -> VecResult<Self> {
                    {
                        if interpolation_factor == 1.0 {
                            return Ok(self);
                        }
                        
                        let delay = delay / self.delta;
                        let len = self.len();
                        let points_half = self.points() / 2;
                        let conv_len =
                            if conv_len > points_half {
                                points_half
                            } else {
                                conv_len
                            };
                        let is_complex = self.is_complex();
                        let new_len = (len as $data_type * interpolation_factor).round() as usize;
                        let new_len = new_len + new_len % 2;
                        if conv_len <= 202 && new_len >= 2000 && 
                            (interpolation_factor.round() - interpolation_factor).abs() < 1e-6 && 
                            delay.abs() < 1e-6 {
                            let interpolation_factor = interpolation_factor.round() as usize;
                            if self.is_complex {
                                return self.interpolate_priv_simd(
                                    function,
                                    interpolation_factor,
                                    conv_len,
                                    new_len,
                                    |x| Self::array_to_complex(x),
                                    |x| Self::array_to_complex_mut(x),
                                    |x,y| x.mul_complex(y),
                                    |x| x.sum_complex())
                            } else {
                                return self.interpolate_priv_simd(
                                    function,
                                    interpolation_factor,
                                    conv_len,
                                    new_len,
                                    |x| x,
                                    |x| x,
                                    |x,y| x * y,
                                    |x| x.sum_real())
                            } 
                        }
                        else if is_complex {
                            let data = &self.data[0..len];
                            let temp = temp_mut!(self, new_len);
                            let temp = Self::array_to_complex_mut(temp);
                            let data = Self::array_to_complex(data);
                            Self::interpolate_priv_scalar(
                                temp, data,
                                function,
                                interpolation_factor, delay, conv_len);
                        }
                        else {
                            let data = &self.data[0..len];
                            let temp = temp_mut!(self, new_len);
                            Self::interpolate_priv_scalar(
                                temp, data,
                                function,
                                interpolation_factor, delay, conv_len);
                        }
                        
                        self.valid_len = new_len;
                    }
                    Ok(self.swap_data_temp())
                }
                
                fn interpolatei(self, function: &RealFrequencyResponse<$data_type>, interpolation_factor: u32) -> VecResult<Self> {
                    if interpolation_factor <= 1 {
                        return Ok(self);
                    }
                    reject_if!(self, !function.is_symmetric() &&!self.is_complex , ErrorReason::ArgumentFunctionMustBeSymmetric);
                    let is_complex = self.is_complex;
                    let freq = try! { 
                        Ok(self)
                        .and_then(|v|v.zero_interleave(interpolation_factor))
                        .and_then(|v|v.fft())
                    };
                    let points = freq.points();
                    let interpolation_factorf = interpolation_factor as $data_type;
                    Ok(freq)
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
                    .and_then(|v|v.real_scale(1.0 / points as $data_type))
                    .and_then(|v| {
                        if is_complex { Ok(v) } else { v.to_real() }
                    })
                }
                
                fn decimatei(mut self, decimation_factor: u32, delay: u32) -> VecResult<Self> {
                    {
                        let mut i = delay as usize;
                        let mut j = 0;
                        let len = self.points();
                        let is_complex = self.is_complex();
                        if is_complex {
                            let mut data = Self::array_to_complex_mut(&mut self.data);
                            let decimation_factor = decimation_factor as usize;
                            while i < len {
                                data[j] = data[i];
                                i += decimation_factor;
                                j += 1;
                            }
                            self.valid_len = j * 2;
                        }
                        else {
                            let mut data = &mut self.data;
                            let decimation_factor = decimation_factor as usize;
                            while i < len {
                                data[j] = data[i];
                                i += decimation_factor;
                                j += 1;
                            }
                            self.valid_len = j;
                        }
                    }
                    
                    Ok(self)
                }
            }
        )*
    }
}
define_interpolation_impl!(f32, Reg32; f64, Reg64);

macro_rules! define_interpolation_forward {
    ($($name:ident, $data_type:ident);*) => {
        $( 
            impl Interpolation<$data_type> for $name<$data_type> {
                fn interpolatef(self, function: &RealImpulseResponse<$data_type>, interpolation_factor: $data_type, delay: $data_type, len: usize) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().interpolatef(function, interpolation_factor, delay, len))
                }
                
                fn interpolatei(self, function: &RealFrequencyResponse<$data_type>, interpolation_factor: u32) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().interpolatei(function, interpolation_factor))
                }
                
                fn decimatei(self, decimation_factor: u32, delay: u32) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().decimatei(decimation_factor, delay))
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
            [0.16666667, 0.044658206, 0.16666667, 0.16666667, 0.16666667, 0.6220085, 
             1.1666667, 0.6220085, 0.16666667, 0.16666667, 0.16666667, 0.044658206];
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
            [0.0, 0.038979173, 0.0000000062572014, 0.15530863, 0.000000015884869, 0.6163295, 
             1.0, 0.61632943, 0.0000000142918966, 0.15530863, 0.000000048099658, 0.038979173];
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
    
    #[test]
    fn decimatei_test() {
        let time = ComplexTimeVector32::from_interleaved(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        let result = time.decimatei(2, 1).unwrap();
        let expected = [2.0, 3.0, 6.0, 7.0, 10.0, 11.0];
        assert_eq_tol(result.data(), &expected, 0.1);
    }
}