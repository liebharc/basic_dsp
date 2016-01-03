//! Types around a convolution, see also https://en.wikipedia.org/wiki/Convolution.
//!
//! Convolutions in this library can be defined in time or frequency domain. In
//! frequency domain the convolution is automatically transformed into a multiplication
//! which is the analog operation to a convolution in time domain.
use RealNumber;
use num::traits::Zero;
use num::complex::{Complex, Complex32,Complex64};
use vector_types::{
    RealTimeVector,
    ComplexTimeVector,
    ComplexFreqVector
};
use vector_types::definitions::{
    DataVector,
    RealVectorOperations,
    ComplexVectorOperations
};

use vector_types::time_freq_impl::{
    TimeDomainOperations,
    FrequencyDomainOperations
};

/// A convolution function in time domain and real number space
pub trait RealTimeConvFunction<T>
    where T: RealNumber {
    /// Calculates the convolution for a real data point
    fn calc(&self, x: T) -> T;
}

/// A convolution function in time domain and complex number space
pub trait ComplexTimeConvFunction<T>
    where T: RealNumber {
    /// Calculates the convolution for a real data point
    fn calc(&self, x: T) -> Complex<T>;
}

/// A convolution function in frequency domain and complex number space
pub trait ComplexFrequencyConvFunction<T>
    where T: RealNumber {
    /// Calculates the convolution for a real data point
    fn calc(&self, x: T) -> Complex<T>;
}

/// Allows to create a lookup table with linear interpolation between table points.
/// This usually speeds up a convolution and sacrifies accuracy.
pub struct RealTimeLinearTableLookup<T>
    where T: RealNumber {
    table: Vec<T>,
    delta: T,
    offset: T
}

/// Allows to create a lookup table with linear interpolation between table points.
/// This usually speeds up a convolution and sacrifies accuracy.
pub struct ComplexTimeLinearTableLookup<T>
    where T: RealNumber {
    table: Vec<Complex<T>>,
    delta: T,
    offset: T
}

/// Allows to create a lookup table with linear interpolation between table points.
/// This usually speeds up a convolution and sacrifies accuracy.
pub struct ComplexFrequencyLinearTableLookup<T>
    where T: RealNumber {
    table: Vec<Complex<T>>,
    delta: T,
    offset: T
}

macro_rules! add_linear_table_lookup_impl {
    ($($name: ident: $conv_type: ident, $($data_type: ident, $result_type:ident),*);*) => {
        $(            
            $(
                impl $conv_type<$data_type> for $name<$data_type> {                
                    fn calc(&self, x: $data_type) -> $result_type {
                        let len = self.table.len();
                        let center = len / 2;
                        let center_float = center as $data_type;
                        let x = x / self.delta - self.offset + center_float;
                        let round_float = x.round();
                        let round = round_float as usize;
                        if round >= len {
                            return $result_type::zero();
                        }
                        
                        let round_tolerance = 1e-6;
                        if (x - round_float).abs() < round_tolerance {
                            return self.table[round];
                        }
                        
                        if x > round_float {
                            let other = round + 1;
                            if other >= len {
                                return self.table[round];
                            }
                            let y0 = self.table[round];
                            let x0 = round_float;
                            let y1 = self.table[other];
                            return y0 + (y1 - y0) * (x - x0);
                        } else {
                            if round == 0 {
                                return self.table[round];
                            }
                            let other = round - 1;
                            let y0 = self.table[round];
                            let x0 = round_float;
                            let y1 = self.table[other];
                            return y0 + (y1 - y0) * (x0 - x);
                        }
                    }
                }

                impl $name<$data_type> {
                    /// Creates a lookup table by putting the pieces together.             
                    pub fn from_raw_parts(table: &[$result_type], delta: $data_type, offset: $data_type) -> Self {
                        let owned_table = Vec::from(table);
                        $name { table: owned_table, delta: delta, offset: offset }
                    }
                    
                    
                    /// Creates a lookup table from another convolution function. The `delta` argument
                    /// can be used to balance performance vs. accuracy.
                    pub fn from_conv_function(other: &$conv_type<$data_type>, delta: $data_type, offset: $data_type, len: usize) -> Self {
                        let center = (len / 2) as isize;
                        let mut table = vec![$result_type::zero(); len];
                        let mut i = -center;
                        for n in &mut table {
                            *n = other.calc((i as $data_type) * delta);
                            i += 1;
                        }
                        $name { table: table, delta: delta, offset: offset }
                    }
                }
            )*
        )*
    }
}
add_linear_table_lookup_impl!(
    RealTimeLinearTableLookup: RealTimeConvFunction, f32, f32, f64, f64;
    ComplexTimeLinearTableLookup: ComplexTimeConvFunction, f32, Complex32, f64, Complex64;
    ComplexFrequencyLinearTableLookup: ComplexFrequencyConvFunction, f32, Complex32, f64, Complex64);
    
macro_rules! add_real_time_linear_table_impl {  
    ($($data_type: ident),*) => {  
        $(
            impl RealTimeLinearTableLookup<$data_type> {
                /// Convert the lookup table into complex number space
                pub fn to_complex(&self) -> ComplexTimeLinearTableLookup<$data_type> {
                    let vector = RealTimeVector::from_array(&self.table);
                    let complex = vector.to_complex().expect("to_complex shouldn't fail");
                    let complex = complex.complex_data();
                    let mut table = Vec::with_capacity(complex.len());
                    for n in complex {
                        table.push(*n);
                    }
                    ComplexTimeLinearTableLookup { table: table, delta: self.delta, offset: self.offset }
                }
            }
        )*
    }
}
add_real_time_linear_table_impl!(f32, f64);

macro_rules! add_complex_time_linear_table_impl {  
    ($($data_type: ident),*) => {  
        $(
            impl ComplexTimeLinearTableLookup<$data_type> {
                /// Convert the lookup table into real number space
                pub fn to_real(self) -> RealTimeLinearTableLookup<$data_type> {
                    let vector = ComplexTimeVector::from_complex(&self.table);
                    let real = vector.to_real().expect("to_complex shouldn't fail");
                    let real = real.data();
                    let mut table = Vec::with_capacity(real.len());
                    for n in real {
                        table.push(*n);
                    }
                    RealTimeLinearTableLookup { table: table, delta: self.delta, offset: self.offset }
                }
                
                /// Convert the lookup table into frequency domain
                pub fn fft(self) -> ComplexFrequencyLinearTableLookup<$data_type> {
                    let vector = ComplexTimeVector::from_complex(&self.table);
                    let freq = vector.fft().expect("vector fft shouldn't fail");
                    let freq = freq.complex_data();
                    let mut table = Vec::with_capacity(freq.len());
                    for n in freq {
                        table.push(*n);
                    }
                    ComplexFrequencyLinearTableLookup { table: table, delta: self.delta, offset: self.offset }
                }
            }
        )*
    }
}
add_complex_time_linear_table_impl!(f32, f64);

macro_rules! add_complex_frequency_linear_table_impl {  
    ($($data_type: ident),*) => {  
        $(
            impl ComplexFrequencyLinearTableLookup<$data_type> {
                    
                /// Convert the lookup table into time domain
                pub fn ifft(self) -> ComplexTimeLinearTableLookup<$data_type> {
                    let vector = ComplexFreqVector::from_complex(&self.table);
                    let time = vector.ifft().expect("vector ifft shouldn't fail");
                    let time = time.complex_data();
                    let mut table = Vec::with_capacity(time.len());
                    for n in time {
                        table.push(*n);
                    }
                    ComplexTimeLinearTableLookup { table: table, delta: self.delta, offset: self.offset }
                }
            }
        )*
    }
}
add_complex_frequency_linear_table_impl!(f32, f64);

/// Raised cosine function according to https://en.wikipedia.org/wiki/Raised-cosine_filter
pub struct  RaiseCosineFuncton<T>
    where T: RealNumber {
    rolloff: T        
}

impl<T> RealTimeConvFunction<T> for RaiseCosineFuncton<T> 
    where T: RealNumber {
    fn calc(&self, x: T) -> T {
        if x == T::zero() {
            return T::one();
        }
        
        let one = T::one();
        let two = T::from(2.0).unwrap();
        let pi = two * one.asin();
        let four = two * two;
        if x.abs() == one / (two * self.rolloff) {
            let arg = pi / two / self.rolloff;
            return (arg).sin() / arg * pi / four;
        }
        
        let pi_x = pi * x;
        let arg = two * self.rolloff * x;
        return pi_x.sin() * (pi_x * self.rolloff).cos() / pi_x / (one - (arg * arg))
    }
}

impl<T> RaiseCosineFuncton<T>
    where T: RealNumber {
    /// Creates a raised cosine function.
    pub fn new(rolloff: T) -> Self {
        RaiseCosineFuncton { rolloff: rolloff }
    }
}

#[cfg(test)]
mod tests {
	use super::*;
    use super::super::RealNumber;
    use std::fmt::Debug;
    
    fn conv_test<T, C>(conv: C, expected: &[T], step: T, tolerance: T) 
        where T: RealNumber + Debug,
              C: RealTimeConvFunction<T> {
        let mut result = vec![T::zero(); expected.len()];
        let mut j = -(expected.len() as isize / 2);
        for i in 0..result.len() {
            result[i] = conv.calc(T::from(j).unwrap() * step);
            j += 1;
        }
        
        for i in 0..result.len() {
            if (result[i] - expected[i]).abs() > tolerance {
                panic!("assertion failed: {:?} != {:?}", result, expected);
            }
        }
    }

	#[test]
	fn raised_cosine32_test()
	{
        let window = RaiseCosineFuncton::new(0.35);
        let expected = 
            [0.0, 0.2171850639713355, 0.4840621929215732, 0.7430526238101408, 0.9312114164253432, 
             1.0, 0.9312114164253432, 0.7430526238101408, 0.4840621929215732, 0.2171850639713355];
        conv_test(window, &expected, 0.2, 1e-4);
	}
    
    #[test]
    fn lookup_table32_test() {
        let window = RaiseCosineFuncton::new(0.35);
        let table = RealTimeLinearTableLookup::<f64>::from_conv_function(&window, 0.2, 0.0, 10);
        let expected = 
            [0.0, 0.2171850639713355, 0.4840621929215732, 0.7430526238101408, 0.9312114164253432, 
             1.0, 0.9312114164253432, 0.7430526238101408, 0.4840621929215732, 0.2171850639713355];
        conv_test(table, &expected, 0.2, 1e-4);
    }
    
    #[test]
    fn linear_interpolation_lookup_table32_test() {
        let window = RaiseCosineFuncton::new(0.35);
        let table = RealTimeLinearTableLookup::<f64>::from_conv_function(&window, 0.4, 0.0, 10);
        let expected = 
            [0.0, 0.2171850639713355, 0.4840621929215732, 0.7430526238101408, 0.9312114164253432, 
             1.0, 0.9312114164253432, 0.7430526238101408, 0.4840621929215732, 0.2171850639713355];
        conv_test(table, &expected, 0.2, 0.1);
    }
}