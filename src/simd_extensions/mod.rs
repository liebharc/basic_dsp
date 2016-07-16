use num::complex::Complex;
use std::mem;

pub trait Simd<T> : Sized
    where T: Sized + Sync + Send
{
    type Array;
    fn to_array(self) -> Self::Array;
    type ComplexArray;
    fn len() -> usize;
    fn load(array: &[T], idx: usize) -> Self;
    fn load_wrap(array: &[T], idx: usize) -> Self;
    fn from_complex(value: Complex<T>) -> Self;
    fn add_real(self, value: T) -> Self;
    fn add_complex(self, value: Complex<T>) -> Self;
    fn scale_real(self, value: T) -> Self;
    fn scale_complex(self, value: Complex<T>) -> Self;
    fn complex_abs_squared(self) -> Self;
    fn complex_abs(self) -> Self;
    /// Same as `complex_abs_squared` but stores the result
    /// as complex number
    fn complex_abs_squared2(self) -> Self;
    /// Same as `complex_abs` but stores the result
    /// as complex number
    fn complex_abs2(self) -> Self;
    fn sqrt(self) -> Self;
    fn store(self, target: &mut [T], index: usize);
    fn store_half(self, target: &mut [T], index: usize);
    fn mul_complex(self, value: Self) -> Self;
    fn div_complex(self, value: Self) -> Self;
    fn sum_real(&self) -> T;
    fn sum_complex(&self) -> Complex<T>;
}

pub trait SimdGeneric<T> : Simd<T>
    where T: Sized + Sync + Send
{
    /// On some CPU architectures memory access needs to be aligned or otherwise
    /// the process will crash. This method takes a vector an divides it in three ranges:
    /// beginning, center, end. Beginning and end may not be loaded directly as SIMD registers.
    /// Center will contain most of the data.
    fn calc_data_alignment_reqs(array: &[T]) -> (usize, usize, usize);
    
    fn from_array(array: Self::Array) -> Self;
    
    fn to_complex_array(self) -> Self::ComplexArray;
    
    fn from_complex_array(array: Self::ComplexArray) -> Self;
    
    fn array_to_regs(array: &[T]) -> &[Self];
    
    fn array_to_regs_mut(array: &mut [T]) -> &mut [Self];
}

macro_rules! simd_generic_impl {
    ($data_type:ident, $reg:ident)
    =>
    {  
        impl SimdGeneric<$data_type> for $reg {
            fn calc_data_alignment_reqs(array: &[$data_type]) -> (usize, usize, usize) {
                let data_length = array.len();
                let addr = array.as_ptr();
                let scalar_left = (addr as usize % mem::size_of::<Self>()) / mem::size_of::<f32>(); 
                if scalar_left > data_length { 
                    (data_length, data_length, 0) 
                } else { 
                    let right = (data_length - scalar_left) % Self::len();
                    (scalar_left, data_length - right, data_length - right)
                }
            }
            
            fn from_array(array: Self::Array) -> Self {
                Self::load(&array, 0)
            }
            
            fn to_complex_array(self) -> Self::ComplexArray {
                unsafe { mem::transmute(self.to_array()) }
            }
            
            fn from_complex_array(array: Self::ComplexArray) -> Self {
                Self::from_array(unsafe { mem::transmute(array) })
            }
            
            fn array_to_regs(array: &[$data_type]) -> &[Self] {
                unsafe { 
                    let len = array.len();
                    let reg_len = Self::len();
                    if len % reg_len != 0 {
                        panic!("Argument must be dividable by {}", reg_len);
                    }
                    let trans: &[Self] = mem::transmute(array);
                    &trans[0 .. len / reg_len]
                }
            }
            
            fn array_to_regs_mut(array: &mut [$data_type]) -> &mut [Self] {
                unsafe { 
                    let len = array.len();
                    let reg_len = Self::len();
                    if len % reg_len != 0 {
                        panic!("Argument must be dividable by {}", reg_len);
                    }
                    let trans: &mut [Self] = mem::transmute(array);
                    &mut trans[0 .. len / reg_len]
                }
            }
        }
    }
}
#[cfg(any(feature = "doc", target_feature="avx2"))]
pub mod avx;

#[cfg(any(feature = "doc", target_feature="avx2"))]
pub use self::avx::{Reg32,Reg64};

#[cfg(any(feature = "doc", all(target_feature = "sse3", not(target_feature = "avx2"))))]
pub mod sse;

#[cfg(any(feature = "doc", all(target_feature = "sse3", not(target_feature = "avx2"))))]
pub use self::sse::{Reg32,Reg64};

#[cfg(any(feature = "doc", not(any(target_feature = "avx2", target_feature="sse3"))))]
pub mod fallback;

#[cfg(any(feature = "doc", not(any(target_feature = "avx2", target_feature="sse3"))))]
pub use self::fallback::{Reg32,Reg64};

simd_generic_impl!(f32, Reg32);
simd_generic_impl!(f64, Reg64);