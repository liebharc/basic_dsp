use traits::*;
use super::Zero;
use std::mem;

pub trait Simd<T>: Sized
    where T: Sized + Sync + Send
{
    type Array;
    fn to_array(self) -> Self::Array;
    type ComplexArray;
    fn len() -> usize;
    fn load_wrap_unchecked(array: &[T], idx: usize) -> Self;
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
    fn store_half_unchecked(self, target: &mut [T], index: usize);
    fn mul_complex(self, value: Self) -> Self;
    fn div_complex(self, value: Self) -> Self;
    fn sum_real(&self) -> T;
    fn sum_complex(&self) -> Complex<T>;
}

pub trait SimdGeneric<T>: Simd<T>
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

    fn iter_over_vector<F>(self, op: F) -> Self where F: FnMut(T) -> T;

    fn iter_over_complex_vector<F>(self, op: F) -> Self where F: FnMut(Complex<T>) -> Complex<T>;

    fn array_to_regs(array: &[T]) -> &[Self];

    fn array_to_regs_mut(array: &mut [T]) -> &mut [Self];

    fn load_unchecked(array: &[T], idx: usize) -> Self;

    fn store_unchecked(self, target: &mut [T], index: usize);

    fn extract(self, idx: u32) -> T;

    fn splat(value: T) -> Self;

    fn store(self, array: &mut [T], index: usize);
}

pub trait SimdApproximations<T> : Simd<T>
    where T: Sized + Sync + Send {
    fn ln_approx(self) -> Self;

    fn exp_approx(self) -> Self;

    fn sin_approx(self) -> Self;

    fn cos_approx(self) -> Self;

    fn sin_cos_approx(self, is_sin: bool) -> Self;
}

#[repr(packed)]
#[derive(Debug, Copy, Clone)]
struct Unalign<T>(T);

macro_rules! simd_generic_impl {
    ($data_type:ident, $reg:ident)
    =>
    {
        impl Zero for $reg {
            fn zero() -> $reg {
                $reg::splat(0.0)
            }
        }

        impl SimdGeneric<$data_type> for $reg {
            #[inline]
            fn calc_data_alignment_reqs(array: &[$data_type]) -> (usize, usize, usize) {
                let data_length = array.len();
                let addr = array.as_ptr();
                let scalar_left = (addr as usize % mem::size_of::<Self>()) / mem::size_of::<f32>();
                if scalar_left + $reg::len() > data_length {
                    // Result order: scalar_left, scalar_right, vectorization_length
                    (data_length, data_length, 0)
                } else {
                    let right = (data_length - scalar_left) % Self::len();
                    (scalar_left, data_length - right, data_length - right)
                }
            }

            #[inline]
            fn from_array(array: Self::Array) -> Self {
                Self::load(&array, 0)
            }

            #[inline]
            fn to_complex_array(self) -> Self::ComplexArray {
                unsafe { mem::transmute(self.to_array()) }
            }

            #[inline]
            fn from_complex_array(array: Self::ComplexArray) -> Self {
                Self::from_array(unsafe { mem::transmute(array) })
            }

            #[inline]
            fn iter_over_vector<F>(self, mut op: F) -> Self
                where F: FnMut($data_type) -> $data_type {
                let mut array = self.to_array();
                for n in &mut array {
                    *n = op(*n);
                }
                Self::from_array(array)
            }

            #[inline]
            fn iter_over_complex_vector<F>(self,  mut op: F) -> Self
                where F: FnMut(Complex<$data_type>) -> Complex<$data_type> {
                let mut array = self.to_complex_array();
                for n in &mut array {
                    *n = op(*n);
                }
                Self::from_complex_array(array)
            }

            #[inline]
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

            #[inline]
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

            #[inline]
            fn load_unchecked(array: &[$data_type], idx: usize) -> Self {
                let loaded = unsafe {
                    let data = array.as_ptr();
                    *(data.offset(idx as isize) as *const Unalign<Self>)
                };
                loaded.0
            }

            #[inline]
            fn store_unchecked(self, array: &mut [$data_type], idx: usize) {
                unsafe {
                    let place = array.as_mut_ptr();
                    *(place.offset(idx as isize) as *mut Unalign<Self>) = Unalign(self)
                }
            }

            #[inline]
            fn extract(self, idx: u32) -> $data_type {
                $reg::extract(self, idx)
            }

            #[inline]
            fn splat(value: $data_type) -> Self {
                $reg::splat(value)
            }

            #[inline]
            fn store(self, array: &mut [$data_type], index: usize) {
                $reg::store(self, array, index);
            }
        }
    }
}
#[cfg(any(feature = "doc", feature="use_avx"))]
mod avx;

#[cfg(any(feature = "doc", feature="use_avx"))]
pub use self::avx::{Reg32, Reg64, IntReg32, IntReg64, UIntReg32, UIntReg64};

#[cfg(any(feature = "doc", all(feature = "use_sse", not(feature = "use_avx"))))]
mod sse;

#[cfg(any(feature = "doc", all(feature = "use_sse", not(feature = "use_avx"))))]
pub use self::sse::{Reg32, Reg64, IntReg32, IntReg64, UIntReg32, UIntReg64};

//#[cfg(any(feature = "doc", any(feature = "use_sse", feature = "use_avx")))]
#[cfg(any(feature = "doc", all(feature = "use_sse", not(feature = "use_avx"))))]
mod approximations;

#[cfg(any(feature = "doc", not(any(feature = "use_avx", feature="use_sse"))))]
mod fallback;

#[cfg(any(feature = "doc", not(any(feature = "use_avx", feature="use_sse"))))]
pub use self::fallback::{Reg32, Reg64};

#[cfg(any(feature = "doc", not(feature="use_sse")))]
mod approx_fallback;

simd_generic_impl!(f32, Reg32);
simd_generic_impl!(f64, Reg64);
