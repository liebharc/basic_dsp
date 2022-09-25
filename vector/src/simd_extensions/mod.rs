use crate::numbers::*;
#[cfg(all(feature = "use_simd"))]
use packed_simd_2::*;
use std;
use std::mem;
use std::ops::*;
mod simd_partition;
pub use self::simd_partition::{EdgeIteratorMut, IndexedEdgeIteratorMut, SimdPartition};

/// SIMD methods which have `f32` or `f64` specific implementation.
pub trait Simd<T>: Sized
where
    T: Sized + Sync + Send,
{
    /// The type of real valued array which matches a SIMD register.
    type Array;

    /// SIMD register to array.
    fn to_array(self) -> Self::Array;

    /// The type of complex valued array which matches a SIMD register.
    type ComplexArray;

    /// Number of elements in a SIMD register.
    const LEN: usize;

    /// Creates a SIMD register loaded with a complex value.
    fn from_complex(value: Complex<T>) -> Self;

    /// Add a real number to the register.
    fn add_real(self, value: T) -> Self;

    /// Add a complex number to the register.
    fn add_complex(self, value: Complex<T>) -> Self;

    /// Scale the register by a real number.
    fn scale_real(self, value: T) -> Self;

    /// Scale the register by a complex number.
    fn scale_complex(self, value: Complex<T>) -> Self;

    /// Store the complex norm squared in the first half of the vector.
    fn complex_abs_squared(self) -> Self;

    /// Store the complex norm in the first half of the vector.
    fn complex_abs(self) -> Self;

    /// Calculates the square root of the register.
    fn sqrt(self) -> Self;

    /// Stores the first half of the vector in an array.
    /// Useful e.g. in combination with `complex_abs_squared`.
    fn store_half(self, target: &mut [T], index: usize);

    /// Multiplies the register with a complex value.
    fn mul_complex(self, value: Self) -> Self;

    /// Divides the register by a complex value.
    fn div_complex(self, value: Self) -> Self;

    /// Calculates the sum of all register elements, assuming that they
    /// are real valued.
    fn sum_real(&self) -> T;

    /// Calculates the sum of all register elements, assuming that they
    /// are complex valued.
    fn sum_complex(&self) -> Complex<T>;

    fn max(self, other: Self) -> Self;

    fn min(self, other: Self) -> Self;

    // Swaps I and Q (or Real and Imag) of a complex vector
    fn swap_iq(self) -> Self;
}

/// Dirty workaround since the stdsimd doesn't implement conversion traits (yet?).
pub trait SimdFrom<T> {
    fn regfrom(src: T) -> Self;
}

/// SIMD methods which share their implementation independent if it's a `f32` or `f64` register.
pub trait SimdGeneric<T>:
    Simd<T>
    + SimdApproximations<T>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Copy
    + Clone
    + Sync
    + Send
    + Sized
    + Zero
where
    T: Sized + Sync + Send,
{
    /// On some CPU architectures memory access needs to be aligned or otherwise
    /// the process will crash. This method takes a vector an divides it in three ranges:
    /// beginning, center, end. Beginning and end may not be loaded directly as SIMD registers.
    /// Center will contain most of the data.
    fn calc_data_alignment_reqs(array: &[T]) -> SimdPartition<T>;

    /// Converts a real valued array which has exactly the size of a SIMD register
    /// into a SIMD register.
    fn from_array(array: Self::Array) -> Self;

    /// Converts the SIMD register into a complex valued array.
    fn to_complex_array(self) -> Self::ComplexArray;

    /// Converts a complex valued array which has exactly the size of a SIMD register
    /// into a SIMD register.
    fn from_complex_array(array: Self::ComplexArray) -> Self;

    /// Executed the given function on each element of the register.
    /// Register elements are assumed to be real valued.
    fn iter_over_vector<F>(self, op: F) -> Self
    where
        F: FnMut(T) -> T;

    /// Executed the given function on each element of the register.
    /// Register elements are assumed to be complex valued.
    fn iter_over_complex_vector<F>(self, op: F) -> Self
    where
        F: FnMut(Complex<T>) -> Complex<T>;

    /// Converts an array slice into a slice of SIMD registers.
    ///
    /// WARNING: `calc_data_alignment_reqs` must have been used before to ensure that
    /// data is loaded with the proper memory alignment. Code will panic otherwise.
    fn array_to_regs(array: &[T]) -> &[Self];

    /// Converts a mutable array slice into a slice of mutable SIMD registers.
    ///
    /// WARNING: `calc_data_alignment_reqs` must have been used before to ensure that
    /// data is loaded with the proper memory alignment. Code will panic otherwise.
    fn array_to_regs_mut(array: &mut [T]) -> &mut [Self];

    /// Loads a SIMD register from an array without any bound checks.
    fn load(array: &[T], idx: usize) -> Self;

    /// Stores a SIMD register into an array.
    fn store(self, array: &mut [T], index: usize);

    /// Returns one element from the register.
    fn extract(self, idx: usize) -> T;

    /// Creates a new SIMD register where every element equals `value`.
    fn splat(value: T) -> Self;
}

/// Approximated and faster implementation of some numeric standard function.
/// The approximations are implemented based on SIMD registers.
/// Refer to the documentation of the `ApproximatedOps` trait (which is part of
/// the public API of this lib) for some information about accuracy and speed.
pub trait SimdApproximations<T> {
    /// Returns the natural logarithm of the number.
    fn ln_approx(self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    fn exp_approx(self) -> Self;

    /// Computes the sine of a number (in radians).
    fn sin_approx(self) -> Self;

    /// Computes the cosine of a number (in radians).
    fn cos_approx(self) -> Self;

    /// An implementation detail which leaked into the trait defintion
    /// for convenience. Use `sin_approx` or `cos_approx` instead of this
    /// function.
    ///
    /// Since the implementation of sine and cosine is almost identical
    /// the implementation is easier with a boolean `is_sin` flag which
    /// determines if the sine or cosine is requried.
    fn sin_cos_approx(self, is_sin: bool) -> Self;
}

fn get_alignment_offset(addr: usize, reg_len: usize) -> usize {
    addr % reg_len
}

macro_rules! simd_generic_impl {
    ($data_type:ident, $mod: ident::$reg:ident) => {
        impl Zero for $mod::$reg {
            fn zero() -> Self {
                Self::splat(0.0)
            }
        }

        impl SimdGeneric<$data_type> for $mod::$reg {
            #[inline]
            fn calc_data_alignment_reqs(array: &[$data_type]) -> SimdPartition<$data_type> {
                let data_length = array.len();
                let addr = array.as_ptr();
                let left = get_alignment_offset(addr as usize, mem::size_of::<Self>());
                assert!(left % mem::size_of::<$data_type>() == 0);
                let left = left / mem::size_of::<$data_type>();
                if left + Self::LEN > data_length {
                    SimdPartition::new_all_scalar(data_length)
                } else {
                    let right = (data_length - left) % Self::LEN;
                    SimdPartition::new_simd(left, right, data_length)
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
            where
                F: FnMut($data_type) -> $data_type,
            {
                let mut array = self.to_array();
                for n in &mut array {
                    *n = op(*n);
                }
                Self::from_array(array)
            }

            #[inline]
            fn iter_over_complex_vector<F>(self, mut op: F) -> Self
            where
                F: FnMut(Complex<$data_type>) -> Complex<$data_type>,
            {
                let mut array = self.to_complex_array();
                for n in &mut array[0..Self::LEN / 2] {
                    *n = op(*n);
                }
                Self::from_complex_array(array)
            }

            #[inline]
            fn array_to_regs(array: &[$data_type]) -> &[Self] {
                if array.is_empty() {
                    return &[];
                }

                assert_eq!(
                    get_alignment_offset(array.as_ptr() as usize, mem::size_of::<Self>()),
                    0
                );
                super::transmute_slice(array)
            }

            #[inline]
            fn array_to_regs_mut(array: &mut [$data_type]) -> &mut [Self] {
                if array.is_empty() {
                    return &mut [];
                }

                assert_eq!(
                    get_alignment_offset(array.as_ptr() as usize, mem::size_of::<Self>()),
                    0
                );
                super::transmute_slice_mut(array)
            }

            #[inline]
            fn load(array: &[$data_type], idx: usize) -> Self {
                Self::from_slice_unaligned(&array[idx..idx + Self::LEN])
            }

            #[inline]
            fn store(self, array: &mut [$data_type], index: usize) {
                self.write_to_slice_unaligned(&mut array[index..index + Self::LEN])
            }

            #[inline]
            fn extract(self, idx: usize) -> $data_type {
                $mod::$reg::extract(self, idx)
            }

            #[inline]
            fn splat(value: $data_type) -> Self {
                Self::splat(value)
            }
        }
    };
}

#[cfg(feature = "use_avx512")]
pub mod avx512;
#[cfg(feature = "use_avx512")]
simd_generic_impl!(f32, avx512::f32x16); // Type isn't implemented in simd
#[cfg(feature = "use_avx512")]
simd_generic_impl!(f64, avx512::f64x8); // Type isn't implemented in simd

#[cfg(all(feature = "use_avx2", target_feature = "avx2"))]
pub mod avx;
#[cfg(all(feature = "use_avx2", target_feature = "avx2"))]
simd_generic_impl!(f32, avx::f32x8);
#[cfg(all(feature = "use_avx2", target_feature = "avx2"))]
simd_generic_impl!(f64, avx::f64x4);

#[cfg(feature = "use_sse2")]
pub mod sse;
#[cfg(all(feature = "use_sse2", target_feature = "sse2"))]
simd_generic_impl!(f32, sse::f32x4);
#[cfg(all(feature = "use_sse2", target_feature = "sse2"))]
simd_generic_impl!(f64, sse::f64x2);

#[cfg(feature = "use_simd")]
mod approximations;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
mod x86;

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[macro_use]
mod other_cpu;

mod approx_fallback;
pub mod fallback;

simd_generic_impl!(f32, fallback::f32x4);
simd_generic_impl!(f64, fallback::f64x2);

pub struct RegType<Reg> {
    _type: std::marker::PhantomData<Reg>,
}

impl<Reg> RegType<Reg> {
    pub fn new() -> Self {
        RegType {
            _type: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_alignment_offset_test() {
        let reg_len = mem::size_of::<fallback::f64x2>();
        assert_eq!(reg_len, 16);
        assert_eq!(get_alignment_offset(0, reg_len), 0);
        assert_eq!(get_alignment_offset(8, reg_len), 8);
        assert_eq!(get_alignment_offset(16, reg_len), 0);
        assert_eq!(get_alignment_offset(24, reg_len), 8);
    }

    #[cfg(all(feature = "use_avx2", target_feature = "avx2"))]
    mod avx {
        use super::super::*;
        #[test]
        fn get_alignment_offset_test() {
            let reg_len = mem::size_of::<f64x4>();
            assert_eq!(reg_len, 32);
            assert_eq!(get_alignment_offset(0, reg_len), 0);
            assert_eq!(get_alignment_offset(8, reg_len), 8);
            assert_eq!(get_alignment_offset(16, reg_len), 16);
            assert_eq!(get_alignment_offset(24, reg_len), 24);
            assert_eq!(get_alignment_offset(32, reg_len), 0);
            assert_eq!(get_alignment_offset(40, reg_len), 8);
        }
    }
}
