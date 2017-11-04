use numbers::*;
use std::mem;
use std::ops::*;
use std;

/// SIMD methods which have `f32` or `f64` specific implementation.
pub trait Simd<T>: Sized
    where T: Sized + Sync + Send
{
    /// The type of real valued array which matches a SIMD register.
    type Array;
    
    /// SIMD register to array.
    fn to_array(self) -> Self::Array;
    
    /// The type of complex valued array which matches a SIMD register.
    type ComplexArray;
    
    /// Number of elements in a SIMD register.
    fn len() -> usize;
    
    /// Loads a SIMD register from an array. If the end of the array is approached
    /// then the load code wraps around to the beginning and starts to load the first 
    /// elements again.
    fn load_wrap_unchecked(array: &[T], idx: usize) -> Self;
    
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
    
    /// Same as `complex_abs_squared` but stores the result
    /// as complex number
    fn complex_abs_squared2(self) -> Self;
    
    /// Same as `complex_abs` but stores the result
    /// as complex number where the imaginary part is 0.
    fn complex_abs2(self) -> Self;
    
    /// Calculates the square root of the register.
    fn sqrt(self) -> Self;
    
    /// Stores the first half of the vector in an array.
    /// Useful e.g. in combination with `complex_abs_squared`.
    fn store_half_unchecked(self, target: &mut [T], index: usize);
    
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
}

/// Dirty workaround since the stdsimd doesn't implement conversion traits (yet?).
pub trait SimdFrom<T> {
    fn regfrom(T) -> Self;
}

/// SIMD methods which share their implementation independent if it's a `f32` or `f64` register.
pub trait SimdGeneric<T>: Simd<T> 
    + Add<Self, Output=Self> + Sub<Self, Output=Self> + Mul<Self, Output=Self> + Div<Self, Output=Self> 
    + Copy + Clone + Sync + Send + Sized + Zero
    where T: Sized + Sync + Send
{
    /// On some CPU architectures memory access needs to be aligned or otherwise
    /// the process will crash. This method takes a vector an divides it in three ranges:
    /// beginning, center, end. Beginning and end may not be loaded directly as SIMD registers.
    /// Center will contain most of the data.
    fn calc_data_alignment_reqs(array: &[T]) -> (usize, usize, usize);

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
    fn iter_over_vector<F>(self, op: F) -> Self where F: FnMut(T) -> T;

    /// Executed the given function on each element of the register.
    /// Register elements are assumed to be complex valued.
    fn iter_over_complex_vector<F>(self, op: F) -> Self where F: FnMut(Complex<T>) -> Complex<T>;

    /// Converts an array slice into a slice of SIMD registers.
    fn array_to_regs(array: &[T]) -> &[Self];

    /// Converts a mutable array slice into a slice of mutable SIMD registers.
    fn array_to_regs_mut(array: &mut [T]) -> &mut [Self];

    /// Loads a SIMD register from an array without any bound checks.
    fn load_unchecked(array: &[T], idx: usize) -> Self;

    /// Stores a SIMD register into an array without any bound checks.
    fn store_unchecked(self, target: &mut [T], index: usize);

    /// Returns one element from the register.
    fn extract(self, idx: u32) -> T;

    /// Creates a new SIMD register where every element equals `value`.
    fn splat(value: T) -> Self;

    /// Stores a SIMD register into an array.
    fn store(self, array: &mut [T], index: usize);
}

/// Approximated and faster implementation of some numeric standard function.
/// The approximations are implemented based on SIMD registers.
/// Refer to the documentation of the `ApproximatedOps` trait (which is part of 
/// the public API of this lib) for some information about accuracy and speed.
pub trait SimdApproximations<T>: Simd<T>
    where T: Sized + Sync + Send
{
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

/// Private struct copied over from the `simd` crate to implement the `load_unchecked` 
/// and `store_unchecked` methods for the `SimdGeneric` trait.
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

#[cfg(any(feature = "doc", feature= "use_sse"))]
mod sse;

//mod approximations;

pub mod fallback;

pub use self::fallback::{Reg32, Reg64};

//#[cfg(any(feature = "doc", not(any(feature = "use_avx", feature="use_sse"))))]
mod approx_fallback;

simd_generic_impl!(f32, Reg32);
simd_generic_impl!(f64, Reg64);

pub struct RegType<Reg> {
    _type: std::marker::PhantomData<Reg>
}

impl<Reg> RegType<Reg> {
    pub fn new() -> Self {
        RegType { 
            _type: std::marker::PhantomData 
        }
    }
}

/// Return the best available SIMD register.
macro_rules! get_reg(
    ($type: ident) => {
        if cfg_feature_enabled!("avx2") && cfg!(feature="use_avx2") {
            RegType::<$type::Reg>::new()
        } else if cfg_feature_enabled!("avx") && cfg!(feature="use_avx") {
            RegType::<$type::Reg>::new()
        } else if cfg_feature_enabled!("sse2") && cfg!(feature="use_sse") {
            RegType::<$type::Reg>::new()
        } else {
            RegType::<$type::Reg>::new()
        }
    }
);

/// Selects a SIMD register type and passes it as 2nd argument to a function.
/// The macro tries to mimic the Rust syntax of a method call.
macro_rules! sel_reg(
    ($self_:ident.$method: ident::<$type: ident>($($args: expr),*)) => {
        $self_.$method(get_reg!($type), $($args),*)
    }
);