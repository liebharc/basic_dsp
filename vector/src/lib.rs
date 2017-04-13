//! Basic digital signal processing (DSP) operations
//!
//! Digital signal processing based on real or complex vectors in time or frequency domain.
//! Vectors are expected to typically have a size which is at least in the order
//! of magnitude of a couple of thousand elements. This crate tries to balance between a clear
//! API and performance in terms of processing speed.
//! This project started as small pet project to learn more about DSP, CPU architecture and Rust.
//! Since learning
//! involves making mistakes, don't expect things to be flawless or even close to flawless.
//!
//! This library isn't suited - from my point of view - for game programming. If you are looking
//! for vector types to do
//! 2D or 3D graphics calculations then you unfortunately have to continue with your search.
//! However there seem to be
//! a lot of suitable crates on `crates.io` for you.
//!
//! The vector types don't distinguish between 1xN or Nx1. This is a difference to other
//! conventions such as in MATLAB or GNU Octave.
//! The reason for this decision is that it seems to be more practical to ignore the
//! shape of the vector.
//!
//! Right now the library uses pretty aggressive parallelization. So this means that it will
//! keep all CPU cores busy
//! even if the performance gain is minimal e.g. because the multi core overhead is nearly as
//! large as the performance boost
//! of multiple cores. In future there will be likely an option which tells the library how it
//! should balance between processing time
//! and CPU utilization. The library also avoids to allocate and free memory so it allocates
//! all of the required temporary memory when a new vector
//! is constructed. Therefore the library is likely not suitable for devices which are tight
//! on memory. On normal desktop computers there is usually plenty of
//! memory available so that the optimization focus is on decreasing the processing time
//! for every (common) operation and to spent little time with memory allocations.

#![cfg_attr(not(feature="std"), no_std)]
#[cfg(not(feature="std"))]
extern crate core as std;

#[cfg(any(feature = "doc", feature="use_sse", feature="use_avx"))]
extern crate simd;
#[cfg(any(feature = "doc", feature="use_gpu"))]
extern crate ocl;
#[cfg(feature="std")]
extern crate num_cpus;
#[cfg(feature="std")]
extern crate crossbeam;
extern crate num_traits;
extern crate num_complex;
extern crate rustfft;
#[cfg(any(feature = "doc", feature="use_gpu"))]
extern crate clfft;
extern crate arrayvec;
mod vector_types;
mod multicore_support;
mod simd_extensions;
pub mod window_functions;
pub mod conv_types;
pub use vector_types::*;
pub use multicore_support::MultiCoreSettings;
mod gpu_support;
use std::ops::*;
use std::mem;
mod inline_vector;

pub mod numbers {
    //! Traits from the `num` crate which are used inside `basic_dsp`. In future
    //! the `RealNumber` and `Zero` trait will likely be found here too.
    pub use num_traits::Float;
    pub use num_traits::One;
    pub use num_complex::Complex;
    pub use num_traits::Num;
    use ToSimd;
    use std::fmt::Debug;
    use num_traits;

    /// A trait for a numeric value which at least supports a subset of the operations defined in this crate.
    /// Can be an integer or a floating point number. In order to have support for all operations in this crate
    /// a must implement the `RealNumber`.
    pub trait DspNumber
        : Num + Copy + Clone + Send + Sync + ToSimd + Debug + num_traits::Signed + num_traits::FromPrimitive
    {
    }
    impl<T> DspNumber for T
        where T: Num + Copy + Clone + Send + Sync + ToSimd + Debug + num_traits::Signed + num_traits::FromPrimitive
    {
    }
}

use numbers::*;

use simd_extensions::*;
use gpu_support::{Gpu32, Gpu64, GpuRegTrait, GpuFloat};

/// Associates a number type with a SIMD register type.
pub trait ToSimd: Sized + Sync + Send {
    /// Type for the SIMD register on the CPU.
    type Reg: Simd<Self> + SimdGeneric<Self> + SimdApproximations<Self>
        + Copy + Sync + Send
        + Add<Output = Self::Reg> + Sub<Output = Self::Reg> + Mul<Output = Self::Reg> + Div<Output = Self::Reg> + Zero;
    /// Type for the SIMD register on the GPU. Defaults to an arbitrary type if GPU support is not
    /// compiled in.
    type GpuReg: GpuRegTrait;
}

impl ToSimd for f32 {
    type Reg = Reg32;
    type GpuReg = Gpu32;
}

impl ToSimd for f64 {
    type Reg = Reg64;
    type GpuReg = Gpu64;
}

/// A real floating pointer number intended to abstract over `f32` and `f64`.
pub trait RealNumber
    : Float + numbers::DspNumber + GpuFloat + num_traits::FloatConst
{
}
impl<T> RealNumber for T
    where T: numbers::DspNumber + GpuFloat + num_traits::FloatConst
{
}

/// This trait is necessary so that we can define zero for types outside this crate.
/// It calls the `num_traits::Zero` trait where possible.
pub trait Zero {
    fn zero() -> Self;
}

impl<T> Zero for T
    where T: numbers::DspNumber {
    fn zero() -> Self {
        <Self as num_traits::Zero>::zero()
    }
}

impl<T> Zero for Complex<T>
    where T: numbers::DspNumber {
    fn zero() -> Self {
        <Self as num_traits::Zero>::zero()
    }
}

fn array_to_complex<T>(array: &[T]) -> &[Complex<T>] {
    unsafe {
        let len = array.len();
        if len % 2 != 0 {
            panic!("Argument must have an even length");
        }
        let trans: &[Complex<T>] = mem::transmute(array);
        &trans[0..len / 2]
    }
}

fn array_to_complex_mut<T>(array: &mut [T]) -> &mut [Complex<T>] {
    unsafe {
        let len = array.len();
        if len % 2 != 0 {
            panic!("Argument must have an even length");
        }
        let trans: &mut [Complex<T>] = mem::transmute(array);
        &mut trans[0..len / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simd_extensions::Simd;

    #[test]
    fn to_simd_test() {
        // This is more a check for syntax. So if it compiles
        // then the test already passes. The assert is then only
        // a sanity check.
        let reg = <f32 as ToSimd>::Reg::splat(1.0);
        let sum = reg.sum_real();
        assert!(sum > 0.0);
    }
}
