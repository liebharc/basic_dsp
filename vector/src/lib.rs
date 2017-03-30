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
extern crate num_cpus;
extern crate crossbeam;
extern crate num;
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
use num::traits::Float;
use std::fmt::Debug;
use std::ops::*;
use num::complex::Complex;
use std::mem;
use arrayvec::ArrayVec;

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
    : Float + Copy + Clone + Send + Sync + ToSimd + Debug + num::Signed + num::FromPrimitive + GpuFloat
     + num::traits::FloatConst
{
}
impl<T> RealNumber for T
    where T: Float + Copy + Clone + Send + Sync + ToSimd + Debug + num::Signed + num::FromPrimitive + GpuFloat
		    + num::traits::FloatConst
{
}

/// This trait is necessary so that we can define zero for types outside this crate.
/// It calls the `num::Zero` trait where possible.
pub trait Zero {
    fn zero() -> Self;
}

impl<T> Zero for T
    where T: RealNumber {
    fn zero() -> Self {
        <Self as num::Zero>::zero()
    }
}

impl<T> Zero for num::Complex<T>
    where T: RealNumber {
    fn zero() -> Self {
        <Self as num::Zero>::zero()
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

/// A type which internally switches between stack and heap allocation.
/// This is supposed to perform faster but the main reason is that this
/// way we automatically have a limited stack allocation available on systems
/// without heap, and on systems with heap allocation we don't have to worry
/// about introducing artifical limits.
///
/// Thanks to: http://stackoverflow.com/questions/27859822/alloca-variable-length-arrays-in-rust
pub enum InlineVector<T> {
    Inline(ArrayVec<[T; 64]>),
    Dynamic(Vec<T>),
}

impl<T> InlineVector<T>
    where T: Copy {
    fn of_size(default: T, n: usize) -> InlineVector<T> {
        let mut result = Self::with_capacity(n);
        for _ in 0..n {
            result.push(default);
        }

        result
    }
}

impl<T> InlineVector<T> {
    fn with_capacity(n: usize) -> InlineVector<T> {
        if n <= 64 {
            InlineVector::Inline(ArrayVec::<[T; 64]>::new())
        } else {
            InlineVector::Dynamic(
                Vec::with_capacity(n)
            )
        }
    }

    fn with_elem(elem: T) -> InlineVector<T> {
        let mut vector = Self::with_capacity(1);
        vector.push(elem);
        vector
    }

    fn empty() -> InlineVector<T> {
        Self::with_capacity(0)
    }

    fn push(&mut self, elem: T) {
        match self {
            &mut InlineVector::Inline(ref mut v) => {
                let res = v.push(elem);
                if res.is_some() {
                    panic!("InlineVector capacity exceeded, please open a defect against `basic_dsp`");
                }
            },
            &mut InlineVector::Dynamic(ref mut v) => v.push(elem)
        };
    }

    fn pop(&mut self) -> Option<T> {
        match self {
            &mut InlineVector::Inline(ref mut v) => {
                v.pop()
            },
            &mut InlineVector::Dynamic(ref mut v) => v.pop()
        }
    }

    fn len(&self) -> usize {
        match self {
            &InlineVector::Inline(ref v) => v.len(),
            &InlineVector::Dynamic(ref v) => v.len()
        }
    }

    fn capacity(&self) -> usize {
        match self {
            &InlineVector::Inline(ref v) => v.capacity(),
            &InlineVector::Dynamic(ref v) => v.capacity()
        }
    }
}

impl<T: Zero + Clone> InlineVector<T> {
    fn try_resize(&mut self, len: usize) -> VoidResult {
        match self {
            &mut InlineVector::Inline(ref v) => {
                if v.capacity() >= len {
                    Ok(())
                } else {
                    Err(ErrorReason::TypeCanNotResize)
                }
            },
            &mut InlineVector::Dynamic(ref mut v) => {
                if v.capacity() >= len {
                    v.resize(len, T::zero());
                Ok(())
                } else {
                    // We could increase the vector capacity, but then
                    // Inline and Dynamic would behave very different and we want
                    // to avoid that
                    Err(ErrorReason::TypeCanNotResize)
                }
            }
        }
    }
}

/// A buffer which stores a single inline vector and never shrinks.
struct InternalBuffer<T>
    where T: RealNumber
{
    temp: InlineVector<T>,
}

impl<T> Index<usize> for InlineVector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match self {
            &InlineVector::Inline(ref v) => &v[index],
            &InlineVector::Dynamic(ref v) => &v[index]
        }
    }
}

impl<T> IndexMut<usize> for InlineVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match self {
            &mut InlineVector::Inline(ref mut v) => &mut v[index],
            &mut InlineVector::Dynamic(ref mut v) => &mut v[index]
        }
    }
}

impl<T> Index<RangeFull> for InlineVector<T> {
    type Output = [T];

    fn index(&self, _index: RangeFull) -> &[T] {
        match self {
            &InlineVector::Inline(ref v) => &v[..],
            &InlineVector::Dynamic(ref v) => &v[..]
        }
    }
}

impl<T> IndexMut<RangeFull> for InlineVector<T> {
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        match self {
            &mut InlineVector::Inline(ref mut v) => &mut v[..],
            &mut InlineVector::Dynamic(ref mut v) => &mut v[..]
        }
    }
}

impl<T> Index<RangeFrom<usize>> for InlineVector<T> {
    type Output = [T];

    fn index(&self, index: RangeFrom<usize>) -> &[T] {
        match self {
            &InlineVector::Inline(ref v) => &v[index],
            &InlineVector::Dynamic(ref v) => &v[index]
        }
    }
}

impl<T> IndexMut<RangeFrom<usize>> for InlineVector<T> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
        match self {
            &mut InlineVector::Inline(ref mut v) => &mut v[index],
            &mut InlineVector::Dynamic(ref mut v) => &mut v[index]
        }
    }
}

impl<T> Index<RangeTo<usize>> for InlineVector<T> {
    type Output = [T];

    fn index(&self, index: RangeTo<usize>) -> &[T] {
        match self {
            &InlineVector::Inline(ref v) => &v[index],
            &InlineVector::Dynamic(ref v) => &v[index]
        }
    }
}

impl<T> IndexMut<RangeTo<usize>> for InlineVector<T> {
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
        match self {
            &mut InlineVector::Inline(ref mut v) => &mut v[index],
            &mut InlineVector::Dynamic(ref mut v) => &mut v[index]
        }
    }
}

impl<T: Clone> Clone for InlineVector<T> {
    fn clone(&self) -> Self {
         match self {
            &InlineVector::Inline(ref v) => InlineVector::Inline(v.clone()),
            &InlineVector::Dynamic(ref v) => InlineVector::Dynamic(v.clone())
        }
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
