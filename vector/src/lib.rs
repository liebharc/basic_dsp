//! Basic digital signal processing (DSP) operations
//!
//! Digital signal processing based on real or complex vectors in time or frequency domain.
//! Vectors are expected to typically have a size which is at least in the order
//! of magnitude of a couple of thousand elements. This crate tries to balance between a clear
//! API and performance in terms of processing speed.
//!
//! Take this example:
//!
//! ```
//! # extern crate num_complex;
//! # extern crate basic_dsp_vector;
//! # use basic_dsp_vector::*;
//! # fn main() {
//! let mut vector1 = vec!(1.0, 2.0).to_real_time_vec();
//! let vector2 = vec!(10.0, 11.0).to_real_time_vec();
//! vector1.add(&vector2).expect("Ignoring error handling in examples");
//! # }
//!
//! ```
//! If `vector2` would be a complex or frequency vector then this won't compile. The type mismatch
//! indicates that a conversation is missing and that this might be a programming mistake. This lib uses
//! the Rust type system to catch such errors.
//!
//! DSP algorithms are often executed in loops. If you work with large vectors you typically try to avoid
//! allocating buffers in every iteration. Preallocating buffers is a common practice to safe a little time
//! with every iteration later on, but also to avoid heap fragmentation. At the same time it's a tedious task
//! to calculate the right buffer sizes for all operations. As an attempt to provide a more convenient solution
//! buffer types exist which don't preallocate, but store temporary memory segments so that they can be reused in the
//! next iteration. Here is an example:
//!
//! ```
//! # use std::f32;
//! # use basic_dsp_vector::*;
//! let vector = vec!(1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254).to_complex_time_vec();
//! let mut buffer = SingleBuffer::new();
//! let _ = vector.fft(&mut buffer);
//! ```
//!
//! The vector types don't distinguish between the shapes 1xN or Nx1. This is a difference to other
//! conventions such as in MATLAB or GNU Octave.
//! The reason for this decision is that most operations are only defined if the shape of the
//! vector matches. So it appears to be more practical and clearer to implement the few operations
//! where the arguments can be of different shapes as seperate methods. The methods `mul` and `dot_product`
//! are one example for this.
//!
//! The trait definitions in this lib can look complex and might be overwhelming at the beginning.
//! There is a wide range of DSP vectors, e.g. a slice can be DSP vector, a boxed array can be a DSP vector,
//! a standard vector can be a DSP vector and so on. This lib tries to work with all of that and tries
//! to allow all those different DSP vector types to work together. The price for this flexibility is a more complex
//! trait definition. As a mental model, this is what the traits are specifiying:
//! Whenever you have a complex vector in time domain, it's binary operations will work with all other
//! complex vectors in time domain, but not with real valued vectors or frequency domain vectors.
//! And the type `GenDspVec` serves as wild card at compile time since it defers all checks to run time.

extern crate arrayvec;
#[cfg(any(feature = "doc", feature = "use_gpu"))]
extern crate clfft;
#[cfg(feature = "std")]
extern crate crossbeam;
#[cfg(feature = "std")]
#[macro_use]
extern crate lazy_static;
#[cfg(feature = "std")]
extern crate num_cpus;
#[cfg(feature = "std")]
extern crate time;
#[cfg(feature = "std")]
extern crate linreg;
extern crate num_complex;
extern crate num_traits;
#[cfg(any(feature = "doc", feature = "use_gpu"))]
extern crate ocl;
extern crate rustfft;
#[cfg(any(feature = "doc", feature = "use_sse", feature = "use_avx"))]
extern crate simd;
#[macro_use]
mod simd_extensions;
pub mod conv_types;
mod multicore_support;
mod vector_types;
pub mod window_functions;
pub use multicore_support::MultiCoreSettings;
pub use multicore_support::print_calibration;
pub use vector_types::*;
mod gpu_support;
mod inline_vector;
use numbers::*;
use std::ops::Range;

pub mod numbers {
    //! Traits from the `num` crate which are used inside `basic_dsp` and extensions to those traits.
    use gpu_support::{Gpu32, Gpu64, GpuFloat, GpuRegTrait};
    pub use num_complex::Complex;
    use num_traits;
    pub use num_traits::Float;
    pub use num_traits::Num;
    pub use num_traits::One;
    use rustfft;
    #[cfg(any(
        feature = "use_sse",
        feature = "use_avx",
        feature = "use_avx512"
    ))]
    use simd;
    use simd_extensions;
    use simd_extensions::*;
    use std::fmt::Debug;

    /// A trait for a numeric value which at least supports a subset of the operations defined in this crate.
    /// Can be an integer or a floating point number. In order to have support for all operations in this crate
    /// a must implement the `RealNumber`.
    pub trait DspNumber:
        Num
        + Copy
        + Clone
        + Send
        + Sync
        + ToSimd
        + Debug
        + num_traits::Signed
        + num_traits::FromPrimitive
        + rustfft::FFTnum
        + 'static
    {
}
    impl<T> DspNumber for T where
        T: Num
            + Copy
            + Clone
            + Send
            + Sync
            + ToSimd
            + Debug
            + num_traits::Signed
            + num_traits::FromPrimitive
            + rustfft::FFTnum
            + 'static
    {}

    /// Associates a number type with a SIMD register type.
    pub trait ToSimd: Sized + Sync + Send {
        /// Type for the SIMD register on the CPU.
        type RegFallback: SimdGeneric<Self>;
        type RegSse: SimdGeneric<Self>;
        type RegAvx: SimdGeneric<Self>;
        type RegAvx512: SimdGeneric<Self>;
        /// Type for the SIMD register on the GPU. Defaults to an arbitrary type if GPU support is not
        /// compiled in.
        type GpuReg: GpuRegTrait;
    }

    impl ToSimd for f32 {
        type RegFallback = simd_extensions::fallback::f32x4;

        #[cfg(feature = "use_sse")]
        type RegSse = simd::f32x4;
        #[cfg(not(feature = "use_sse"))]
        type RegSse = simd_extensions::fallback::f32x4;

        #[cfg(feature = "use_avx")]
        type RegAvx = simd::x86::avx::f32x8;
        #[cfg(not(feature = "use_avx"))]
        type RegAvx = simd_extensions::fallback::f32x4;

        #[cfg(feature = "use_avx512")]
        type RegAvx512 = stdsimd::simd::f32x16; // Type is missing in SIMD
        #[cfg(not(feature = "use_avx512"))]
        type RegAvx512 = simd_extensions::fallback::f32x4;

        type GpuReg = Gpu32;
    }

    impl ToSimd for f64 {
        type RegFallback = simd_extensions::fallback::f64x2;

        #[cfg(feature = "use_sse")]
        type RegSse = simd::x86::sse2::f64x2;
        #[cfg(not(feature = "use_sse"))]
        type RegSse = simd_extensions::fallback::f64x2;

        #[cfg(feature = "use_avx")]
        type RegAvx = simd::x86::avx::f64x4;
        #[cfg(not(feature = "use_avx"))]
        type RegAvx = simd_extensions::fallback::f64x2;

        #[cfg(feature = "use_avx512")]
        type RegAvx512 = stdsimd::simd::f64x8; // Type is missing in SIMD
        #[cfg(not(feature = "use_avx512"))]
        type RegAvx512 = simd_extensions::fallback::f64x2;

        type GpuReg = Gpu64;
    }

    /// A real floating pointer number intended to abstract over `f32` and `f64`.
    pub trait RealNumber: Float + DspNumber + GpuFloat + num_traits::FloatConst {}
    impl<T> RealNumber for T where T: Float + DspNumber + GpuFloat + num_traits::FloatConst {}

    /// This trait is necessary so that we can define zero for types outside this crate.
    /// It calls the `num_traits::Zero` trait where possible.
    pub trait Zero {
        fn zero() -> Self;
    }

    impl<T> Zero for T
    where
        T: DspNumber,
    {
        fn zero() -> Self {
            <Self as num_traits::Zero>::zero()
        }
    }

    impl<T> Zero for Complex<T>
    where
        T: DspNumber,
    {
        fn zero() -> Self {
            <Self as num_traits::Zero>::zero()
        }
    }
}

/// Transmutes a slice. Both S and D must be `#[repr(C)]`.
/// The code panics if the slice has a length which doesn't allow conversion.
fn transmute_slice<S, D>(source: &[S]) -> &[D] {
    let len = get_target_slice_len::<S, D>(source);
    unsafe {
        let trans: &[D] = &*(source as *const [S] as *const [D]);
        std::slice::from_raw_parts(trans.as_ptr(), len)
    }
}

/// Transmutes a mutable slice. Both S and D must be `#[repr(C)]`.
/// The code panics if the slice has a length which doesn't allow conversion.
fn transmute_slice_mut<S, D>(source: &mut [S]) -> &mut[D] {
    let len = get_target_slice_len::<S, D>(source);
    unsafe {
        let trans: &mut [D] = &mut *(source as *mut [S] as *mut [D]);
        std::slice::from_raw_parts_mut(trans.as_mut_ptr(), len)
    }
}

/// Helper method which finds the length of a target slice and also perform checks on the length
fn get_target_slice_len<S, D>(source: &[S]) -> usize {
    let to_larger_type = std::mem::size_of::<D>() >= std::mem::size_of::<S>();
    if to_larger_type {
        assert_eq!(std::mem::size_of::<D>() % std::mem::size_of::<S>(), 0);
        let ratio = std::mem::size_of::<D>() / std::mem::size_of::<S>();
        assert_eq!(source.len() % ratio, 0);
        source.len() / ratio
    } else {
        assert_eq!(std::mem::size_of::<S>() % std::mem::size_of::<D>(), 0);
        let ratio = std::mem::size_of::<S>() / std::mem::size_of::<D>();
        source.len() * ratio
    }
}

// Returns a complex slice from a real slice
fn array_to_complex<T>(array: &[T]) -> &[Complex<T>] {
    transmute_slice(array)
}

// Returns a complex slice from a real slice
fn array_to_complex_mut<T>(array: &mut [T]) -> &mut [Complex<T>] {
    transmute_slice_mut(array)
}

/// Copies memory inside a slice
fn memcpy<T: Copy>(data: &mut [T], from: Range<usize>, to: usize) {
    use std::ptr::copy;
    assert!(from.start <= from.end);
    assert!(from.end <= data.len());
    assert!(to <= data.len() - (from.end - from.start));
    unsafe {
        let ptr = data.as_mut_ptr();
        copy(
            ptr.add(from.start),
            ptr.add(to),
            from.end - from.start,
        )
    }
}

// Zeros a range within the slice
fn memzero<T: Copy>(data: &mut [T], range: Range<usize>) {
    use std::ptr::write_bytes;
    assert!(range.start <= range.end);
    assert!(range.end <= data.len());
    unsafe {
        let ptr = data.as_mut_ptr();
        write_bytes(ptr.add(range.start), 0, range.end - range.start);
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
        let reg = <f32 as ToSimd>::RegFallback::splat(1.0);
        let sum = reg.sum_real();
        assert!(sum > 0.0);
    }
}
