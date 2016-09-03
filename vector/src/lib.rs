#![feature(box_syntax)]
#![feature(cfg_target_feature)]
//! Basic digital signal processing (DSP) operations
//!
//! Digital signal processing based on real or complex vectors in time or frequency domain. Vectors are expected to typically have a size which is at least in the order
//! of magnitude of a couple of thousand elements. This crate tries to balance between a clear API and performance in terms of processing speed.
//! This project started as small pet project to learn more about DSP, CPU architecture and Rust. Since learning
//! involves making mistakes, don't expect things to be flawless or even close to flawless.
//!
//! This library isn't suited - from my point of view - for game programming. If you are looking for vector types to do
//! 2D or 3D graphics calculations then you unfortunately have to continue with your search. However there seem to be
//! a lot of suitable crates on `crates.io` for you.
//!
//! The vector types don't distinguish between 1xN or Nx1. This is a difference to other conventions such as in MATLAB or GNU Octave.
//! The reason for this decision is that it seems to be more practical to ignore the shape of the vector.
//!
//! Right now the library uses pretty aggressive parallelization. So this means that it will keep all CPU cores busy
//! even if the performance gain is minimal e.g. because the multi core overhead is nearly as large as the performance boost
//! of multiple cores. In future there will be likely an option which tells the library how it should balance between processing time
//! and CPU utilization. The library also avoids to allocate and free memory so it allocates all of the required temporary memory when a new vector
//! is constructed. Therefore the library is likely not suitable for devices which are tight on memory. On normal desktop computers there is usually plenty of
//! memory available so that the optimization focus is on decreasing the processing time for every (common) operation and to spent little time with memory allocations.

/// Like `try!` but for operations returning a vector.
///
/// Operations which return a vector on success even return an error reason
/// together with a vector on failure. So even if the operation has failed the
/// vector can still be reused and thus memory allocation can be avoided. If
/// this is undesired then this macro can be used instead of `try!` to just
/// return the error reason.
#[macro_export]
macro_rules! try_vec {
    ( $ expr : expr ) => {
        match $expr {
            Ok(vec) => vec,
            Err((reason, _)) => return Err(reason)
        };
    };
}

extern crate simd;
extern crate num_cpus;
extern crate crossbeam;
extern crate num;
extern crate rustfft;
mod vector_types;
#[allow(dead_code)] // TODO: Remove this as soon as vector_types2 is able to replace vector_types (also rename at the same point)
pub mod vector_types2;
mod multicore_support;
mod simd_extensions;
pub mod window_functions;
pub mod conv_types;
pub mod combined_ops;
pub use vector_types::
    {
        DataVecDomain,
        DataVec,
        TransRes,
        VoidResult,
        ErrorReason,
        GenericVectorOps,
        RealVectorOps,
        ComplexVectorOps,
        TimeDomainOperations,
        FrequencyDomainOperations,
        SymmetricFrequencyDomainOperations,
        SymmetricTimeDomainOperations,
        GenericDataVec,
        ComplexFreqVector,
        ComplexTimeVector,
        RealTimeVector,
        RealFreqVector,
        DataVec32,
        RealTimeVector32,
        ComplexTimeVector32,
        RealFreqVector32,
        ComplexFreqVector32,
        DataVec64,
        RealTimeVector64,
        ComplexTimeVector64,
        RealFreqVector64,
        ComplexFreqVector64,
        Statistics,
        RededicateOps,
        ScaleOps,
        OffsetOps,
        DotProductOps,
        StatisticsOps,
        Convolution,
        ConvolutionOps,
        FrequencyMultiplication,
        CrossCorrelationOps,
        InterpolationOps,
        RealInterpolationOps,
        PaddingOption,
        VectorIter,
        ComplexIndex,
        ComplexIndexMut,
        RealIndex,
        RealIndexMut,
        InterleavedIndex,
        InterleavedIndexMut
    };
pub use multicore_support::MultiCoreSettings;
use num::traits::Float;
use std::fmt::Debug;

use simd_extensions::*;

/// Associates a number type with a SIMD register type.
pub trait ToSimd : Sized + Sync + Send {
    type Reg: Simd<Self> + SimdGeneric<Self> + Copy ;
}

impl ToSimd for f32 {
    type Reg = Reg32;
}

impl ToSimd for f64 {
    type Reg = Reg64;
}

 /// A real floating pointer number intended to abstract over `f32` and `f64`.
 pub trait RealNumber : Float + Copy + Clone + Send + Sync + ToSimd + Debug  { }
 impl<T> RealNumber for T
  where T: Float + Copy + Clone + Send + Sync + ToSimd + Debug  {}

#[cfg(test)]
mod tests {
    use super::*;
    use simd_extensions::Simd;

    fn void_try_method() -> Result<i32, ErrorReason> {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vector = DataVec32::from_array(false, DataVecDomain::Time, &array);
        let mut dest = DataVec32::from_array(false, DataVecDomain::Time, &array);
        let _ = try!(vector.get_magnitude(&mut dest));
        Ok(0)
    }

    fn scalar_try_method() -> Result<i32, ErrorReason> {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vector = DataVec32::from_array(false, DataVecDomain::Time, &array);
        let dest = DataVec32::from_array(false, DataVecDomain::Time, &array);
        let _ = try!(vector.complex_dot_product(&dest));
        Ok(0)
    }

    fn vec_try_method() -> Result<i32, ErrorReason> {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vector = DataVec32::from_array(false, DataVecDomain::Time, &array);
        let array = [1.0, 2.0, 3.0, 4.0];
        let dest = DataVec32::from_array(false, DataVecDomain::Time, &array);
        let _ = try_vec!(vector.add(&dest));
        Ok(0)
    }

    /// This test should make sure that there are convenient error handling
    /// methods available.
    #[test]
    fn construct_real_time_vector_32_test()
    {
        let res = void_try_method();
        assert!(res.is_err());
        let res = scalar_try_method();
        assert!(res.is_err());
        let res = vec_try_method();
        assert!(res.is_err());
    }

    #[test]
    fn to_simd_test()
    {
        // This is more a check for syntax. So if it compiles
        // then the test already passes. The assert is then only
        // a sanity check.
        let reg = <f32 as ToSimd>::Reg::splat(1.0);
        let sum = reg.sum_real();
        assert!(sum > 0.0);
    }
}
