//! This code will be used if no GPU support is selected. This way the feature flag only
//! needs to be considered in a small section of the lib and the rest remains unaffected.
use super::GpuSupport;
use numbers::*;
use std::ops::Range;

pub type Gpu32 = f32;

pub type Gpu64 = f64;

/// This trait is required to interface between `basic_dsp` and `opencl`.
/// Without the feature flag `use_gpu` is will default to a `num` trait so
/// that other code can always rely that this type is defined.
pub trait GpuFloat: Float {}

/// This trait is required to interface between `basic_dsp` and `opencl`.
/// Without the feature flag `use_gpu` is will default to a `num` trait so
/// that other code can always rely that this type is defined.
pub trait GpuRegTrait: Float {}

impl<T> GpuFloat for T where T: Float {}

impl<T> GpuRegTrait for T where T: Float {}

impl<T: RealNumber> GpuSupport<T> for T {
    fn has_gpu_support() -> bool {
        return false;
    }

    fn gpu_convolve_vector(_: bool, _: &[T], _: &mut [T], _: &[T]) -> Option<Range<usize>> {
        None
    }

    fn is_supported_fft_len(_: bool, _: usize) -> bool {
        false
    }

    fn fft(_: bool, _: &[T], _: &mut [T], _: bool) {
        panic!("GPU support not available, call `has_gpu_support` first.")
    }

    fn overlap_discard(
        _: &mut [T],
        _: &mut [T],
        _: &mut [T],
        _: &[T],
        _: usize,
        _: usize,
    ) -> usize {
        panic!("GPU support not available, call `has_gpu_support` first.")
    }
}
