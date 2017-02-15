use super::GpuSupport;
use RealNumber;
use num::traits::Float;
use std::ops::Range;

pub type Gpu32 = f32;

pub type Gpu64 = f64;

pub trait GpuFloat : Float {}

pub trait GpuRegTrait : Float {}

impl<T> GpuFloat for T
    where T: Float {}

impl<T> GpuRegTrait for T
    where T: Float {}

impl<T: RealNumber> GpuSupport<T> for T {
    fn has_gpu_support() -> bool {
       return false;
    }

    fn gpu_convolve_vector(_: bool, _: &[T], _: &mut [T], _: &[T]) -> Option<Range<usize>> {
        None
    }

    fn is_supported_fft_len(_: bool, _: usize ) -> bool {
        false
    }

    fn fft(
        _: bool,
        _: &[T],
        _: &mut [T],
        _: bool) {
        panic!("GPU support not available, call `has_gpu_support` first.")
    }

    fn mul_freq_response(
        source: &[T],
        target: &mut [T],
        freq_resp: &[T]) {
        panic!("GPU support not available, call `has_gpu_support` first.")
    }
}
