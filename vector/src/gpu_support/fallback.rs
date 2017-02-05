use super::GpuSupport;
use RealNumber;
use num::traits::Float;
use std::ops::Range;

pub type Gpu32 = f32;

pub type Gpu64 = f64;

pub trait GpuRegTrait : Float {}

impl<T> GpuRegTrait for T
    where T: Float {}

impl<T: RealNumber> GpuSupport<T> for T {
    fn has_gpu_support() -> bool {
       return false;
    }

    fn gpu_convolve_vector(_: bool, _: &[T], _: &mut [T], _: &[T]) -> Option<Range<usize>> {
        None
    }
}
