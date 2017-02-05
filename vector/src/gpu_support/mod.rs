#[cfg(any(feature = "doc", feature="use_gpu"))]
mod ocl;

#[cfg(any(feature = "doc", feature="use_gpu"))]
pub use self::ocl::*;

#[cfg(any(feature = "doc", not(feature="use_gpu")))]
mod fallback;

#[cfg(any(feature = "doc", not(feature="use_gpu")))]
pub use self::fallback::*;

use RealNumber;
use std::ops::Range;

/// Trait which adds GPU support to types like `f32` and `f64`.
pub trait GpuSupport<T: RealNumber> {
    /// Indicates whether or not GPU support is available for this type. All other
    /// methods will panic if there is no GPU support so better check this one first.
    fn has_gpu_support() -> bool;

    /// Convolve a vector on the GPU.
    fn gpu_convolve_vector(
        is_complex: bool,
        source: &[T],
        target: &mut [T],
        imp_resp: &[T]) -> Option<Range<usize>>;
}
