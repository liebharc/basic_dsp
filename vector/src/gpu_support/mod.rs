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

    /// Indicates whether or not the parameters are supported by the FFT implementation.
    fn is_supported_fft_len(is_complex: bool, len: usize) -> bool;

    /// FFT on the GPU.
    fn fft(
        is_complex: bool,
        source: &[T],
        target: &mut [T],
        reverse: bool);

    // Calculates the FFT of `source`, multiplies `freq_resp` and stores the IFFT in `target`.
    fn mul_freq_response(
        source: &[T],
        target: &mut [T],
        freq_resp: &[T]);
}
