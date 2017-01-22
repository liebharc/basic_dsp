use super::GpuSupport;
use RealNumber;

impl<T: RealNumber> GpuSupport<T> for T {
    fn has_gpu_support() -> bool {
       return false;
    }

    fn gpu_convolve_vector(_: &[T], _: &mut [T], _: &[T]) {
        panic!("This version was compiled without GPU support")
    }
}
