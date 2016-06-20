use num::complex::Complex;

pub trait Simd<T> : Sized
    where T: Sized + Sync + Send
{
    fn array_to_regs(array: &[T]) -> &[Self];
    fn array_to_regs_mut(array: &mut [T]) -> &mut [Self];
    fn len() -> usize;
    fn load(array: &[T], idx: usize) -> Self;
    fn load_wrap(array: &[T], idx: usize) -> Self;
    fn from_complex(value: Complex<T>) -> Self;
    fn add_real(self, value: T) -> Self;
    fn add_complex(self, value: Complex<T>) -> Self;
    fn scale_real(self, value: T) -> Self;
    fn scale_complex(self, value: Complex<T>) -> Self;
    fn complex_abs_squared(self) -> Self;
    fn complex_abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn store(self, target: &mut [T], index: usize);
    fn store_half(self, target: &mut [T], index: usize);
    fn mul_complex(self, value: Self) -> Self;
    fn div_complex(self, value: Self) -> Self;
    fn sum_real(&self) -> T;
    fn sum_complex(&self) -> Complex<T>;
}

#[cfg(any(feature = "doc", target_feature="avx2"))]
pub mod avx;

#[cfg(any(feature = "doc", target_feature="avx2"))]
pub use self::avx::{Reg32,Reg64};

#[cfg(any(feature = "doc", all(target_feature = "sse3", not(target_feature = "avx2"))))]
pub mod sse;

#[cfg(any(feature = "doc", all(target_feature = "sse3", not(target_feature = "avx2"))))]
pub use self::sse::{Reg32,Reg64};

#[cfg(any(feature = "doc", not(any(target_feature = "avx2", target_feature="sse3"))))]
pub mod fallback;

#[cfg(any(feature = "doc", not(any(target_feature = "avx2", target_feature="sse3"))))]
pub use self::fallback::{Reg32,Reg64};