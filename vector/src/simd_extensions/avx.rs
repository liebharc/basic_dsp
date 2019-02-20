use super::{Simd, SimdFrom};
use crate::numbers::*;
use std::mem;
use std::arch::x86_64::*;
pub use packed_simd::{f32x8, f64x4};
use packed_simd::{FromCast, i32x8, i64x4};

/// This value must be read in groups of 2 bits.
const SWAP_IQ_PS: i32 = 0b1011_0001;

const SWAP_IQ_PD: i32 = 0b0101;

impl Simd<f32> for f32x8 {
    type Array = [f32; 8];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 8];
        self.write_to_slice_unaligned(&mut target);
        target
    }

    type ComplexArray = [Complex<f32>; 4];

    const LEN: usize = 8;

    #[inline]
    fn from_complex(value: Complex<f32>) -> f32x8 {
        f32x8::new(
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
        )
    }

    #[inline]
    fn add_real(self, value: f32) -> f32x8 {
        let increment = f32x8::splat(value);
        self + increment
    }

    #[inline]
    fn add_complex(self, value: Complex<f32>) -> f32x8 {
        let increment = f32x8::from_complex(value);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f32) -> f32x8 {
        let scale_vector = f32x8::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f32>) -> f32x8 {
        let scaling_real = f32x8::splat(value.re);
        let scaling_imag = f32x8::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm256_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn mul_complex(self, value: f32x8) -> f32x8 {
        let scaling_real = f32x8::new(
            value.extract(0),
            value.extract(0),
            value.extract(2),
            value.extract(2),
            value.extract(4),
            value.extract(4),
            value.extract(6),
            value.extract(6),
        );
        let scaling_imag = f32x8::new(
            value.extract(1),
            value.extract(1),
            value.extract(3),
            value.extract(3),
            value.extract(5),
            value.extract(5),
            value.extract(7),
            value.extract(7),
        );
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm256_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn div_complex(self, value: f32x8) -> f32x8 {
        let scaling_imag = f32x8::new(
            self.extract(0),
            self.extract(0),
            self.extract(2),
            self.extract(2),
            self.extract(4),
            self.extract(4),
            self.extract(6),
            self.extract(6),
        );
        let scaling_real = f32x8::new(
            self.extract(1),
            self.extract(1),
            self.extract(3),
            self.extract(3),
            self.extract(5),
            self.extract(5),
            self.extract(7),
            self.extract(7),
        );
        let parallel = scaling_real * value;
        let shuffled = value.swap_iq();
        let cross = scaling_imag * shuffled;
        let mul: f32x8 = unsafe {
            mem::transmute(_mm256_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        };
        let square = shuffled * shuffled;
        let square_shuffled = square.swap_iq();
        let sum = square + square_shuffled;
        let div = mul / sum;
        div.swap_iq()
    }

    #[inline]
    fn complex_abs_squared(self) -> f32x8 {
        let squared = self * self;
        unsafe {
            mem::transmute(_mm256_hadd_ps(
                mem::transmute(squared),
                mem::transmute(squared),
            ))
        }
    }

    #[inline]
    fn complex_abs(self) -> f32x8 {
        let squared_sum = self.complex_abs_squared();
        squared_sum.sqrt()
    }

    #[inline]
    fn sqrt(self) -> f32x8 {
        self.sqrt()
    }

    #[inline]
    fn store_half(self, target: &mut [f32], index: usize) {
        target[index] = self.extract(0);
        target[index + 1] = self.extract(1);
        target[index + 2] = self.extract(4);
        target[index + 3] = self.extract(5);
    }

    #[inline]
    fn sum_real(&self) -> f32 {
        self.extract(0)
            + self.extract(1)
            + self.extract(2)
            + self.extract(3)
            + self.extract(4)
            + self.extract(5)
            + self.extract(6)
            + self.extract(7)
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(
            self.extract(0) + self.extract(2) + self.extract(4) + self.extract(6),
            self.extract(1) + self.extract(3) + self.extract(5) + self.extract(7),
        )
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    #[inline]
    fn swap_iq(self) -> Self {
        unsafe { mem::transmute(_mm256_permute_ps(mem::transmute(self), SWAP_IQ_PS)) }
    }
}

impl Simd<f64> for f64x4 {
    type Array = [f64; 4];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 4];
        self.write_to_slice_unaligned(&mut target);
        target
    }

    type ComplexArray = [Complex<f64>; 2];

    const LEN: usize = 4;

    #[inline]
    fn from_complex(value: Complex<f64>) -> f64x4 {
        f64x4::new(value.re, value.im, value.re, value.im)
    }

    #[inline]
    fn add_real(self, value: f64) -> f64x4 {
        let increment = f64x4::splat(value);
        self + increment
    }

    #[inline]
    fn add_complex(self, value: Complex<f64>) -> f64x4 {
        let increment = f64x4::new(value.re, value.im, value.re, value.im);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f64) -> f64x4 {
        let scale_vector = f64x4::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f64>) -> f64x4 {
        let scaling_real = f64x4::splat(value.re);
        let scaling_imag = f64x4::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm256_addsub_pd(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn mul_complex(self, value: f64x4) -> f64x4 {
        let scaling_real = f64x4::new(
            value.extract(0),
            value.extract(0),
            value.extract(2),
            value.extract(2),
        );
        let scaling_imag = f64x4::new(
            value.extract(1),
            value.extract(1),
            value.extract(3),
            value.extract(3),
        );
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm256_addsub_pd(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn div_complex(self, value: f64x4) -> f64x4 {
        let scaling_imag = f64x4::new(
            self.extract(0),
            self.extract(0),
            self.extract(2),
            self.extract(2),
        );
        let scaling_real = f64x4::new(
            self.extract(1),
            self.extract(1),
            self.extract(3),
            self.extract(3),
        );
        let parallel = scaling_real * value;
        let shuffled = value.swap_iq();
        let cross = scaling_imag * shuffled;
        let mul: f64x4 = unsafe {
            mem::transmute(_mm256_addsub_pd(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        };
        let square = shuffled * shuffled;
        let square_shuffled = square.swap_iq();
        let sum = square + square_shuffled;
        let div = mul / sum;
        div.swap_iq()
    }

    #[inline]
    fn complex_abs_squared(self) -> f64x4 {
        let squared = self * self;
        unsafe {
            mem::transmute(_mm256_hadd_pd(
                mem::transmute(squared),
                mem::transmute(squared),
            ))
        }
    }

    #[inline]
    fn complex_abs(self) -> f64x4 {
        let squared_sum = self.complex_abs_squared();
        squared_sum.sqrt()
    }

    #[inline]
    fn sqrt(self) -> f64x4 {
        self.sqrt()
    }

    #[inline]
    fn store_half(self, target: &mut [f64], index: usize) {
        target[index] = self.extract(0);
        target[index + 1] = self.extract(1);
    }

    #[inline]
    fn sum_real(&self) -> f64 {
        self.extract(0) + self.extract(1) + self.extract(2) + self.extract(3)
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(
            self.extract(0) + self.extract(2),
            self.extract(1) + self.extract(3),
        )
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    #[inline]
    fn swap_iq(self) -> Self {
        unsafe { mem::transmute(_mm256_permute_pd(mem::transmute(self), SWAP_IQ_PD)) }
    }
}

impl SimdFrom<f32x8> for i32x8 {
    fn regfrom(value: f32x8) -> Self {
        Self::from_cast(value)
    }
}

impl SimdFrom<i32x8> for f32x8 {
    fn regfrom(value: i32x8) -> Self {
        Self::from_cast(value)
    }
}

impl SimdFrom<f64x4> for i64x4 {
    fn regfrom(value: f64x4) -> Self {
        Self::from_cast(value)
    }
}

impl SimdFrom<i64x4> for f64x4 {
    fn regfrom(value: i64x4) -> Self {
        Self::from_cast(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_test_f32() {
        let vec = f32x8::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let result = vec.swap_iq();
        assert_eq!(result.extract(0), vec.extract(1));
        assert_eq!(result.extract(1), vec.extract(0));
        assert_eq!(result.extract(2), vec.extract(3));
        assert_eq!(result.extract(3), vec.extract(2));
        assert_eq!(result.extract(4), vec.extract(5));
        assert_eq!(result.extract(5), vec.extract(4));
        assert_eq!(result.extract(6), vec.extract(7));
        assert_eq!(result.extract(7), vec.extract(6));
    }

    #[test]
    fn shuffle_test_f64() {
        let vec = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let result = vec.swap_iq();
        assert_eq!(result.extract(0), vec.extract(1));
        assert_eq!(result.extract(1), vec.extract(0));
        assert_eq!(result.extract(2), vec.extract(3));
        assert_eq!(result.extract(3), vec.extract(2));
    }
}
