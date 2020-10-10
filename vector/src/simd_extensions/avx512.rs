use super::{Simd, SimdFrom};
use crate::numbers::*;
pub use packed_simd_2::{f32x16, f64x8};
use packed_simd_2::{i32x16, i64x8, FromCast};
use std::arch::x86_64::*;
use std::mem;

/// This value must be read in groups of 2 bits.
const SWAP_IQ_PS: i32 = 0b1011_0001;

/// This value must be read in groups of 2 bits:
/// 10 means that the third position (since it's the third bit pair)
/// will be replaced with the value of the second position (10b = 2d)
const SWAP_IQ_PD: i32 = 0b1011_0001;

impl Simd<f32> for f32x16 {
    type Array = [f32; 16];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 16];
        self.store(&mut target, 0);
        target
    }

    type ComplexArray = [Complex<f32>; 8];

    const LEN: usize = 16;

    #[inline]
    fn from_complex(value: Complex<f32>) -> f32x16 {
        f32x16::new(
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
        )
    }

    #[inline]
    fn add_real(self, value: f32) -> f32x16 {
        let increment = f32x16::splat(value);
        self + increment
    }

    #[inline]
    fn add_complex(self, value: Complex<f32>) -> f32x16 {
        let increment = f32x16::from_complex(value);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f32) -> f32x16 {
        let scale_vector = f32x16::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f32>) -> f32x16 {
        let scaling_real = f32x16::splat(value.re);
        let scaling_imag = f32x16::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = unsafe { _mm512_permute_ps(self, SWAP_IQ_PS) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm512_addsub_ps(parallel, cross) }
    }

    #[inline]
    fn mul_complex(self, value: f32x16) -> f32x16 {
        let scaling_real = f32x16::new(
            value.extract(0),
            value.extract(0),
            value.extract(2),
            value.extract(2),
            value.extract(4),
            value.extract(4),
            value.extract(6),
            value.extract(6),
            value.extract(8),
            value.extract(8),
            value.extract(10),
            value.extract(10),
            value.extract(12),
            value.extract(12),
            value.extract(14),
            value.extract(14),
        );
        let scaling_imag = f32x16::new(
            value.extract(1),
            value.extract(1),
            value.extract(3),
            value.extract(3),
            value.extract(5),
            value.extract(5),
            value.extract(7),
            value.extract(7),
            value.extract(9),
            value.extract(9),
            value.extract(11),
            value.extract(11),
            value.extract(13),
            value.extract(13),
            value.extract(15),
            value.extract(15),
        );
        let parallel = scaling_real * self;
        let shuffled = unsafe { _mm512_permute_ps(self, SWAP_IQ_PS) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm512_addsub_ps(parallel, cross) }
    }

    #[inline]
    fn div_complex(self, value: f32x16) -> f32x16 {
        let scaling_real = f32x16::new(
            value.extract(0),
            value.extract(0),
            value.extract(2),
            value.extract(2),
            value.extract(4),
            value.extract(4),
            value.extract(6),
            value.extract(6),
            value.extract(8),
            value.extract(8),
            value.extract(10),
            value.extract(10),
            value.extract(12),
            value.extract(12),
            value.extract(14),
            value.extract(14),
        );
        let scaling_imag = f32x16::new(
            value.extract(1),
            value.extract(1),
            value.extract(3),
            value.extract(3),
            value.extract(5),
            value.extract(5),
            value.extract(7),
            value.extract(7),
            value.extract(9),
            value.extract(9),
            value.extract(11),
            value.extract(11),
            value.extract(13),
            value.extract(13),
            value.extract(15),
            value.extract(15),
        );
        let parallel = scaling_real * value;
        let shuffled = unsafe { _mm512_permute_ps(value, SWAP_IQ_PS) };
        let cross = scaling_imag * shuffled;
        let mul = unsafe { _mm512_addsub_ps(parallel, cross) };
        let square = shuffled * shuffled;
        let square_shuffled = unsafe { _mm512_permute_ps(square, SWAP_IQ_PS) };
        let sum = square + square_shuffled;
        let div = mul / sum;
        unsafe { _mm512_permute_ps(div, SWAP_IQ_PS) }
    }

    #[inline]
    fn complex_abs_squared(self) -> f32x16 {
        let squared = self * self;
        unsafe { _mm512_hadd_ps(squared, squared) }
    }

    #[inline]
    fn complex_abs(self) -> f32x16 {
        let squared_sum = self.complex_abs_squared();
        unsafe { _mm512_sqrt_ps(squared_sum) }
    }

    #[inline]
    fn sqrt(self) -> f32x16 {
        unsafe { _mm512_sqrt_ps(self) }
    }

    #[inline]
    fn store_half(self, target: &mut [f32], index: usize) {
        target[index] = self.extract(0);
        target[index + 1] = self.extract(1);
        target[index + 2] = self.extract(2);
        target[index + 3] = self.extract(3);
        target[index + 4] = self.extract(4);
        target[index + 5] = self.extract(5);
        target[index + 6] = self.extract(6);
        target[index + 7] = self.extract(7);
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
            + self.extract(8)
            + self.extract(9)
            + self.extract(10)
            + self.extract(11)
            + self.extract(12)
            + self.extract(13)
            + self.extract(14)
            + self.extract(15)
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(
            self.extract(0)
                + self.extract(2)
                + self.extract(4)
                + self.extract(6)
                + self.extract(8)
                + self.extract(10)
                + self.extract(12)
                + self.extract(14),
            self.extract(1)
                + self.extract(3)
                + self.extract(5)
                + self.extract(7)
                + self.extract(9)
                + self.extract(11)
                + self.extract(13)
                + self.extract(15),
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
}

impl Simd<f64> for f64x8 {
    type Array = [f64; 8];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 8];
        self.store(&mut target, 0);
        target
    }

    type ComplexArray = [Complex<f64>; 4];

    const LEN: usize = 8;

    #[inline]
    fn from_complex(value: Complex<f64>) -> f64x8 {
        f64x8::new(value.re, value.im, value.re, value.im)
    }

    #[inline]
    fn add_real(self, value: f64) -> f64x8 {
        let increment = f64x8::splat(value);
        self + increment
    }

    #[inline]
    fn add_complex(self, value: Complex<f64>) -> f64x8 {
        let increment = f64x8::new(
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
        );
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f64) -> f64x8 {
        let scale_vector = f64x8::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f64>) -> f64x8 {
        let scaling_real = f64x8::splat(value.re);
        let scaling_imag = f64x8::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = unsafe { _mm512_permute_pd(self, SWAP_IQ_PD) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm512_addsub_pd(parallel, cross) }
    }

    #[inline]
    fn mul_complex(self, value: f64x8) -> f64x8 {
        let scaling_real = f64x8::new(
            value.extract(0),
            value.extract(0),
            value.extract(2),
            value.extract(2),
            value.extract(4),
            value.extract(4),
            value.extract(6),
            value.extract(6),
        );
        let scaling_imag = f64x8::new(
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
        let shuffled = unsafe { _mm512_permute_pd(self, SWAP_IQ_PD) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm512_addsub_pd(parallel, cross) }
    }

    #[inline]
    fn div_complex(self, value: f64x8) -> f64x8 {
        let scaling_imag = f64x8::new(
            self.extract(0),
            self.extract(0),
            self.extract(2),
            self.extract(2),
            self.extract(4),
            self.extract(4),
            self.extract(6),
            self.extract(6),
        );
        let scaling_real = f64x8::new(
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
        let shuffled = unsafe { _mm512_permute_pd(value, SWAP_IQ_PD) };
        let cross = scaling_imag * shuffled;
        let mul = unsafe { _mm512_addsub_pd(parallel, cross) };
        let square = shuffled * shuffled;
        let square_shuffled = unsafe { _mm512_permute_pd(square, SWAP_IQ_PD) };
        let sum = square + square_shuffled;
        let div = mul / sum;
        unsafe { _mm512_permute_pd(div, SWAP_IQ_PD) }
    }

    #[inline]
    fn complex_abs_squared(self) -> f64x8 {
        let squared = self * self;
        unsafe { _mm512_hadd_pd(squared, squared) }
    }

    #[inline]
    fn complex_abs(self) -> f64x8 {
        let squared_sum = self.complex_abs_squared();
        unsafe { _mm512_sqrt_pd(squared_sum) }
    }

    #[inline]
    fn sqrt(self) -> f64x8 {
        unsafe { _mm512_sqrt_pd(self) }
    }

    #[inline]
    fn store_half(self, target: &mut [f64], index: usize) {
        target[index] = self.extract(0);
        target[index + 1] = self.extract(1);
        target[index + 2] = self.extract(2);
        target[index + 3] = self.extract(3);
    }

    #[inline]
    fn sum_real(&self) -> f64 {
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
    fn sum_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(
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
}

impl SimdFrom<f32x16> for i32x16 {
    fn regfrom(value: f32x16) -> Self {
        Self::from_cast(value)
    }
}

impl SimdFrom<i32x16> for f32x16 {
    fn regfrom(value: f32x16) -> Self {
        Self::from_cast(value)
    }
}

impl SimdFrom<f64x8> for i64x8 {
    fn regfrom(value: f64x8) -> Self {
        Self::from_cast(value)
    }
}

impl SimdFrom<i64x8> for f64x8 {
    fn regfrom(value: f64x8) -> Self {
        Self::from_cast(value)
    }
}
