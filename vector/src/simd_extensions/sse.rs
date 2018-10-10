use super::{Simd, SimdFrom};
use numbers::*;
use simd::x86::sse2::*;
use simd::*;
use simd;
use std::arch::x86_64::*;
use std::mem;

/// This value must be read in groups of 2 bits:
/// 10 means that the third position (since it's the third bit pair)
/// will be replaced with the value of the second position (10b = 2d)
const SWAP_IQ_PS: i32 = 0b1011_0001;

impl Simd<f32> for f32x4 {
    type Array = [f32; 4];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 4];
        self.store(&mut target, 0);
        target
    }

    type ComplexArray = [Complex<f32>; 2];

    const LEN: usize = 4;

    #[inline]
    fn from_complex(value: Complex<f32>) -> f32x4 {
        f32x4::new(value.re, value.im, value.re, value.im)
    }

    #[inline]
    fn add_real(self, value: f32) -> f32x4 {
        let increment = f32x4::splat(value);
        self + increment
    }

    #[inline]
    fn add_complex(self, value: Complex<f32>) -> f32x4 {
        let increment = f32x4::new(value.re, value.im, value.re, value.im);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f32) -> f32x4 {
        let scale_vector = f32x4::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f32>) -> f32x4 {
        let scaling_real = f32x4::splat(value.re);
        let scaling_imag = f32x4::splat(value.im);
        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn mul_complex(self, value: f32x4) -> f32x4 {
        let scaling_real = f32x4::new(
            value.extract(0),
            value.extract(0),
            value.extract(2),
            value.extract(2),
        );

        let scaling_imag = f32x4::new(
            value.extract(1),
            value.extract(1),
            value.extract(3),
            value.extract(3),
        );

        let parallel = scaling_real * self;
        let shuffled = self.swap_iq();
        let cross = scaling_imag * shuffled;
        unsafe {
            mem::transmute(_mm_addsub_ps(
                mem::transmute(parallel),
                mem::transmute(cross),
            ))
        }
    }

    #[inline]
    fn div_complex(self, value: f32x4) -> f32x4 {
        let scaling_imag = f32x4::new(
            self.extract(0),
            self.extract(0),
            self.extract(2),
            self.extract(2),
        );

        let scaling_real = f32x4::new(
            self.extract(1),
            self.extract(1),
            self.extract(3),
            self.extract(3),
        );

        let parallel = scaling_real * value;
        let shuffled = value.swap_iq();
        let cross = scaling_imag * shuffled;
        let mul: f32x4 = unsafe {
            mem::transmute(_mm_addsub_ps(
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
    fn complex_abs_squared(self) -> f32x4 {
        let squared = self * self;
        unsafe {
            mem::transmute(_mm_hadd_ps(
                mem::transmute(squared),
                mem::transmute(squared),
            ))
        }
    }

    #[inline]
    fn complex_abs(self) -> f32x4 {
        let squared_sum = self.complex_abs_squared();
        squared_sum.sqrt()
    }

    #[inline]
    fn sqrt(self) -> f32x4 {
        self.sqrt()
    }

    #[inline]
    fn store_half(self, target: &mut [f32], index: usize) {
        target[index] = self.extract(0);
        target[index + 1] = self.extract(1);
    }

    #[inline]
    fn sum_real(&self) -> f32 {
        self.extract(0) + self.extract(1) + self.extract(2) + self.extract(3)
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(
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
        unsafe { mem::transmute(_mm_permute_ps(mem::transmute(self), SWAP_IQ_PS)) }
    }
}

impl Simd<f64> for f64x2 {
    type Array = [f64; 2];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 2];
        self.store(&mut target, 0);
        target
    }

    type ComplexArray = [Complex<f64>; 1];

    const LEN: usize = 2;

    #[inline]
    fn from_complex(value: Complex<f64>) -> f64x2 {
        f64x2::new(value.re, value.im)
    }

    #[inline]
    fn add_real(self, value: f64) -> f64x2 {
        let increment = f64x2::splat(value);
        self + increment
    }

    #[inline]
    fn add_complex(self, value: Complex<f64>) -> f64x2 {
        let increment = f64x2::new(value.re, value.im);
        self + increment
    }

    #[inline]
    fn scale_real(self, value: f64) -> f64x2 {
        let scale_vector = f64x2::splat(value);
        self * scale_vector
    }

    #[inline]
    fn scale_complex(self, value: Complex<f64>) -> f64x2 {
        let complex = Complex::new(self.extract(0), self.extract(1));
        let result = complex * value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn mul_complex(self, value: f64x2) -> f64x2 {
        let complex = Complex::new(self.extract(0), self.extract(1));
        let value = Complex::new(value.extract(0), value.extract(1));
        let result = complex * value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn div_complex(self, value: f64x2) -> f64x2 {
        let complex = Complex::new(self.extract(0), self.extract(1));
        let value = Complex::new(value.extract(0), value.extract(1));
        let result = complex / value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn complex_abs_squared(self) -> f64x2 {
        let a = self.extract(0);
        let b = self.extract(1);
        let result = a * a + b * b;
        f64x2::new(result, 0.0)
    }

    #[inline]
    fn complex_abs(self) -> f64x2 {
        let a = self.extract(0);
        let b = self.extract(1);
        let result = (a * a + b * b).sqrt();
        f64x2::new(result, 0.0)
    }

    #[inline]
    fn sqrt(self) -> f64x2 {
        simd::x86::sse2::Sse2F64x2::sqrt(self)
    }

    #[inline]
    fn store_half(self, target: &mut [f64], index: usize) {
        target[index] = self.extract(0);
    }

    #[inline]
    fn sum_real(&self) -> f64 {
        self.extract(0) + self.extract(1)
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(self.extract(0), self.extract(1))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        simd::x86::sse2::Sse2F64x2::max(self, other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        simd::x86::sse2::Sse2F64x2::min(self, other)
    }

    #[inline]
    fn swap_iq(self) -> Self {
        f64x2::new(self.extract(1), self.extract(0))
    }
}

impl SimdFrom<f32x4> for i32x4 {
    fn regfrom(value: f32x4) -> Self {
        value.to_i32()
    }
}

impl SimdFrom<i32x4> for f32x4 {
    fn regfrom(value: i32x4) -> Self {
        value.to_f32()
    }
}

impl SimdFrom<f64x2> for i64x2 {
    fn regfrom(value: f64x2) -> Self {
        value.to_i64()
    }
}

impl SimdFrom<i64x2> for f64x2 {
    fn regfrom(value: i64x2) -> Self {
        value.to_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_test() {
        let vec = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let result = vec.swap_iq();
        assert_eq!(result.extract(0), vec.extract(1));
        assert_eq!(result.extract(1), vec.extract(0));
        assert_eq!(result.extract(2), vec.extract(3));
        assert_eq!(result.extract(3), vec.extract(2));
    }
}
