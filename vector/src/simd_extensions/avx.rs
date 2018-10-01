use numbers::*;
use super::{Simd, SimdFrom};
use simd::simd::*;
use simd::vendor::*;

pub type Reg32 = f32x8;

pub type Reg64 = f64x4;

pub type IntReg32 = i32x8;

pub type IntReg64 = i64x4;

pub type UIntReg32 = u32x8;

pub type UIntReg64 = u64x4;

/// This value must be read in groups of 3 bits.
const SWAP_IQ_PS: i32 = 0b10110001;

/// This value must be read in groups of 2 bits:
/// 10 means that the third position (since it's the third bit pair)
/// will be replaced with the value of the second position (10b = 2d)
const SWAP_IQ_PD: i32 = 0b10110001;

impl Simd<f32> for f32x8 {
    type Array = [f32; 8];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 8];
        self.store(&mut target, 0);
        target
    }

    type ComplexArray = [Complex<f32>; 4];
	
	const LEN: usize = 8;

    #[inline]
    fn load_wrap_unchecked(array: &[f32], idx: usize) -> f32x8 {
        let mut temp = [0.0; 8];
        for (i, t) in temp.iter_mut().enumerate() {
            *t = unsafe { *array.get_unchecked((idx + i) % array.len()) };
        }
        f32x8::load(&temp, 0)
    }

    #[inline]
    fn from_complex(value: Complex<f32>) -> f32x8 {
        f32x8::new(value.re,
                   value.im,
                   value.re,
                   value.im,
                   value.re,
                   value.im,
                   value.re,
                   value.im)
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
        let shuffled = unsafe { _mm256_permute_ps(self, SWAP_IQ_PS) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm256_addsub_ps(parallel, cross) }
    }

    #[inline]
    fn mul_complex(self, value: f32x8) -> f32x8 {
        let scaling_real = f32x8::new(value.extract(0),
                                      value.extract(0),
                                      value.extract(2),
                                      value.extract(2),
                                      value.extract(4),
                                      value.extract(4),
                                      value.extract(6),
                                      value.extract(6));
        let scaling_imag = f32x8::new(value.extract(1),
                                      value.extract(1),
                                      value.extract(3),
                                      value.extract(3),
                                      value.extract(5),
                                      value.extract(5),
                                      value.extract(7),
                                      value.extract(7));
        let parallel = scaling_real * self;
        let shuffled = unsafe { _mm256_permute_ps(self, SWAP_IQ_PS) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm256_addsub_ps(parallel, cross) }
    }

    #[inline]
    fn div_complex(self, value: f32x8) -> f32x8 {
        let scaling_imag = f32x8::new(self.extract(0),
                                      self.extract(0),
                                      self.extract(2),
                                      self.extract(2),
                                      self.extract(4),
                                      self.extract(4),
                                      self.extract(6),
                                      self.extract(6));
        let scaling_real = f32x8::new(self.extract(1),
                                      self.extract(1),
                                      self.extract(3),
                                      self.extract(3),
                                      self.extract(5),
                                      self.extract(5),
                                      self.extract(7),
                                      self.extract(7));
        let parallel = scaling_real * value;
        let shuffled = unsafe { _mm256_permute_ps(value, SWAP_IQ_PS) };
        let cross = scaling_imag * shuffled;
        let mul = unsafe { _mm256_addsub_ps(parallel, cross) };
        let square = shuffled * shuffled;
        let square_shuffled = unsafe { _mm256_permute_ps(square, SWAP_IQ_PS) };
        let sum = square + square_shuffled;
        let div = mul / sum;
        unsafe { _mm256_permute_ps(div, SWAP_IQ_PS) }
    }

    #[inline]
    fn complex_abs_squared(self) -> f32x8 {
        let squared = self * self;
        unsafe { _mm256_hadd_ps(squared, squared) }
    }

    #[inline]
    fn complex_abs(self) -> f32x8 {
        let squared_sum = self.complex_abs_squared();
        unsafe { _mm256_sqrt_ps(squared_sum) }
    }

    #[inline]
    fn complex_abs_squared2(self) -> f32x8 {
        self.complex_abs_squared()
    }

    #[inline]
    fn complex_abs2(self) -> f32x8 {
        self.complex_abs()
    }

    #[inline]
    fn sqrt(self) -> f32x8 {
        unsafe { _mm256_sqrt_ps(self) }
    }

    #[inline]
    fn store_half_unchecked(self, target: &mut [f32], index: usize) {
        unsafe {
            *target.get_unchecked_mut(index) = self.extract(0);
            *target.get_unchecked_mut(index + 1) = self.extract(1);
            *target.get_unchecked_mut(index + 2) = self.extract(2);
            *target.get_unchecked_mut(index + 3) = self.extract(3);
        }
    }

    #[inline]
    fn sum_real(&self) -> f32 {
        self.extract(0) + self.extract(1) + self.extract(2) + self.extract(3) +
        self.extract(4) + self.extract(5) + self.extract(6) + self.extract(7)
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(self.extract(0) + self.extract(2) + self.extract(4) + self.extract(6),
                            self.extract(1) + self.extract(3) + self.extract(5) + self.extract(7))
    }
    
    #[inline]
    fn max(self, other: Self) -> Self {
        unsafe { _mm256_max_ps(self, other) }
    }
    
    #[inline]
    fn min(self, other: Self) -> Self {
        unsafe { _mm256_min_ps(self, other) }
    }
}

impl Simd<f64> for f64x4 {
    type Array = [f64; 4];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 4];
        self.store(&mut target, 0);
        target
    }

    type ComplexArray = [Complex<f64>; 2];
	
	const LEN: usize = 4;

    #[inline]
    fn load_wrap_unchecked(array: &[f64], idx: usize) -> f64x4 {
        let mut temp = [0.0; 4];
        for (i, t) in temp.iter_mut().enumerate() {
            *t = unsafe { *array.get_unchecked((idx + i) % array.len()) };
        }
        f64x4::load(&temp, 0)
    }

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
        let shuffled = unsafe { _mm256_permute_pd(self, SWAP_IQ_PD) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm256_addsub_pd(parallel, cross) }
    }

    #[inline]
    fn mul_complex(self, value: f64x4) -> f64x4 {
        let scaling_real = f64x4::new(value.extract(0),
                                      value.extract(0),
                                      value.extract(2),
                                      value.extract(2));
        let scaling_imag = f64x4::new(value.extract(1),
                                      value.extract(1),
                                      value.extract(3),
                                      value.extract(3));
        let parallel = scaling_real * self;
        let shuffled = unsafe { _mm256_permute_pd(self, SWAP_IQ_PD) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm256_addsub_pd(parallel, cross) }
    }

    #[inline]
    fn div_complex(self, value: f64x4) -> f64x4 {
        let scaling_imag = f64x4::new(self.extract(0),
                                      self.extract(0),
                                      self.extract(2),
                                      self.extract(2));
        let scaling_real = f64x4::new(self.extract(1),
                                      self.extract(1),
                                      self.extract(3),
                                      self.extract(3));
        let parallel = scaling_real * value;
        let shuffled = unsafe { _mm256_permute_pd(value, SWAP_IQ_PD) };
        let cross = scaling_imag * shuffled;
        let mul = unsafe { _mm256_addsub_pd(parallel, cross) };
        let square = shuffled * shuffled;
        let square_shuffled = unsafe { _mm256_permute_pd(square, SWAP_IQ_PD) };
        let sum = square + square_shuffled;
        let div = mul / sum;
        unsafe { _mm256_permute_pd(div, SWAP_IQ_PD) }
    }

    #[inline]
    fn complex_abs_squared(self) -> f64x4 {
        let squared = self * self;
        unsafe { _mm256_hadd_pd(squared, squared) }
    }

    #[inline]
    fn complex_abs(self) -> f64x4 {
        let squared_sum = self.complex_abs_squared();
        unsafe { _mm256_sqrt_pd(squared_sum) }
    }

    #[inline]
    fn complex_abs_squared2(self) -> f64x4 {
        self.complex_abs_squared()
    }

    #[inline]
    fn complex_abs2(self) -> f64x4 {
        self.complex_abs()
    }

    #[inline]
    fn sqrt(self) -> f64x4 {
        unsafe { _mm256_sqrt_pd(self) }
    }

    #[inline]
    fn store_half_unchecked(self, target: &mut [f64], index: usize) {
        let mut temp = [0.0; 4];
        self.store(&mut temp, 0);
        unsafe {
            *target.get_unchecked_mut(index) = temp[0];
            *target.get_unchecked_mut(index + 1) = temp[2];
        }
    }

    #[inline]
    fn sum_real(&self) -> f64 {
        self.extract(0) + self.extract(1) + self.extract(2) + self.extract(3)
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(self.extract(0) + self.extract(2),
                            self.extract(1) + self.extract(3))
    }
    
    #[inline]
    fn max(self, other: Self) -> Self {
        unsafe { _mm256_max_pd(self, other) }
    }
    
    #[inline]
    fn min(self, other: Self) -> Self {
        unsafe { _mm256_min_pd(self, other) }
    }
}

impl SimdFrom<f32x8> for i32x8 {
    fn regfrom(value: f32x8) -> Self {
        value.as_i32x8()
    }
}

impl SimdFrom<i32x8> for f32x8 {
    fn regfrom(value: i32x8) -> Self {
        value.as_f32x8()
    }
}

impl SimdFrom<f64x4> for i64x4 {
    fn regfrom(value: f64x4) -> Self {
        value.as_i64x4()
    }
}

impl SimdFrom<i64x4> for f64x4 {
    fn regfrom(value: i64x4) -> Self {
        value.as_f64x4()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_test() {
        let vec = f32x8::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let result = unsafe { _mm256_permute_ps(vec, SWAP_IQ_PS) };
        let expected = 
            f32x8::new(vec.extract(1),
                 vec.extract(0),
                 vec.extract(3),
                 vec.extract(2),
                 vec.extract(5),
                 vec.extract(4),
                 vec.extract(7),
                 vec.extract(6));
        assert_eq!(result, expected);
    }
}