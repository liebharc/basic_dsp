use stdsimd::simd::*;
use stdsimd::vendor::*;
use numbers::*;
use super::{Simd, SimdFrom};

pub type Reg32 = f32x4;

pub type Reg64 = f64x2;

pub type IntReg32 = i32x4;

pub type IntReg64 = i64x2;

pub type UIntReg32 = u32x4;

pub type UIntReg64 = u64x2;

/// This value must be read in groups of 2 bits:
/// 10 means that the third position (since it's the third bit pair)
/// will be replaced with the value of the second position (10b = 2d)
const SHUFFLE_PS: i32 = 0b10110001;

impl Simd<f32> for f32x4 {
    type Array = [f32; 4];

    #[inline]
    fn to_array(self) -> Self::Array {
        let mut target = [0.0; 4];
        self.store(&mut target, 0);
        target
    }

    type ComplexArray = [Complex<f32>; 2];

    #[inline]
    fn len() -> usize {
        4
    }

    #[inline]
    fn load_wrap_unchecked(array: &[f32], idx: usize) -> f32x4 {
        let mut temp = [0.0; 4];
        for (i, t) in temp.iter_mut().enumerate() {
            *t = unsafe { *array.get_unchecked((idx + i) % array.len()) };
        }
        f32x4::load(&temp, 0)
    }

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
        let shuffled = unsafe { _mm_permute_ps(self, SHUFFLE_PS) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm_addsub_ps(parallel, cross) }
    }

    #[inline]
    fn mul_complex(self, value: f32x4) -> f32x4 {
        let scaling_real = f32x4::new(value.extract(0),
                                      value.extract(0),
                                      value.extract(2),
                                      value.extract(2));

        let scaling_imag = f32x4::new(value.extract(1),
                                      value.extract(1),
                                      value.extract(3),
                                      value.extract(3));

        let parallel = scaling_real * self;
        let shuffled = unsafe { _mm_permute_ps(self, SHUFFLE_PS) };
        let cross = scaling_imag * shuffled;
        unsafe { _mm_addsub_ps(parallel, cross) }
    }

    #[inline]
    fn div_complex(self, value: f32x4) -> f32x4 {
        let scaling_imag = f32x4::new(self.extract(0),
                                      self.extract(0),
                                      self.extract(2),
                                      self.extract(2));

        let scaling_real = f32x4::new(self.extract(1),
                                      self.extract(1),
                                      self.extract(3),
                                      self.extract(3));
                                                      
        let parallel = scaling_real * value;
        let shuffled = unsafe { _mm_permute_ps(value, SHUFFLE_PS) };
        let cross = scaling_imag * shuffled;
        let mul = unsafe { _mm_addsub_ps(parallel, cross) };
        let square = shuffled * shuffled;
        let square_shuffled = unsafe { _mm_permute_ps(square, SHUFFLE_PS) };
        let sum = square + square_shuffled;
        let div = mul / sum;
        unsafe { _mm_permute_ps(div, SHUFFLE_PS) }
    }

    #[inline]
    fn complex_abs_squared(self) -> f32x4 {
        let squared = self * self;

        unsafe { _mm_hadd_ps(squared, squared) }
    }

    #[inline]
    fn complex_abs(self) -> f32x4 {
        let squared = self * self;

        let squared_sum = unsafe { _mm_hadd_ps(squared, squared) };
        unsafe { _mm_sqrt_ps(squared_sum) }
    }

    #[inline]
    fn complex_abs_squared2(self) -> f32x4 {

        let abs = self.complex_abs_squared();
        f32x4::new(abs.extract(0),
                   abs.extract(2),
                   abs.extract(1),
                   abs.extract(3))
    }

    #[inline]
    fn complex_abs2(self) -> f32x4 {

        let abs = self.complex_abs();
        f32x4::new(abs.extract(0),
                   abs.extract(2),
                   abs.extract(1),
                   abs.extract(3))
    }

    #[inline]
    fn sqrt(self) -> f32x4 {
        unsafe { _mm_sqrt_ps(self) }
    }

    #[inline]
    fn store_half_unchecked(self, target: &mut [f32], index: usize) {
        unsafe {
            *target.get_unchecked_mut(index) = self.extract(0);
            *target.get_unchecked_mut(index + 1) = self.extract(1);                             
        }
    }

    #[inline]
    fn sum_real(&self) -> f32 {
        self.extract(0) + self.extract(1) + self.extract(2) + self.extract(3)
                                                                             
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(self.extract(0) + self.extract(2),
                            self.extract(1) + self.extract(3))
    }
    
    #[inline]
    fn max(self, other: Self) -> Self {
        unsafe { _mm_max_ps(self, other) }
    }
    
    #[inline]
    fn min(self, other: Self) -> Self {
        unsafe { _mm_min_ps(self, other) }
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

    #[inline]
    fn len() -> usize {
        2
    }

    #[inline]
    fn load_wrap_unchecked(array: &[f64], idx: usize) -> f64x2 {
        let mut temp = [0.0; 2];
        for (i, t) in temp.iter_mut().enumerate() {
            *t = unsafe { *array.get_unchecked((idx + i) % array.len()) };
        }

        f64x2::load(&temp, 0)
    }

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
    fn complex_abs_squared2(self) -> f64x2 {
        self.complex_abs_squared()
    }

    #[inline]
    fn complex_abs2(self) -> f64x2 {
        self.complex_abs()
    }

    #[inline]
    fn sqrt(self) -> f64x2 {
        unsafe { _mm_sqrt_pd(self) }
    }

    #[inline]
    fn store_half_unchecked(self, target: &mut [f64], index: usize) {
        unsafe {
            *target.get_unchecked_mut(index) = self.extract(0);
        }
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
        unsafe { _mm_max_pd(self, other) }
    }
    
    #[inline]
    fn min(self, other: Self) -> Self {
        unsafe { _mm_min_pd(self, other) }
    }
}

impl SimdFrom<f32x4> for i32x4 {
    fn regfrom(value: f32x4) -> Self {
        value.as_i32x4()
    }
}

impl SimdFrom<i32x4> for f32x4 {
    fn regfrom(value: i32x4) -> Self {
        value.as_f32x4()
    }
}

impl SimdFrom<f64x2> for i64x2 {
    fn regfrom(value: f64x2) -> Self {
        value.as_i64x2()
    }
}

impl SimdFrom<i64x2> for f64x2 {
    fn regfrom(value: i64x2) -> Self {
        value.as_f64x2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_test() {
        let vec = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let result = unsafe { _mm_permute_ps(vec, SHUFFLE_PS) };
        let expected = 
            f32x4::new(vec.extract(1),
                vec.extract(0),
                vec.extract(3),
                vec.extract(2));
        assert_eq!(result, expected);
    }
}