use numbers::*;
use super::Simd;
use std::ops::*;

#[allow(non_camel_case_types)]
// To stay consistent with the `simd` crate
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct f32x4(f32, f32, f32, f32);

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct f64x2(f64, f64);

impl f32x4 {
    #[inline]
    pub fn new(v1: f32, v2: f32, v3: f32, v4: f32) -> Self {
        f32x4(v1, v2, v3, v4)
    }

    #[inline]
    pub fn splat(value: f32) -> Self {
        f32x4(value, value, value, value)
    }

    #[inline]
    pub fn load(array: &[f32], index: usize) -> Self {
        let array = &array[index..index + 4];
        unsafe {
            f32x4(*array.get_unchecked(0),
                  *array.get_unchecked(1),
                  *array.get_unchecked(2),
                  *array.get_unchecked(3))
        }
    }

    #[inline]
    pub fn store(self, array: &mut [f32], index: usize) {
        let array = &mut array[index..index + 4];
        unsafe {
            *array.get_unchecked_mut(0) = self.0;
            *array.get_unchecked_mut(1) = self.1;
            *array.get_unchecked_mut(2) = self.2;
            *array.get_unchecked_mut(3) = self.3;
        }
    }

    #[inline]
    pub fn extract(self, index: u32) -> f32 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            _ => panic!("{} out of bounds for type f32x4", index),
        }
    }
}

impl f64x2 {
    #[inline]
    pub fn new(v1: f64, v2: f64) -> Self {
        f64x2(v1, v2)
    }

    #[inline]
    pub fn splat(value: f64) -> Self {
        f64x2(value, value)
    }

    #[inline]
    pub fn load(array: &[f64], index: usize) -> Self {
        let array = &array[index..index + 2];
        unsafe { f64x2(*array.get_unchecked(0), *array.get_unchecked(1)) }
    }

    #[inline]
    pub fn store(self, array: &mut [f64], index: usize) {
        let array = &mut array[index..index + 2];
        unsafe {
            *array.get_unchecked_mut(0) = self.0;
            *array.get_unchecked_mut(1) = self.1;
        }
    }

    #[inline]
    pub fn extract(self, index: u32) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            _ => panic!("{} out of bounds for type f32x4", index),
        }
    }
}

impl Add for f32x4 {
    type Output = Self;
    #[inline]
    fn add(self, x: Self) -> Self {
        f32x4(self.0 + x.0, self.1 + x.1, self.2 + x.2, self.3 + x.3)
    }
}

impl Sub for f32x4 {
    type Output = Self;
    #[inline]
    fn sub(self, x: Self) -> Self {
        f32x4(self.0 - x.0, self.1 - x.1, self.2 - x.2, self.3 - x.3)
    }
}

impl Mul for f32x4 {
    type Output = Self;
    #[inline]
    fn mul(self, x: Self) -> Self {
        f32x4(self.0 * x.0, self.1 * x.1, self.2 * x.2, self.3 * x.3)
    }
}

impl Div for f32x4 {
    type Output = Self;
    #[inline]
    fn div(self, x: Self) -> Self {
        f32x4(self.0 / x.0, self.1 / x.1, self.2 / x.2, self.3 / x.3)
    }
}

impl Add for f64x2 {
    type Output = Self;
    #[inline]
    fn add(self, x: Self) -> Self {
        f64x2(self.0 + x.0, self.1 + x.1)
    }
}

impl Sub for f64x2 {
    type Output = Self;
    #[inline]
    fn sub(self, x: Self) -> Self {
        f64x2(self.0 - x.0, self.1 - x.1)
    }
}

impl Mul for f64x2 {
    type Output = Self;
    #[inline]
    fn mul(self, x: Self) -> Self {
        f64x2(self.0 * x.0, self.1 * x.1)
    }
}

impl Div for f64x2 {
    type Output = Self;
    #[inline]
    fn div(self, x: Self) -> Self {
        f64x2(self.0 / x.0, self.1 / x.1)
    }
}

pub type Reg32 = f32x4;

pub type Reg64 = f64x2;

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
        let a = Complex::<f32>::new(self.0, self.1);
        let b = Complex::<f32>::new(self.2, self.3);
        let a = a * value;
        let b = b * value;
        f32x4::new(a.re, a.im, b.re, b.im)
    }

    #[inline]
    fn mul_complex(self, value: f32x4) -> f32x4 {
        let a = Complex::<f32>::new(self.0, self.1);
        let b = Complex::<f32>::new(self.2, self.3);
        let c = Complex::<f32>::new(value.0, value.1);
        let a = a * c;
        let d = Complex::<f32>::new(value.2, value.3);
        let b = b * d;
        f32x4::new(a.re, a.im, b.re, b.im)
    }

    #[inline]
    fn div_complex(self, value: f32x4) -> f32x4 {
        let a = Complex::<f32>::new(self.0, self.1);
        let b = Complex::<f32>::new(self.2, self.3);
        let c = Complex::<f32>::new(value.0, value.1);
        let a = a / c;
        let d = Complex::<f32>::new(value.2, value.3);
        let b = b / d;
        f32x4::new(a.re, a.im, b.re, b.im)
    }

    #[inline]
    fn complex_abs_squared(self) -> f32x4 {
        let squared = self * self;
        f32x4::new(squared.0 + squared.1, squared.2 + squared.3, 0.0, 0.0)
    }

    #[inline]
    fn complex_abs(self) -> f32x4 {
        let squared = self * self;
        f32x4::new((squared.0 + squared.1).sqrt(),
                   (squared.2 + squared.3).sqrt(),
                   0.0,
                   0.0)
    }
    #[inline]

    fn complex_abs_squared2(self) -> f32x4 {
        let squared = self * self;
        f32x4::new(squared.0 + squared.1, 0.0, squared.2 + squared.3, 0.0)
    }

    #[inline]
    fn complex_abs2(self) -> f32x4 {
        let squared = self * self;
        f32x4::new((squared.0 + squared.1).sqrt(),
                   0.0,
                   (squared.2 + squared.3).sqrt(),
                   0.0)
    }

    #[inline]
    fn sqrt(self) -> f32x4 {
        f32x4::new(self.0.sqrt(), self.1.sqrt(), self.2.sqrt(), self.3.sqrt())
    }

    #[inline]
    fn store_half_unchecked(self, target: &mut [f32], index: usize) {
        unsafe {
            *target.get_unchecked_mut(index) = self.0;
            *target.get_unchecked_mut(index + 1) = self.1;
        }
    }

    #[inline]
    fn sum_real(&self) -> f32 {
        self.0 + self.1 + self.2 + self.3
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(self.0 + self.2, self.1 + self.3)
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
        let complex = Complex::new(self.0, self.1);
        let result = complex * value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn mul_complex(self, value: f64x2) -> f64x2 {
        let complex = Complex::new(self.0, self.1);
        let value = Complex::new(value.0, value.1);
        let result = complex * value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn div_complex(self, value: f64x2) -> f64x2 {
        let complex = Complex::new(self.0, self.1);
        let value = Complex::new(value.0, value.1);
        let result = complex / value;
        f64x2::new(result.re, result.im)
    }

    #[inline]
    fn complex_abs_squared(self) -> f64x2 {
        let a = self.0;
        let b = self.1;
        let result = a * a + b * b;
        f64x2::new(result, 0.0)
    }

    #[inline]
    fn complex_abs(self) -> f64x2 {
        let a = self.0;
        let b = self.1;
        let result = (a * a + b * b).sqrt();
        f64x2::new(result, 0.0)
    }

    #[inline]
    fn complex_abs_squared2(self) -> f64x2 {
        let a = self.0;
        let b = self.1;
        let result = a * a + b * b;
        f64x2::new(result, 0.0)
    }

    #[inline]
    fn complex_abs2(self) -> f64x2 {
        let a = self.0;
        let b = self.1;
        let result = (a * a + b * b).sqrt();
        f64x2::new(result, 0.0)
    }

    #[inline]
    fn sqrt(self) -> f64x2 {
        f64x2::new(self.0.sqrt(), self.1.sqrt())
    }

    #[inline]
    fn store_half_unchecked(self, target: &mut [f64], index: usize) {
        unsafe {
            *target.get_unchecked_mut(index) = self.0;
        }
    }

    #[inline]
    fn sum_real(&self) -> f64 {
        self.0 + self.1
    }

    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(self.0, self.1)
    }
}
