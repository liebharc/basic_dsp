use simd::f32x4;
use num::complex::Complex;
use super::Simd;
use simd::x86::sse2::f64x2;

pub type Reg32 = f32x4;

pub type Reg64 = f64x2;

impl Simd<f32> for f32x4
{
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
        for i in 0..temp.len() {
            temp[i] = unsafe {*array.get_unchecked((idx + i) % array.len())};
        }
        f32x4::load(&temp, 0)
    }
    
    #[inline]
    fn from_complex(value: Complex<f32>) -> f32x4 {
        f32x4::new(value.re, value.im, value.re, value.im)
    }
    
    #[inline]
    fn add_real(self, value: f32) -> f32x4
    {
        let increment = f32x4::splat(value);
        self + increment
    }
    
    #[inline]
    fn add_complex(self, value: Complex<f32>) -> f32x4
    {
        let increment = f32x4::new(value.re, value.im, value.re, value.im);
        self + increment
    }
    
    #[inline]
    fn scale_real(self, value: f32) -> f32x4
    {
        let scale_vector = f32x4::splat(value); 
        self * scale_vector
    }
    
    #[inline]
    fn scale_complex(self, value: Complex<f32>) -> f32x4
    {
        let a = Complex::<f32>::new(self.extract(0), self.extract(1));
        let b = Complex::<f32>::new(self.extract(2), self.extract(3));
        let a = a * value;
        let b = b * value;
        f32x4::new(a.re, a.im, b.re, b.im)
    }
    
    #[inline]
    fn mul_complex(self, value: f32x4) -> f32x4
    {
        let a = Complex::<f32>::new(self.extract(0), self.extract(1));
        let b = Complex::<f32>::new(self.extract(2), self.extract(3));
        let c = Complex::<f32>::new(value.extract(0), value.extract(1));
        let a = a * c;
        let d = Complex::<f32>::new(value.extract(2), value.extract(3));
        let b = b * d;
        f32x4::new(a.re, a.im, b.re, b.im)
    }
    
    #[inline]
    fn div_complex(self, value: f32x4) -> f32x4
    {
        let a = Complex::<f32>::new(self.extract(0), self.extract(1));
        let b = Complex::<f32>::new(self.extract(2), self.extract(3));
        let c = Complex::<f32>::new(value.extract(0), value.extract(1));
        let a = a / c;
        let d = Complex::<f32>::new(value.extract(2), value.extract(3));
        let b = b / d;
        f32x4::new(a.re, a.im, b.re, b.im)
    }
    
    #[inline]
    fn complex_abs_squared(self) -> f32x4
    {
        let squared = self * self;
        f32x4::new(squared.extract(0) + squared.extract(1), squared.extract(2) + squared.extract(3), 0.0, 0.0)
    }
    
    #[inline]
    fn complex_abs(self) -> f32x4
    {
        let squared = self * self;
        f32x4::new((squared.extract(0) + squared.extract(1)).sqrt(), (squared.extract(2) + squared.extract(3)).sqrt(), 0.0, 0.0)
    }
    #[inline]
    
    fn complex_abs_squared2(self) -> f32x4
    {
        let squared = self * self;
        f32x4::new(squared.extract(0) + squared.extract(1), 0.0, squared.extract(2) + squared.extract(3), 0.0)
    }
    
    #[inline]
    fn complex_abs2(self) -> f32x4
    {
        let squared = self * self;
        f32x4::new((squared.extract(0) + squared.extract(1)).sqrt(), 0.0, (squared.extract(2) + squared.extract(3)).sqrt(), 0.0)
    }
    
    #[inline]
    fn sqrt(self) -> f32x4 {
        self.sqrt()
    }
    
    #[inline]
    fn store_half_unchecked(self, target: &mut [f32], index: usize)
    {
        unsafe {
            *target.get_unchecked_mut(index) = self.extract(0);
            *target.get_unchecked_mut(index + 1) = self.extract(1);
        }
    }
    
    #[inline]
    fn sum_real(&self) -> f32 {
        self.extract(0) +
        self.extract(1) +
        self.extract(2) +
        self.extract(3)
    }
    
    #[inline]
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(self.extract(0) + self.extract(2), self.extract(1) + self.extract(3))
    }
} 

impl Simd<f64> for f64x2
{
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
        for i in 0..temp.len() {
            temp[i] = unsafe {*array.get_unchecked((idx + i) % array.len())};
        }
        
        f64x2::load(&temp, 0)
    }
    
    #[inline]
    fn from_complex(value: Complex<f64>) -> f64x2 {
        f64x2::new(value.re, value.im)
    }
    
    #[inline]
    fn add_real(self, value: f64) -> f64x2
    {
        let increment = f64x2::splat(value);
        self + increment
    }
    
    #[inline]
    fn add_complex(self, value: Complex<f64>) -> f64x2
    {
        let increment = f64x2::new(value.re, value.im);
        self + increment
    }
    
    #[inline]
    fn scale_real(self, value: f64) -> f64x2
    {
        let scale_vector = f64x2::splat(value); 
        self * scale_vector
    }
    
    #[inline]
    fn scale_complex(self, value: Complex<f64>) -> f64x2
    {
        let complex = Complex::new(self.extract(0), self.extract(1));
        let result = complex * value;
        f64x2::new(result.re, result.im)
    }
    
    #[inline]
    fn mul_complex(self, value: f64x2) -> f64x2
    {
        let complex = Complex::new(self.extract(0), self.extract(1));
        let value = Complex::new(value.extract(0), value.extract(1));
        let result = complex * value;
        f64x2::new(result.re, result.im)
    }
    
    #[inline]
    fn div_complex(self, value: f64x2) -> f64x2
    {
        let complex = Complex::new(self.extract(0), self.extract(1));
        let value = Complex::new(value.extract(0), value.extract(1));
        let result = complex / value;
        f64x2::new(result.re, result.im)
    }
    
    #[inline]
    fn complex_abs_squared(self) -> f64x2
    {
        let a = self.extract(0);
        let b = self.extract(1);
        let result = a * a + b * b;
        f64x2::new(result, 0.0)
    }
    
    #[inline]
    fn complex_abs(self) -> f64x2
    {
        let a = self.extract(0);
        let b = self.extract(1);
        let result = (a * a + b * b).sqrt();
        f64x2::new(result, 0.0)
    }
    
    #[inline]
    fn complex_abs_squared2(self) -> f64x2
    {
        let a = self.extract(0);
        let b = self.extract(1);
        let result = a * a + b * b;
        f64x2::new(result, 0.0)
    }
    
    #[inline]
    fn complex_abs2(self) -> f64x2
    {
        let a = self.extract(0);
        let b = self.extract(1);
        let result = (a * a + b * b).sqrt();
        f64x2::new(result, 0.0)
    }
    
    #[inline]
    fn sqrt(self) -> f64x2 {
        f64x2::new(self.extract(0).sqrt(), self.extract(1).sqrt())
    }
    
    #[inline]
    fn store_half_unchecked(self, target: &mut [f64], index: usize)
    {
        unsafe {
            *target.get_unchecked_mut(index) = self.extract(0);
        }
    } 
    
    #[inline]
    fn sum_real(&self) -> f64 {
        self.extract(0) +
        self.extract(1)
    }
    
    #[inline]
    fn sum_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(self.extract(0), self.extract(1))
    }
} 
