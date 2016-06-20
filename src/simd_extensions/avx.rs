use num::complex::Complex;
use super::Simd;
use simd::x86::avx::{f32x8,f64x4,AvxF32x8,AvxF64x4};
use std::mem;

pub type Reg32 = f32x8;

pub type Reg64 = f64x4;

impl Simd<f32> for f32x8
{
    fn array_to_regs(array: &[f32]) -> &[Self] {
        unsafe { 
            let len = array.len();
            let reg_len = Self::len();
            if len % reg_len != 0 {
                panic!("Argument must be dividable by {}", reg_len);
            }
            let trans: &[Self] = mem::transmute(array);
            &trans[0 .. len / reg_len]
        }
   }
    
    fn array_to_regs_mut(array: &mut [f32]) -> &mut [Self] {
        unsafe { 
            let len = array.len();
            let reg_len = Self::len();
            if len % reg_len != 0 {
                panic!("Argument must be dividable by {}", reg_len);
            }
            let trans: &mut [Self] = mem::transmute(array);
            &mut trans[0 .. len / reg_len]
        }
    }
    
    fn len() -> usize {
        8
    }
    
    fn load(array: &[f32], idx: usize) -> f32x8 {
        f32x8::load(array, idx)
    }
    
    fn load_wrap(array: &[f32], idx: usize) -> f32x8 {
        let mut temp = [0.0; 8];
        for i in 0..temp.len() {
            temp[i] = array[(idx + i) % array.len()];
        }
        f32x8::load(&temp, 0)
    }
    
    fn from_complex(value: Complex<f32>) -> f32x8 {
        f32x8::new(value.re, value.im, value.re, value.im,
                   value.re, value.im, value.re, value.im)
    }
    
    fn add_real(self, value: f32) -> f32x8
    {
        let increment = f32x8::splat(value);
        self + increment
    }
    
    fn add_complex(self, value: Complex<f32>) -> f32x8
    {
        let increment = f32x8::from_complex(value);
        self + increment
    }
    
    fn scale_real(self, value: f32) -> f32x8
    {
        let scale_vector = f32x8::splat(value); 
        self * scale_vector
    }
    
    fn scale_complex(self, value: Complex<f32>) -> f32x8
    {
        let scaling_real = f32x8::splat(value.re);
        let scaling_imag = f32x8::splat(value.im);
        let parallel = scaling_real * self;
        // There should be a shufps operation which shuffles the vector self
        let shuffled = f32x8::new(self.extract(1), self.extract(0), self.extract(3), self.extract(2),
                                  self.extract(5), self.extract(4), self.extract(7), self.extract(6)); 
        let cross = scaling_imag * shuffled;
        parallel.addsub(cross)
    }
    
    fn mul_complex(self, value: f32x8) -> f32x8
    {
        let scaling_real = f32x8::new(value.extract(0), value.extract(0), value.extract(2), value.extract(2),
                                      value.extract(4), value.extract(4), value.extract(6), value.extract(6));
        let scaling_imag = f32x8::new(value.extract(1), value.extract(1), value.extract(3), value.extract(3),
                                      value.extract(5), value.extract(5), value.extract(7), value.extract(7));
        let parallel = scaling_real * self;
        // There should be a shufps operation which shuffles the vector self
        let shuffled = f32x8::new(self.extract(1), self.extract(0), self.extract(3), self.extract(2),
                                  self.extract(5), self.extract(4), self.extract(7), self.extract(6)); 
        let cross = scaling_imag * shuffled; 
        parallel.addsub(cross)
    }
    
    fn div_complex(self, value: f32x8) -> f32x8
    {
                let scaling_imag = f32x8::new(self.extract(0), self.extract(0), self.extract(2), self.extract(2),
                                      self.extract(4), self.extract(4), self.extract(6), self.extract(6));
                let scaling_real = f32x8::new(self.extract(1), self.extract(1), self.extract(3), self.extract(3),
                                      self.extract(5), self.extract(5), self.extract(7), self.extract(7));
                let parallel = scaling_real * value;
                // There should be a shufps operation which shuffles the vector self
                let shuffled = f32x8::new(value.extract(1), value.extract(0), value.extract(3), value.extract(2),
                                  value.extract(5), value.extract(4), value.extract(7), value.extract(6));
                let cross = scaling_imag * shuffled;
                let mul = parallel.addsub(cross);
                let square = shuffled * shuffled;
                let square_shuffled = f32x8::new(square.extract(1), square.extract(0), square.extract(3), square.extract(2),
                                         square.extract(5), square.extract(4), square.extract(7), square.extract(6));
                let sum = square + square_shuffled;
                let div = mul / sum;
                f32x8::new(div.extract(1), div.extract(0), div.extract(3), div.extract(2),
                   div.extract(5), div.extract(4), div.extract(7), div.extract(6))
    }
    
    fn complex_abs_squared(self) -> f32x8
    {
        let squared = self * self;
        squared.hadd(squared)
    }
    
    fn complex_abs(self) -> f32x8
    {
        let squared_sum = self.complex_abs_squared();
        AvxF32x8::sqrt(squared_sum)
    }
    
    fn sqrt(self) -> f32x8 {
        AvxF32x8::sqrt(self)
    }
    
    fn store(self, target: &mut [f32], index: usize)
    {
        self.store(target, index);
    }
    
    fn store_half(self, target: &mut [f32], index: usize)
    {
        let mut temp = [0.0; 8];
        self.store(&mut temp, 0);
        target[index] = temp[0];
        target[index + 1] = temp[1];
        target[index + 2] = temp[4];
        target[index + 3] = temp[5];
    }
    
    fn sum_real(&self) -> f32 {
        self.extract(0) +
        self.extract(1) +
        self.extract(2) +
        self.extract(3) +
        self.extract(4) +
        self.extract(5) +
        self.extract(6) +
        self.extract(7)
    }
    
    fn sum_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(self.extract(0) + self.extract(2) + self.extract(4) + self.extract(6),
                            self.extract(1) + self.extract(3) + self.extract(5) + self.extract(7))
    }
}

impl Simd<f64> for f64x4
{
    fn array_to_regs(array: &[f64]) -> &[Self] {
        unsafe { 
            let len = array.len();
            let reg_len = Self::len();
            if len % reg_len != 0 {
                panic!("Argument must be dividable by {}", reg_len);
            }
            let trans: &[Self] = mem::transmute(array);
            &trans[0 .. len / reg_len]
        }
    }
    
    fn array_to_regs_mut(array: &mut [f64]) -> &mut [Self] {
        unsafe { 
            let len = array.len();
            let reg_len = Self::len();
            if len % reg_len != 0 {
                panic!("Argument must be dividable by {}", reg_len);
            }
            let trans: &mut [Self] = mem::transmute(array);
            &mut trans[0 .. len / reg_len]
        }
    }
    
    fn len() -> usize {
        4
    }
    
    fn load(array: &[f64], idx: usize) -> f64x4 {
        f64x4::load(array, idx)
    }
    
    fn load_wrap(array: &[f64], idx: usize) -> f64x4 {
        let mut temp = [0.0; 4];
        for i in 0..temp.len() {
            temp[i] = array[(idx + i) % array.len()];
        }
        f64x4::load(&temp, 0)
    }
    
    fn from_complex(value: Complex<f64>) -> f64x4 {
        f64x4::new(value.re, value.im, value.re, value.im)
    }
    
    fn add_real(self, value: f64) -> f64x4
    {
        let increment = f64x4::splat(value);
        self + increment
    }
    
    fn add_complex(self, value: Complex<f64>) -> f64x4
    {
        let increment = f64x4::new(value.re, value.im, value.re, value.im);
        self + increment
    }
    
    fn scale_real(self, value: f64) -> f64x4
    {
        let scale_vector = f64x4::splat(value); 
        self * scale_vector
    }
    
    fn scale_complex(self, value: Complex<f64>) -> f64x4
    {
        let scaling_real = f64x4::splat(value.re);
        let scaling_imag = f64x4::splat(value.im);
        let parallel = scaling_real * self;
        // There should be a shufps operation which shuffles the vector self
        let shuffled = f64x4::new(self.extract(1), self.extract(0), self.extract(3), self.extract(2)); 
        let cross = scaling_imag * shuffled;
        parallel.addsub(cross)
    }
    
    fn mul_complex(self, value: f64x4) -> f64x4
    {
        let scaling_real = f64x4::new(value.extract(0), value.extract(0), value.extract(2), value.extract(2));
        let scaling_imag = f64x4::new(value.extract(1), value.extract(1), value.extract(3), value.extract(3));
        let parallel = scaling_real * self;
        // There should be a shufps operation which shuffles the vector self
        let shuffled = f64x4::new(self.extract(1), self.extract(0), self.extract(3), self.extract(2)); 
        let cross = scaling_imag * shuffled;
        parallel.addsub(cross)
    }
    
    fn div_complex(self, value: f64x4) -> f64x4
    {
        let scaling_imag = f64x4::new(self.extract(0), self.extract(0), self.extract(2), self.extract(2));
        let scaling_real = f64x4::new(self.extract(1), self.extract(1), self.extract(3), self.extract(3));
        let parallel = scaling_real * value;
        // There should be a shufps operation which shuffles the vector self
        let shuffled = f64x4::new(value.extract(1), value.extract(0), value.extract(3), value.extract(2)); 
        let cross = scaling_imag * shuffled;
        let mul = parallel.addsub(cross);
        let square = shuffled * shuffled;
        let square_shuffled = f64x4::new(square.extract(1), square.extract(0), square.extract(3), square.extract(2));
        let sum = square + square_shuffled;
        let div = mul / sum;
        f64x4::new(div.extract(1), div.extract(0), div.extract(3), div.extract(2))
    }
    
    fn complex_abs_squared(self) -> f64x4
    {
        let squared = self * self;
        squared.hadd(squared)
    }
    
    fn complex_abs(self) -> f64x4
    {
        let squared_sum = self.complex_abs_squared();
        AvxF64x4::sqrt(squared_sum)
    }
    
    fn sqrt(self) -> f64x4 {
        f64x4::new(self.extract(0).sqrt(), self.extract(1).sqrt(), self.extract(2).sqrt(), self.extract(3).sqrt())
    }
    
    fn store(self, target: &mut [f64], index: usize)
    {
        self.store(target, index);
    } 
    
    fn store_half(self, target: &mut [f64], index: usize)
    {
        let mut temp = [0.0; 4];
        self.store(&mut temp, 0);
        target[index] = temp[0];
        target[index + 1] = temp[2];
    }
    
    fn sum_real(&self) -> f64 {
        self.extract(0) +
        self.extract(1) +
        self.extract(2) +
        self.extract(3)
    }
    
    fn sum_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(self.extract(0) + self.extract(2), self.extract(1) + self.extract(3))
    }
}
