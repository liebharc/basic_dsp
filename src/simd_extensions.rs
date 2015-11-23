use simd::f32x4;
use simd::x86::sse3::Sse3F32x4;
use num::complex::{Complex32, Complex64};
use simd::x86::sse2::f64x2; // Using the avx f64x4 would be more attractive but isn't supported by all x86 CPUs

pub trait SimdExtensions
{
	type Real;
	type Complex;
	fn add_real(self, value: Self::Real) -> Self;
	fn add_complex(self, value: Self::Complex) -> Self;
	fn scale_real(self, value: Self::Real) -> Self;
	fn scale_complex(self, value: Self::Complex) -> Self;
	fn complex_abs_squared(self) -> Self;
	fn complex_abs(self) -> Self;
	fn store_half(self, target: &mut [Self::Real], index: usize);
	fn mul_complex(self, value: Self) -> Self;
	fn div_complex(self, value: Self) -> Self;
}

impl SimdExtensions for f32x4
{
	type Real = f32;
	type Complex = Complex32;
	
	fn add_real(self, value: f32) -> f32x4
	{
		let increment = f32x4::splat(value);
		self + increment
	}
	
	fn add_complex(self, value: Complex32) -> f32x4
	{
		let increment = f32x4::new(value.re, value.im, value.re, value.im);
		self + increment
	}
	
	fn scale_real(self, value: f32) -> f32x4
	{
		let scale_vector = f32x4::splat(value); 
		self * scale_vector
	}
	
	fn scale_complex(self, value: Complex32) -> f32x4
	{
		let scaling_real = f32x4::splat(value.re);
		let scaling_imag = f32x4::splat(value.im);
		let parallel = scaling_real * self;
		// There should be a shufps operation which shuffles the vector self
		let shuffled = f32x4::new(self.extract(1), self.extract(0), self.extract(3), self.extract(2)); 
		let cross = scaling_imag * shuffled;
		parallel.addsub(cross)
	}
	
	fn mul_complex(self, value: f32x4) -> f32x4
	{
		let scaling_real = f32x4::new(value.extract(0), value.extract(0), value.extract(2), value.extract(2));
		let scaling_imag = f32x4::new(value.extract(1), value.extract(1), value.extract(3), value.extract(3));
		let parallel = scaling_real * self;
		// There should be a shufps operation which shuffles the vector self
		let shuffled = f32x4::new(self.extract(1), self.extract(0), self.extract(3), self.extract(2)); 
		let cross = scaling_imag * shuffled;
		parallel.addsub(cross)
	}
	
	fn div_complex(self, value: f32x4) -> f32x4
	{
		let scaling_imag = f32x4::new(self.extract(0), self.extract(0), self.extract(2), self.extract(2));
		let scaling_real = f32x4::new(self.extract(1), self.extract(1), self.extract(3), self.extract(3));
		let parallel = scaling_real * value;
		// There should be a shufps operation which shuffles the vector self
		let shuffled = f32x4::new(value.extract(1), value.extract(0), value.extract(3), value.extract(2)); 
		let cross = scaling_imag * shuffled;
		let mul = parallel.addsub(cross);
		let square = shuffled * shuffled;
		let square_shuffled = f32x4::new(square.extract(1), square.extract(0), square.extract(3), square.extract(2));
		let sum = square + square_shuffled;
		let div = mul / sum;
		f32x4::new(div.extract(1), div.extract(0), div.extract(3), div.extract(2))
	}
	
	fn complex_abs_squared(self) -> f32x4
	{
		let squared = self * self;
		squared.hadd(squared)
	}
	
	fn complex_abs(self) -> f32x4
	{
		let squared = self * self;
		let squared_sum = squared.hadd(squared);
		squared_sum.sqrt()
	}
	
	fn store_half(self, target: &mut [f32], index: usize)
	{
		let mut temp = [0.0; 4];
		self.store(&mut temp, 0);
		target[index] = temp[0];
		target[index + 1] = temp[1];
	} 
} 

impl SimdExtensions for f64x2
{
	type Real = f64;
	type Complex = Complex64;
	
	fn add_real(self, value: f64) -> f64x2
	{
		let increment = f64x2::splat(value);
		self + increment
	}
	
	fn add_complex(self, value: Complex64) -> f64x2
	{
		let increment = f64x2::new(value.re, value.im);
		self + increment
	}
	
	fn scale_real(self, value: f64) -> f64x2
	{
		let scale_vector = f64x2::splat(value); 
		self * scale_vector
	}
	
	fn scale_complex(self, value: Complex64) -> f64x2
	{
		let complex = Complex64::new(self.extract(0), self.extract(1));
		let result = complex * value;
		f64x2::new(result.re, result.im)
	}
	
	fn mul_complex(self, value: f64x2) -> f64x2
	{
		let complex = Complex64::new(self.extract(0), self.extract(1));
		let value = Complex64::new(value.extract(0), value.extract(1));
		let result = complex * value;
		f64x2::new(result.re, result.im)
	}
	
	fn div_complex(self, value: f64x2) -> f64x2
	{
		let complex = Complex64::new(self.extract(0), self.extract(1));
		let value = Complex64::new(value.extract(0), value.extract(1));
		let result = complex / value;
		f64x2::new(result.re, result.im)
	}
	
	fn complex_abs_squared(self) -> f64x2
	{
		let a = self.extract(0);
		let b = self.extract(1);
		let result = a * a + b * b;
		f64x2::new(result, result)
	}
	
	fn complex_abs(self) -> f64x2
	{
		let a = self.extract(0);
		let b = self.extract(1);
		let result = (a * a + b * b).sqrt();
		f64x2::new(result, result)
	}
	
	fn store_half(self, target: &mut [f64], index: usize)
	{
		target[index] = self.extract(0);
	} 
} 