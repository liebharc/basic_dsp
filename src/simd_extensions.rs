use simd::f32x4;
use simd::x86::sse3::Sse3F32x4;
use num::complex::Complex32;

pub trait SimdExtensions32
{
	fn add_real(self, value: f32) -> f32x4;
	fn add_complex(self, value: Complex32) -> f32x4;
	fn scale_real(self, value: f32) -> f32x4;
	fn scale_complex(self, value: Complex32) -> f32x4;
	fn complex_abs_squared(self) -> f32x4;
	fn complex_abs(self) -> f32x4;
}

impl SimdExtensions32 for f32x4
{
	fn add_real(self, value: f32) -> f32x4
	{
		let increment = f32x4::splat(value);
		self + increment
	}
	
	fn add_complex(self, value: Complex32) -> f32x4
	{
		let increment = f32x4::load(&[value.re, value.im, value.re, value.im], 0);
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
} 