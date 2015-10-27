use simd::f32x4;
use num::complex::Complex32;

pub trait SimdExtensions32
{
	fn add_real(self, value: f32) -> f32x4;
	fn add_complex(self, value: Complex32) -> f32x4;
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
} 