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
	fn store_half(self, target: &mut [f32], index: usize);
	fn mul_complex(self, value: f32x4) -> f32x4;
	fn div_complex(self, value: f32x4) -> f32x4;
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