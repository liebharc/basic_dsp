use simd::f32x4;

pub trait SimdExtensions32
{
	fn add_real(self, value: f32) -> f32x4;
}

impl SimdExtensions32 for f32x4
{
	fn add_real(self, value: f32) -> f32x4
	{
		let increment = f32x4::splat(value);
		self + increment
	}
} 