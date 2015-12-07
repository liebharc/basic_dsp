use multicore_support::{Chunk, Complexity};
use super::super::general::{
	DataVector,
	GenericVectorOperations,
    VecResult,
    ErrorReason};
use super::DataVector32;
use simd::f32x4;
use num::complex::Complex32;
use simd_extensions::SimdExtensions;

#[inline]
impl GenericVectorOperations for DataVector32
{
	fn add_vector(mut self, summand: &Self) -> VecResult<Self>
	{
		{
			let len = self.len();
            reject_if!(self, len != summand.len(), ErrorReason::VectorsMustHaveTheSameSize);
			
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &summand.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector1 + vector2;
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] + other[i];
				i += 1;
			}
		}
		
		Ok(self)
	}
	
	fn subtract_vector(mut self, subtrahend: &Self) -> VecResult<Self>
	{
		{
			let len = self.len();
			reject_if!(self, len != subtrahend.len(), ErrorReason::VectorsMustHaveTheSameSize);
				
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &subtrahend.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4,   |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2 - vector1;
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] - other[i];
				i += 1;
			}
		}
		
		Ok(self)
	}
	
	fn multiply_vector(self, factor: &Self) -> VecResult<Self>
	{
		let len = self.len();
		reject_if!(self, len != factor.len(), ErrorReason::VectorsMustHaveTheSameSize);
		
		if self.is_complex
		{
			Ok(self.multiply_vector_complex(factor))
		}
		else
		{
			Ok(self.multiply_vector_real(factor))
		}
	}
	
	fn divide_vector(self, divisor: &Self) -> VecResult<Self>
	{
		let len = self.len();
		reject_if!(self, len != divisor.len(), ErrorReason::VectorsMustHaveTheSameSize);
		
		if self.is_complex
		{
			Ok(self.divide_vector_complex(divisor))
		}
		else
		{
			Ok(self.divide_vector_real(divisor))
		}
	}
	
	fn zero_pad(mut self, points: usize) -> VecResult<Self>
	{
		{
			let len_before = self.len();
			let len = if self.is_complex { 2 * points } else { points };
			self.reallocate(len);
			let array = &mut self.data;
			for i in len_before..len
			{
				array[i] = 0.0;
			}
		}
		
		Ok(self)
	}
	
	fn zero_interleave(self) -> VecResult<Self>
	{
		if self.is_complex
		{
			Ok(self.zero_interleave_complex())
		}
		else
		{
			Ok(self.zero_interleave_real())
		}
	}
	
	fn diff(mut self) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let mut target = &mut self.temp;
			let org = &self.data;
			if self.is_complex {
				self.valid_len -= 2;
				Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 2, &mut target, data_length, 2, |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    let mut len = target.len();
                    if range.end == original.len() - 1
                    {
                        len -= 2;
                    }
                    
                    while i < len
                    { 
                        target[i] = original[j + 2] - original[i];
                        i += 1;
                        j += 1;
                    }
                });
			}
			else {
				self.valid_len -= 1;
				Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 1, &mut target, data_length, 1, |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    let mut len = target.len();
                    if range.end >= original.len() - 1
                    {
                        len -= 1;
                    }
                        
                    while i < len
                    { 
                        target[i] = original[j + 1] - original[j];
                        i += 1;
                        j += 1;
                    }
                });
			}
		}
		
		Ok(self.swap_data_temp())
	}
	
	fn diff_with_start(mut self) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let mut target = &mut self.temp;
			let org = &self.data;
			if self.is_complex {
				Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 2, &mut target, data_length, 2, |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    if j == 0 {
                        i = 2;
                        j = 2;
                        target[0] = original[0];
                        target[1] = original[1];
                    }
                    
                    while i < target.len()
                    { 
                        target[i] = original[j] - original[j - 2];
                        i += 1;
                        j += 1;
                    }
                });
			}
			else {
				Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 1, &mut target, data_length, 1, |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    if j == 0 {
                        i = 1;
                        j = 1;
                        target[0] = original[0];
                    }
                    
                    while i < target.len()
                    { 
                        target[i] = original[j] - original[j - 1];
                        i += 1;
                        j += 1;
                    }
                });
			}
		}
		
		Ok(self.swap_data_temp())
	}
	
	fn cum_sum(mut self) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let mut data = &mut self.data;
			let mut i = 0;
			let mut j = 1;
			if self.is_complex {
				j = 2;
			}
			
			while j < data_length {
				data[j] = data[j] + data[i];
				i += 1;
				j += 1;
			}
		}
		Ok(self)
	}
}

impl DataVector32 {
    fn multiply_vector_complex(mut self, factor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &factor.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4, |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2.mul_complex(vector1);
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				let complex1 = Complex32::new(array[i], array[i + 1]);
				let complex2 = Complex32::new(other[i], other[i + 1]);
				let result = complex1 * complex2;
				array[i] = result.re;
				array[i + 1] = result.im;
				i += 2;
			}
		}
		self
	}
	
	fn multiply_vector_real(mut self, factor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &factor.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4, |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2 * vector1;
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }        
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] * other[i];
				i += 1;
			}
		}
		self
	}
	
	fn divide_vector_complex(mut self, divisor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &divisor.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2.div_complex(vector1);
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				let complex1 = Complex32::new(array[i], array[i + 1]);
				let complex2 = Complex32::new(other[i], other[i + 1]);
				let result = complex1 / complex2;
				array[i] = result.re;
				array[i + 1] = result.im;
				i += 2;
			}
		}
		self
	}
	
	fn divide_vector_real(mut self, divisor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &divisor.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2 / vector1;
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] / other[i];
				i += 1;
			}
		}
		self
	}
    
    fn zero_interleave_complex(mut self) -> Self
	{
		{
			let new_len = 2 * self.len();
			self.reallocate(new_len);
			let data_length = new_len;
			let mut target = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(Complexity::Small, &source, data_length, 4, &mut target, data_length, 4,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len() / 2 {
                    if i % 2 == 0
                    {
                        target[2 * i] = original[j];
                        target[2 * i + 1] = original[j + 1];
                        j += 2;
                    }
                    else
                    {
                        target[2 * i] = 0.0;
                        target[2 * i + 1] = 0.0;
                    }
                    
                    i += 1;
                }
            });
		}
		self.swap_data_temp()
	}
	
	fn zero_interleave_real(mut self) -> Self
	{
		{
			let new_len = 2 * self.len();
			self.reallocate(new_len);
			let data_length = new_len;
			let mut target = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(Complexity::Small, &source, data_length, 4, &mut target, data_length, 2,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len() {
                    if i % 2 == 0
                    {
                        target[i] = original[j];
                        j += 1;
                    }
                    else
                    {
                        target[i] = 0.0;
                    }
                    
                    i += 1;
                }
            });
		}
		self.swap_data_temp()
	}
}