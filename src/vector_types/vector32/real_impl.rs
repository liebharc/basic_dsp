use multicore_support::{Chunk, Complexity};
use super::super::general::{
	DataVector,
    VecResult,
    GenericVectorOperations,
	RealVectorOperations};
use super::DataVector32;
use simd::f32x4;
use simd_extensions::SimdExtensions;
use num::traits::Float;

#[inline]
impl RealVectorOperations for DataVector32
{
	type ComplexPartner = DataVector32;
	
	extern fn real_offset(mut self, offset: f32) -> VecResult<Self>
	{
        {
            let len = self.len();
            let mut array = &mut self.data;
            Chunk::execute_partial_with_arguments(Complexity::Small, &mut array, len, 1, offset, |array, v| {
                for i in 0..array.len()
                {
                    array[i] += v;
                }
            });
        }
		Ok(self)
	}
	
	extern fn real_scale(mut self, factor: f32) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			Chunk::execute_partial_with_arguments(Complexity::Small, &mut array, vectorization_length, 4, factor, |array, value| {
                let mut i = 0;
                while i < array.len()
                { 
                    let vector = f32x4::load(array, i);
                    let scaled = vector.scale_real(value);
                    scaled.store(array, i);
                    i += 4;
                }
            });
			for i in vectorization_length..data_length
			{
				array[i] = array[i] * factor;
			}
		}
		Ok(self)
	}
	
	extern fn real_abs(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Small, &mut array, length, 1, |chunk| {
                for i in 0..chunk.len() {
                    chunk[i] = chunk[i].abs();
                }
            });
		}
		Ok(self)
	}
	
	extern fn real_sqrt(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].sqrt();
                    i += 1;
                }
            });
		}
		Ok(self)
	}
	
	extern fn real_square(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i] * array[i];
                    i += 1;
                }
            });
		}
		Ok(self)
	}
	
	extern fn real_root(mut self, degree: Self::E) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, degree, |array, base| {
                let base = 1.0 / base;
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].powf(base);
                    i += 1;
                }
            });
		}
		Ok(self)
	}
	
	extern fn real_power(mut self, exponent: Self::E) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, exponent, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].powf(base);
                    i += 1;
                }
            });
		}
		Ok(self)
	}
	
	extern fn real_logn(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].ln();
                    i += 1;
                }
            });
		}
		Ok(self)
	}
	
	extern fn real_expn(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].exp();
                    i += 1;
                }
            });
		}
		Ok(self)
	}

	extern fn real_log_base(mut self, base: Self::E) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, base, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].log(base);
                    i += 1;
                }
            });
		}
		Ok(self)
	}
	
	extern fn real_exp_base(mut self, base: Self::E) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, base, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = base.powf(array[i]);
                    i += 1;
                }
            });
		}
		Ok(self)
	}
	
	extern fn to_complex(self) -> VecResult<Self>
	{
		let result = self.zero_interleave();
        match result {
            Ok(mut vec) => { 
                vec.is_complex = true;
                Ok(vec)
            },
            Err((r, mut vec)) => {
                vec.is_complex = true;
                Err((r, vec))
            }
        }
	}
	
	extern fn wrap(mut self, divisor: Self::E) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Small, &mut array, length, 1, divisor, |array, value| {
                let mut i = 0;
                while i < array.len() {
                    array[i] = array[i] % value;
                    i += 1;
                }
            });
		}
		Ok(self)
	}
	
	extern fn unwrap(mut self, divisor: Self::E) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let mut data = &mut self.data;
			let mut i = 0;
			let mut j = 1;
			let half = divisor / 2.0;
			while j < data_length {
				let mut diff = data[j] - data[i];
				diff = diff % divisor;
				if diff > half {
					diff -= divisor;
				}
				else if diff < -half {
					diff += divisor;
				}
				data[j] = data[i] + diff;
				
				i += 1;
				j += 1;
			}
		}
		Ok(self)
	}
    
    extern fn real_sin(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].sin();
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    extern fn real_cos(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].cos();
                    i += 1;
                }
            });
		}
		Ok(self)
	}
}