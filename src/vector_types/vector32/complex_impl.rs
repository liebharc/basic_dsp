use multicore_support::Chunk;
use super::super::general::{
	DataVector,
	ComplexVectorOperations};
use super::{DataVector32, DEFAULT_GRANUALRITY};
use simd::f32x4;
use simd_extensions::SimdExtensions;
use num::complex::Complex32;
use num::traits::Float;
use std::ops::Range;

#[inline]
impl ComplexVectorOperations for DataVector32
{
	type RealPartner = DataVector32;
	type Complex = Complex32;
	
	fn complex_offset(mut self, offset: Complex32)  -> DataVector32
	{
		{
            let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
            let mut array = &mut self.data;
            let vector_offset = f32x4::new(offset.re, offset.im, offset.re, offset.im);
            Chunk::execute_partial_with_arguments(&mut array, vectorization_length, DEFAULT_GRANUALRITY, vector_offset, |array, v| {
                let mut i = 0;
                while i < array.len()
                {
                    let data = f32x4::load(array, i);
                    let result = data + v;
                    result.store(array, i);
                    i += 4;
                }
            });
            
            let mut i = vectorization_length;
            while i < data_length
			{
				array[i] += offset.re;
				array[i + 1] += offset.im;
				i += 2;
			}
        }
        
		self
	}
	
	fn complex_scale(mut self, factor: Complex32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			Chunk::execute_partial_with_arguments(&mut array, vectorization_length, DEFAULT_GRANUALRITY, factor, |array, value| {
                let mut i = 0;
                while i < array.len()
                { 
                    let vector = f32x4::load(array, i);
                    let scaled = vector.scale_complex(value);
                    scaled.store(array, i);
                    i += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				let complex = Complex32::new(array[i], array[i + 1]);
				let result = complex * factor;
				array[i] = result.re;
				array[i + 1] = result.im;
				i += 2;
			}
		}
		self
	}
	
	fn complex_abs(mut self) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let array = &self.data;
			let mut temp = &mut self.temp;
			Chunk::execute_original_to_target(&array, vectorization_length, 4, &mut temp, vectorization_length / 2, 2, DataVector32::complex_abs_simd);
			let mut i = vectorization_length;
			while i + 1 < data_length
			{
				temp[i / 2] = (array[i] * array[i] + array[i + 1] * array[i + 1]).sqrt();
				i += 2;
			}
			self.is_complex = false;
			self.valid_len = self.valid_len / 2;
		}
		
		self.swap_data_temp()
	}
	
	fn get_complex_abs(&self, destination: &mut DataVector32)
	{
		let data_length = self.len();
		destination.reallocate(data_length / 2);
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let array = &self.data;
		let mut temp = &mut destination.data;
		Chunk::execute_original_to_target(&array, vectorization_length, 4, &mut temp, vectorization_length / 2, 2, DataVector32::complex_abs_simd);
		let mut i = vectorization_length;
		while i + 1 < data_length
		{
			temp[i / 2] = (array[i] * array[i] + array[i + 1] * array[i + 1]).sqrt();
			i += 2;
		}
		
		destination.is_complex = false;
		destination.delta = self.delta;
	}
	
	fn complex_abs_squared(mut self) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let mut temp = &mut self.temp;
			Chunk::execute_partial_with_temp(&mut array, vectorization_length, 4, &mut temp, vectorization_length / 2, 2, |array, target| {
                let mut i = 0;
                let mut j = 0;
                while i < array.len()
                { 
                    let vector = f32x4::load(array, i);
                    let result = vector.complex_abs_squared();
                    result.store_half(target, j);
                    i += 4;
                    j += 2;
                }
            });
			let mut i = vectorization_length;
			while i + 1 < data_length
			{
				temp[i / 2] = array[i] * array[i] + array[i + 1] * array[i + 1];
				i += 2;
			}
			self.is_complex = false;
			self.valid_len = self.valid_len / 2;
		}
		self.swap_data_temp()
	}
	
	fn complex_conj(mut self) -> DataVector32
	{
		{
			let mut array = &mut self.data;
			Chunk::execute(&mut array, 2, |array| {
                let mut i = 1;
                while i < array.len() {
                    array[i] = -array[i];
                    i += 2;
                }
            });
		}
		
		self
	}
	
	fn to_real(mut self) -> DataVector32
	{
		{
			let len = self.len();
			let mut array = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1, |original, range, target| {
                let mut i = range.start;
                let mut j = 0;
                while j < target.len()
                { 
                    target[j] = original[i];
                    i += 2;
                    j += 1;
                }
            });
		}
		
		self.is_complex = false;
		self.valid_len = self.valid_len / 2;
		self.swap_data_temp()
	}

	fn to_imag(mut self) -> DataVector32
	{
		{
			let len = self.len();
			let mut array = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1, |original, range, target| {
                let mut i = range.start + 1;
                let mut j = 0;
                while j < target.len()
                { 
                    target[j] = original[i];
                    i += 2;
                    j += 1;
                }
            });
		}
		
		self.is_complex = false;
		self.valid_len = self.valid_len / 2;
		self.swap_data_temp()
	}	
			
	fn get_real(&self, destination: &mut DataVector32)
	{
		let len = self.len();
		destination.reallocate(len / 2);
		destination.delta = self.delta;
		destination.is_complex = false;
		let mut array = &mut destination.data;
		let source = &self.data;
		Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1, |original, range, target| {
            let mut i = range.start;
            let mut j = 0;
            while j < target.len()
            { 
                target[j] = original[i];
                i += 2;
                j += 1;
            }
        });
	}
	
	fn get_imag(&self, destination: &mut DataVector32)
	{
		let len = self.len();
		destination.reallocate(len / 2);
		destination.delta = self.delta;
		destination.is_complex = false;
		let mut array = &mut destination.data;
		let source = &self.data;
		Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1,  |original, range, target| {
            let mut i = range.start + 1;
            let mut j = 0;
            while j < target.len()
            { 
                target[j] = original[i];
                i += 2;
                j += 1;
            }
        });
	}
	
	fn phase(mut self) -> DataVector32
	{
		{
			let len = self.len();
			let mut array = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1,  DataVector32::phase_par);
		}
		
		self.is_complex = false;
		self.valid_len = self.valid_len / 2;
		self.swap_data_temp()
	}
	
	fn get_phase(&self, destination: &mut DataVector32)
	{
		let len = self.len();
		destination.reallocate(len / 2);
		destination.delta = self.delta;
		destination.is_complex = false;
		let mut array = &mut destination.data;
		let source = &self.data;
		Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1,  DataVector32::phase_par);
	}
}

impl DataVector32 {
    fn complex_abs_simd(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
		let mut i = 0;
		let mut j = range.start;
		while i < target.len()
		{ 
			let vector = f32x4::load(original, j);
			let result = vector.complex_abs();
			result.store_half(target, i);
			j += 4;
			i += 2;
		}
	}
    
    fn phase_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
		let mut i = range.start;
		let mut j = 0;
		while j < target.len()
		{ 
			let complex = Complex32::new(original[i], original[i + 1]);
			target[j] = complex.arg();
			i += 2;
			j += 1;
		}
	}
}