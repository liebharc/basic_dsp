use multicore_support::Chunk;
use super::general::{
	DataVector,
	DataVectorDomain,
	GenericVectorOperations,
	RealVectorOperations,
	ComplexVectorOperations};
use simd::f32x4;
use simd_extensions::SimdExtensions32;
use num::complex::Complex32;
use num::traits::Float;
use std::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};
use std::mem;

/// An alternative way to define operations on a vector.
/// Warning: Highly unstable and not even fully implemented right now.
///
/// In future this enum will likely be deleted or hidden and be replaced with a builder
/// pattern. The advantage of this is that with the builder we have the means to define at 
/// compile time what kind of vector will result from the given set of operations.
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
#[allow(dead_code)]
pub enum Operation32
{
	AddReal(f32),
	AddComplex(Complex32),
	//AddVector(&'a DataVector32<'a>),
	MultiplyReal(f32),
	MultiplyComplex(Complex32),
	//MultiplyVector(&'a DataVector32<'a>),
	AbsReal,
	AbsComplex,
	Sqrt
}

define_vector_struct!(struct DataVector32, f32);
define_real_basic_struct_members!(impl DataVector32, DataVectorDomain::Time);
define_complex_basic_struct_members!(impl DataVector32, DataVectorDomain::Frequency);

define_vector_struct!(struct RealTimeVector32, f32);
define_real_basic_struct_members!(impl RealTimeVector32, DataVectorDomain::Time);
define_generic_operations_forward!(from: RealTimeVector32, to: DataVector32);
define_real_operations_forward!(from: RealTimeVector32, to: DataVector32, complex_partner: ComplexTimeVector32);

define_vector_struct!(struct RealFreqVector32, f32);
define_real_basic_struct_members!(impl RealFreqVector32, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: RealFreqVector32, to: DataVector32);
define_real_operations_forward!(from: RealFreqVector32, to: DataVector32, complex_partner: ComplexFreqVector32);

define_vector_struct!(struct ComplexTimeVector32, f32);
define_complex_basic_struct_members!(impl ComplexTimeVector32, DataVectorDomain::Time);
define_generic_operations_forward!(from: ComplexTimeVector32, to: DataVector32);
define_complex_operations_forward!(from: ComplexTimeVector32, to: DataVector32, complex: Complex32, real_partner: RealTimeVector32);

define_vector_struct!(struct ComplexFreqVector32, f32);
define_complex_basic_struct_members!(impl ComplexFreqVector32, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: ComplexFreqVector32, to: DataVector32);
define_complex_operations_forward!(from: ComplexFreqVector32, to: DataVector32, complex: Complex32, real_partner: RealTimeVector32);

const DEFAULT_GRANUALRITY: usize = 4;

#[inline]
impl GenericVectorOperations for DataVector32
{
	fn add_vector(mut self, summand: &DataVector32) -> DataVector32
	{
		{
			let len = self.len();
			if len != summand.len()
			{
				panic!("Vectors must have the same size");
			}
			
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &summand.data;
			Chunk::execute_original_to_target(&other,  &mut array, vectorization_length, 4,  DataVector32::add_vector_simd);
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] + other[i];
				i += 1;
			}
		}
		
		self
	}
	
	fn subtract_vector(mut self, subtrahend: &DataVector32) -> DataVector32
	{
		{
			let len = self.len();
			if len != subtrahend.len()
			{
				panic!("Vectors must have the same size");
			}
				
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &subtrahend.data;
			Chunk::execute_original_to_target(&other,  &mut array, vectorization_length, 4,  DataVector32::sub_vector_simd);
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] - other[i];
				i += 1;
			}
		}
		
		self
	}
	
	fn multiply_vector(self, factor: &DataVector32) -> DataVector32
	{
		let len = self.len();
		if len != factor.len()
		{
			panic!("Vectors must have the same size");
		}
		
		if self.is_complex
		{
			self.multiply_vector_complex(factor)
		}
		else
		{
			self.multiply_vector_real(factor)
		}
	}
	
	fn divide_vector(self, divisor: &DataVector32) -> DataVector32
	{
		let len = self.len();
		if len != divisor.len()
		{
			panic!("Vectors must have the same size");
		}
		
		if self.is_complex
		{
			self.divide_vector_complex(divisor)
		}
		else
		{
			self.divide_vector_real(divisor)
		}
	}
	
	fn zero_pad(mut self, points: usize) -> Self
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
		
		self
	}
	
	fn zero_interleave(self) -> Self
	{
		if self.is_complex
		{
			self.zero_interleave_complex()
		}
		else
		{
			self.zero_interleave_real()
		}
	}
}

#[inline]
impl RealVectorOperations for DataVector32
{
	type ComplexPartner = DataVector32;
	
	fn real_offset(mut self, offset: f32) -> DataVector32
	{
		self.inplace_offset(&[offset, offset, offset, offset]);
		self
	}
	
	fn real_scale(mut self, factor: f32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, DataVector32::inplace_real_scale_simd, factor);
			for i in vectorization_length..data_length
			{
				array[i] = array[i] * factor;
			}
		}
		
		self
	}
	
	fn real_abs(mut self) -> DataVector32
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(&mut array, length, 1, DataVector32::abs_real_par);
		}
		self
	}
	
	fn real_sqrt(mut self) -> DataVector32
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(&mut array, length, 1, DataVector32::real_sqrt_par);
		}
		self
	}
	
	fn to_complex(self) -> DataVector32
	{
		let mut result = self.zero_interleave_real();
		result.is_complex = true;
		result
	}
}

#[inline]
impl ComplexVectorOperations for DataVector32
{
	type RealPartner = DataVector32;
	type Complex = Complex32;
	
	fn complex_offset(mut self, offset: Complex32)  -> DataVector32
	{
		self.inplace_offset(&[offset.re, offset.im, offset.re, offset.im]);
		self
	}
	
	fn complex_scale(mut self, factor: Complex32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			Chunk::execute_partial_with_arguments(&mut array, vectorization_length, DEFAULT_GRANUALRITY, DataVector32::inplace_complex_scale_simd, factor);
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
			let mut array = &mut self.data;
			let mut temp = &mut self.temp;
			Chunk::execute_partial_with_temp(&mut array, vectorization_length, 4, &mut temp, vectorization_length / 2, 2, DataVector32::complex_abs_simd);
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
	
	fn complex_abs_squared(mut self) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let mut temp = &mut self.temp;
			Chunk::execute_partial_with_temp(&mut array, vectorization_length, 4, &mut temp, vectorization_length / 2, 2, DataVector32::complex_abs_squared_simd);
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
			Chunk::execute(&mut array, 2, DataVector32::complex_conj_par);
		}
		
		self
	}
}

#[inline]
impl DataVector32
{
	/// Perform a set of operations on the given vector. 
	/// Warning: Highly unstable and not even fully implemented right now.
	///
	/// With this approach we change how we operate on vectors. If you perform
	/// `M` operations on a vector with the length `N` you iterate wit hall other methods like this:
	///
	/// ```
	/// // pseudocode:
	/// // for m in M:
	/// //  for n in N:
	/// //    execute m on n
	/// ```
	///
	/// with this method the pattern is changed slighly:
	///
	/// ```
	/// // pseudocode:
	/// // for n in N:
	/// //  for m in M:
	/// //    execute m on n
	/// ```
	///
	/// Both variants have the same complexity however the second one is benificial since we
	/// have increased locality this way. This should help us by making better use of registers and 
	/// CPU buffers. This might also help since for large data we might have the chance in future to 
	/// move the data to a GPU, run all operations and get the result back. In this case the GPU is fast
	/// for many operations but the roundtrips on the bus should be minimized to keep the speed advantage.
	pub fn perform_operations(mut self, operations: &[Operation32])
		-> DataVector32
	{
		if operations.len() == 0
		{
			return DataVector32 { data: self.data, .. self };
		}
		
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		if scalar_length > 0
		{
			panic!("perform_operations requires right now that the array length is dividable by 4")
		}
		
		{
			let mut array = &mut self.data;
			Chunk::execute_partial_with_arguments(&mut array, vectorization_length, DEFAULT_GRANUALRITY, DataVector32::perform_operations_par, operations);
		}
		DataVector32 { data: self.data, .. self }
	}
	
	fn perform_operations_par(array: &mut [f32], operations: &[Operation32])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let mut vector = f32x4::load(array, i);
			let mut j = 0;
			while j < operations.len()
			{
				let operation = &operations[j];
				match *operation
				{
					Operation32::AddReal(value) =>
					{
						vector = vector.add_real(value);
					}
					Operation32::AddComplex(value) =>
					{
						vector = vector.add_complex(value);
					}
					/*Operation32::AddVector(value) =>
					{
						// TODO
					}*/
					Operation32::MultiplyReal(value) =>
					{
						vector = vector.scale_real(value);
					}
					Operation32::MultiplyComplex(value) =>
					{
						vector = vector.scale_complex(value);
					}
					/*Operation32::MultiplyVector(value) =>
					{
						// TODO
					}*/
					Operation32::AbsReal =>
					{
						vector.store(array, i);
						{
							let mut content = &mut array[i .. i + 4];
							let mut k = 0;
							while k < 4
							{
								content[k] = content[k].abs();
								k = k + 1;
							}
						}
						vector = f32x4::load(array, i);
					}
					Operation32::AbsComplex =>
					{
						vector = vector.complex_abs();
					}
					Operation32::Sqrt =>
					{
						vector.store(array, i);
						{
							let mut content = &mut array[i .. i + 4];
							let mut k = 0;
							while k < 4
							{
								content[k] = content[k].sqrt();
								k = k + 1;
							}
						}
						vector = f32x4::load(array, i);
					}
				}
				j += 1;
			}
		
			vector.store(array, i);	
			i += 4;
		}
	}
	
	fn reallocate(&mut self, length: usize)
	{
		if length > self.allocated_len()
		{
			let data = &mut self.data;
			data.resize(length, 0.0);
			let temp = &mut self.temp;
			temp.resize(length, 0.0);
		}
		
		self.valid_len = length;
	}
	
	fn add_vector_simd(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn sub_vector_simd(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn multiply_vector_complex(mut self, factor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &factor.data;
			Chunk::execute_original_to_target(&other,  &mut array, vectorization_length, 4,  DataVector32::mul_vector_complex_simd);
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
	
	fn mul_vector_complex_simd(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn multiply_vector_real(mut self, factor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &factor.data;
			Chunk::execute_original_to_target(&other,  &mut array, vectorization_length, 4,  DataVector32::mul_vector_real_simd);
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] * other[i];
				i += 1;
			}
		}
		self
	}
	
	fn mul_vector_real_simd(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn divide_vector_complex(mut self, divisor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &divisor.data;
			Chunk::execute_original_to_target(&other,  &mut array, vectorization_length, 4,  DataVector32::divide_vector_complex_simd);
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
	
	fn divide_vector_complex_simd(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn divide_vector_real(mut self, divisor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &divisor.data;
			Chunk::execute_original_to_target(&other,  &mut array, vectorization_length, 4,  DataVector32::div_vector_real_simd);
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] / other[i];
				i += 1;
			}
		}
		self
	}
	
	fn div_vector_real_simd(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn inplace_offset(&mut self, offset: &[f32; 4]) 
	{
		let increment_vector = f32x4::load(offset, 0); 
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, DEFAULT_GRANUALRITY, DataVector32::inplace_offset_simd, increment_vector);
		for i in vectorization_length..data_length
		{
			array[i] = array[i] + offset[i % 2];
		}
	}
		
	fn inplace_offset_simd(array: &mut [f32], increment_vector: f32x4)
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let vector = f32x4::load(array, i);
			let incremented = vector + increment_vector;
			incremented.store(array, i);
			i += 4;
		}
	}
		
	fn inplace_real_scale_simd(array: &mut [f32], value: f32)
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let vector = f32x4::load(array, i);
			let scaled = vector.scale_real(value);
			scaled.store(array, i);
			i += 4;
		}
	}
	
	fn inplace_complex_scale_simd(array: &mut [f32], value: Complex32)
	{
		let mut i = 0;
		while i < array.len()
		{
			let vector = f32x4::load(array, i);
			let result = vector.scale_complex(value);
			result.store(array, i);
			i += 4;
		}
	}
		
	fn abs_real_par<T>(array: &mut [T])
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].abs();
			i += 1;
		}
	}
	
	fn real_sqrt_par<T>(array: &mut [T])
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].sqrt();
			i += 1;
		}
	}
	
	fn complex_abs_simd(array: &[f32], target: &mut [f32])
	{
		let mut i = 0;
		let mut j = 0;
		while i < array.len()
		{ 
			let vector = f32x4::load(array, i);
			let result = vector.complex_abs();
			result.store_half(target, j);
			i += 4;
			j += 2;
		}
	}
	
	fn complex_abs_squared_simd(array: &[f32], target: &mut [f32])
	{
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
	}
	
	fn complex_conj_par(array: &mut [f32])
	{
		let mut i = 1;
		while i < array.len() {
			array[i] = -array[i];
			i += 2;
		}
	}
	
	fn zero_interleave_complex(mut self) -> Self
	{
		{
			let new_len = 2 * self.len();
			self.reallocate(new_len);
			let data_length = new_len;
			let mut target = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(&source,  &mut target, data_length, 4,  DataVector32::zero_interleave_complex_par);
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
			Chunk::execute_original_to_target(&source,  &mut target, data_length, 2,  DataVector32::zero_interleave_real_par);
		}
		self.swap_data_temp()
	}
	
	fn zero_interleave_complex_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn zero_interleave_real_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn swap_data_temp(mut self) -> DataVector32
	{
		let temp = self.temp;
		self.temp = self.data;
		self.data = temp;
		self
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use super::super::general::{
		DataVector,
		DataVectorDomain,
		GenericVectorOperations,
		RealVectorOperations,
		ComplexVectorOperations};
	use num::complex::Complex32;

	#[test]
	fn construct_real_time_vector_32_test()
	{
		let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let vector = RealTimeVector32::from_array(&array);
		assert_eq!(vector.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
		assert_eq!(vector.delta(), 1.0);
		assert_eq!(vector.domain(), DataVectorDomain::Time);
	}
	
	#[test]
	fn add_real_one_32_test()
	{
		let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let vector = RealTimeVector32::from_array(&mut data);
		let result = vector.real_offset(1.0);
		let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn add_real_two_32_test()
	{
		// Test also that vector calls are possible
		let data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
		let result = RealTimeVector32::from_array(&data);
		let result = result.real_offset(2.0);
		assert_eq!(result.data, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn add_complex_32_test()
	{
		let data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
		let result = ComplexTimeVector32::from_interleaved(&data);
		let result = result.complex_offset(Complex32::new(1.0, -1.0));
		assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn multiply_real_two_32_test()
	{
		let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let result = RealTimeVector32::from_array(&data);
		let result = result.real_scale(2.0);
		let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn multiply_complex_32_test()
	{
		let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let result = ComplexTimeVector32::from_interleaved(&data);
		let result = result.complex_scale(Complex32::new(2.0, -3.0));
		let expected = [8.0, 1.0, 18.0, -1.0, 28.0, -3.0, 38.0, -5.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn abs_real_32_test()
	{
		let data = [-1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 7.0, -8.0];
		let result = RealTimeVector32::from_array(&data);
		let result = result.real_abs();
		let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn abs_complex_32_test()
	{
		let data = [3.0, 4.0, -3.0, 4.0, 3.0, -4.0, -3.0, -4.0];
		let result = ComplexTimeVector32::from_interleaved(&data);
		let result = result.complex_abs();
		let expected = [5.0, 5.0, 5.0, 5.0];
		assert_eq!(result.data(), expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn abs_complex_squared_32_test()
	{
		let data = [-1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 7.0, -8.0, 9.0, 10.0];
		let result = ComplexTimeVector32::from_interleaved(&data);
		let result = result.complex_abs_squared();
		let expected = [5.0, 25.0, 61.0, 113.0, 181.0];
		assert_eq!(result.data(), expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn indexer_test()
	{
		let data = [1.0, 2.0, 3.0, 4.0];
		let mut result = ComplexTimeVector32::from_interleaved(&data);
		assert_eq!(result[0], 1.0);
		result[0] = 5.0;
		assert_eq!(result[0], 5.0);
		let expected = [5.0, 2.0, 3.0, 4.0];
		assert_eq!(result.data(), expected);
	}
	
	#[test]
	fn add_vector_test()
	{
		let data1 = [1.0, 2.0, 3.0, 4.0];
		let vector1 = ComplexTimeVector32::from_interleaved(&data1);
		let data2 = [5.0, 7.0, 9.0, 11.0];
		let vector2 = ComplexTimeVector32::from_interleaved(&data2);
		let result = vector1.add_vector(&vector2);
		let expected = [6.0, 9.0, 12.0, 15.0];
		assert_eq!(result.data(), expected);
	}
	
	#[test]
	fn multiply_complex_vector_32_test()
	{
		let a = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
		let b = ComplexTimeVector32::from_interleaved(&[2.0, -3.0, -2.0, 3.0, 2.0, -3.0, -2.0, 3.0]);
		let result = a.multiply_vector(&b);
		let expected = [8.0, 1.0, -18.0, 1.0, 28.0, -3.0, -38.0, 5.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn divide_complex_vector_32_test()
	{
		let a = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
		let b = ComplexTimeVector32::from_interleaved(&[-1.0, 0.0, 0.0, 1.0, 2.0, -3.0]);
		let result = a.divide_vector(&b);
		let expected = [-1.0, -2.0, 4.0, -3.0, -8.0/13.0, 27.0/13.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
}