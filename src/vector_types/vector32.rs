use multicore_support::Chunk;
use super::general::{
	DataVector,
	DataVectorDomain,
	GenericVectorOperations,
	RealVectorOperations,
	TimeDomainOperations,
	FrequencyDomainOperations,
	ComplexVectorOperations};
use simd::f32x4;
use simd_extensions::SimdExtensions;
use num::complex::Complex32;
use num::traits::Float;
use std::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};
use std::mem;
use rustfft::FFT;

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
			Chunk::execute_original_to_target(&other, vectorization_length, 4, &mut array, vectorization_length, 4,  DataVector32::add_vector_simd);
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
			Chunk::execute_original_to_target(&other, vectorization_length, 4, &mut array, vectorization_length, 4,  DataVector32::sub_vector_simd);
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
	
	fn diff(mut self) -> Self
	{
		{
			let data_length = self.len();
			let mut target = &mut self.temp;
			let org = &self.data;
			if self.is_complex {
				self.valid_len -= 2;
				Chunk::execute_original_to_target(&org, data_length, 2, &mut target, data_length, 2, DataVector32::complex_diff_par);
			}
			else {
				self.valid_len -= 1;
				Chunk::execute_original_to_target(&org, data_length, 1, &mut target, data_length, 1, DataVector32::real_diff_par);
			}
		}
		
		self.swap_data_temp()
	}
	
	fn diff_with_start(mut self) -> Self
	{
		{
			let data_length = self.len();
			let mut target = &mut self.temp;
			let org = &self.data;
			if self.is_complex {
				Chunk::execute_original_to_target(&org, data_length, 2, &mut target, data_length, 2, DataVector32::complex_diff_with_start_par);
			}
			else {
				Chunk::execute_original_to_target(&org, data_length, 1, &mut target, data_length, 1, DataVector32::real_diff_with_start_par);
			}
		}
		
		self.swap_data_temp()
	}
	
	fn cum_sum(mut self) -> Self
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
		self
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
	
	fn real_square(mut self) -> Self
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(&mut array, length, 1, DataVector32::real_square_par);
		}
		self
	}
	
	fn real_root(mut self, degree: Self::E) -> Self
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(&mut array, length, 1, DataVector32::real_root_par, degree);
		}
		self
	}
	
	fn real_power(mut self, exponent: Self::E) -> Self
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(&mut array, length, 1, DataVector32::real_power_par, exponent);
		}
		self
	}
	
	fn real_logn(mut self) -> Self
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(&mut array, length, 1, DataVector32::real_logn_par);
		}
		self
	}
	
	fn real_expn(mut self) -> Self
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(&mut array, length, 1, DataVector32::real_expn_par);
		}
		self
	}

	fn real_log_base(mut self, base: Self::E) -> Self
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(&mut array, length, 1, DataVector32::real_log_base_par, base);
		}
		self
	}
	
	fn real_exp_base(mut self, base: Self::E) -> Self
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(&mut array, length, 1, DataVector32::real_exp_base_par, base);
		}
		self
	}
	
	fn to_complex(self) -> DataVector32
	{
		let mut result = self.zero_interleave_real();
		result.is_complex = true;
		result
	}
	
	fn wrap(mut self, divisor: Self::E) -> Self
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(&mut array, length, 1, DataVector32::real_modulo_par, divisor);
		}
		self
	}
	
	fn unwrap(mut self, divisor: Self::E) -> Self
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
		self
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
			let array = &self.data;
			let mut temp = &mut self.temp;
			Chunk::execute_original_to_target(&array, vectorization_length, 4, &mut temp, vectorization_length / 2, 2,  DataVector32::complex_abs_simd);
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
		Chunk::execute_original_to_target(&array, vectorization_length, 4, &mut temp, vectorization_length / 2, 2,  DataVector32::complex_abs_simd);
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
	
	fn to_real(mut self) -> DataVector32
	{
		{
			let len = self.len();
			let mut array = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1,  DataVector32::copy_real_parts_par);
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
			Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1,  DataVector32::copy_imag_parts_par);
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
		Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1,  DataVector32::copy_real_parts_par);
	}
	
	fn get_imag(&self, destination: &mut DataVector32)
	{
		let len = self.len();
		destination.reallocate(len / 2);
		destination.delta = self.delta;
		destination.is_complex = false;
		let mut array = &mut destination.data;
		let source = &self.data;
		Chunk::execute_original_to_target(&source, len, 2, &mut array, len / 2, 1,  DataVector32::copy_imag_parts_par);
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
	
	fn copy_real_parts_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
		let mut i = range.start;
		let mut j = 0;
		while j < target.len()
		{ 
			target[j] = original[i];
			i += 2;
			j += 1;
		}
	}
	
	fn copy_imag_parts_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
		let mut i = range.start + 1;
		let mut j = 0;
		while j < target.len()
		{ 
			target[j] = original[i];
			i += 2;
			j += 1;
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
	
	fn multiply_vector_complex(mut self, factor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &factor.data;
			Chunk::execute_original_to_target(&other, vectorization_length, 4, &mut array, vectorization_length, 4,  DataVector32::mul_vector_complex_simd);
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
			Chunk::execute_original_to_target(&other, vectorization_length, 4, &mut array, vectorization_length, 4,  DataVector32::mul_vector_real_simd);
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
			Chunk::execute_original_to_target(&other, vectorization_length, 4, &mut array, vectorization_length, 4,  DataVector32::divide_vector_complex_simd);
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
			Chunk::execute_original_to_target(&other, vectorization_length, 4, &mut array, vectorization_length, 4,  DataVector32::div_vector_real_simd);
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
	
	fn real_square_par<T>(array: &mut [T])
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i] * array[i];
			i += 1;
		}
	}
	
	fn real_root_par(array: &mut [f32], base: f32)
	{
		let base = 1.0 / base;
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].powf(base);
			i += 1;
		}
	}
	
	fn real_power_par<T>(array: &mut [T], base: T)
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].powf(base);
			i += 1;
		}
	}
	
	fn real_logn_par<T>(array: &mut [T])
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].ln();
			i += 1;
		}
	}
	
	fn real_expn_par<T>(array: &mut [T])
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].exp();
			i += 1;
		}
	}
	
	fn real_log_base_par<T>(array: &mut [T], base: T)
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].log(base);
			i += 1;
		}
	}
	
	fn real_exp_base_par<T>(array: &mut [T], base: T)
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = base.powf(array[i]);
			i += 1;
		}
	}
	
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
			Chunk::execute_original_to_target(&source, data_length, 4, &mut target, data_length, 4,  DataVector32::zero_interleave_complex_par);
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
			Chunk::execute_original_to_target(&source, data_length, 4, &mut target, data_length, 2,  DataVector32::zero_interleave_real_par);
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
	
	fn complex_diff_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn real_diff_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn complex_diff_with_start_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn real_diff_with_start_par(original: &[f32], range: Range<usize>, target: &mut [f32])
	{
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
	}
	
	fn real_modulo_par(array: &mut [f32], value: f32) {
		let mut i = 0;
		while i < array.len() {
			array[i] = array[i] % value;
			i += 1;
		}
	}
	
	fn array_to_complex(array: &[f32]) -> &[Complex32] {
		unsafe { 
			let len = array.len();
			let trans: &[Complex32] = mem::transmute(array);
			&trans[0 .. len / 2]
		}
	}
	
	fn array_to_complex_mut(array: &mut [f32]) -> &mut [Complex32] {
		unsafe { 
			let len = array.len();
			let trans: &mut [Complex32] = mem::transmute(array);
			&mut trans[0 .. len / 2]			
		}
	}
}

impl TimeDomainOperations for DataVector32 {
	type FreqPartner = DataVector32;
	fn plain_fft(mut self) -> DataVector32 {
		{
			let points = self.points();
			let rbw = (points as f32)  / self.delta;
			self.delta = rbw;
			let mut fft = FFT::new(points, false);
			let signal = &self.data;
			let spectrum = &mut self.temp;
			let signal = DataVector32::array_to_complex(signal);
			let spectrum = DataVector32::array_to_complex_mut(spectrum);
			fft.process(signal, spectrum);
		}
		self.swap_data_temp()
	}
}

impl TimeDomainOperations for ComplexTimeVector32 {
	type FreqPartner = ComplexFreqVector32;
	fn plain_fft(self) -> ComplexFreqVector32 {
		ComplexFreqVector32::from_gen(self.to_gen().plain_fft())
	}
}

impl FrequencyDomainOperations for DataVector32 {
	type TimePartner = DataVector32;
	fn plain_ifft(mut self) -> DataVector32 {
		{
			let points = self.points();
			let mut fft = FFT::new(points, true);
			let delta = (points as f32)  / self.delta;
			self.delta = delta;
			let signal = &self.data;
			let spectrum = &mut self.temp;
			let signal = DataVector32::array_to_complex(signal);
			let spectrum = DataVector32::array_to_complex_mut(spectrum);
			fft.process(signal, spectrum);
		}
		self.swap_data_temp()
	}
}

impl FrequencyDomainOperations for ComplexFreqVector32 {
	type TimePartner = ComplexTimeVector32;
	fn plain_ifft(self) -> ComplexTimeVector32 {
		ComplexTimeVector32::from_gen(self.to_gen().plain_ifft())
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
	
	#[test]
	fn array_to_complex_test()
	{
		let a = [1.0; 10];
		let c = DataVector32::array_to_complex(&a);
		let expected = [Complex32::new(1.0, 1.0); 5];
		assert_eq!(&expected, c);
	}
	
	#[test]
	fn array_to_complex_mut_test()
	{
		let mut a = [1.0; 10];
		let c = DataVector32::array_to_complex_mut(&mut a);
		let expected = [Complex32::new(1.0, 1.0); 5];
		assert_eq!(&expected, c);
	}
}