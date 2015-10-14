use multicore_support::{Chunk,DataBuffer};
use simd::f32x4;
use simd::x86::sse3::Sse3F32x4;
use num::complex::Complex32;
use std::mem;
//use std::ops::Add; // TODO { Div, Mul, Neg, Sub};

pub trait DataVector
{
	type E;
	fn data(&self) -> &[Self::E];
	fn delta(&self) -> Self::E;
	fn domain(&self) -> DataVectorDomain;
	fn is_complex(&self) -> bool;
	fn len(&self) -> usize;
}

#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum DataVectorDomain {
	Time,
    Frequency
}

macro_rules! define_vector_struct {
    (struct $name:ident,$data_type:ident) => {
		pub struct $name<'a>
		{
			data: &'a mut [$data_type],
			delta: $data_type,
			domain: DataVectorDomain,
			is_complex: bool
		}
		
		#[inline]
		impl<'a> DataVector for $name<'a>
		{
			type E = $data_type;
			
			fn len(&self) -> usize 
			{
				self.data.len()
			}
			
			fn data(&self) -> &[$data_type]
			{
				self.data
			}
			
			fn delta(&self) -> $data_type
			{
				self.delta
			}
			
			fn domain(&self) -> DataVectorDomain
			{
				self.domain
			}
			
			fn is_complex(&self) -> bool
			{
				self.is_complex
			}
		}
    }
}

macro_rules! define_real_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn from_array<'b>(data: &'b mut [<$name as DataVector>::E]) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: false
				}
			}
			
			pub fn from_array_with_delta<'b>(data: &'b mut [<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false
				}
			}
		}
	 }
}

macro_rules! define_real_operations_forward {
    (from: $name:ident, to: $gen_type:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn inplace_real_offset(&mut self, offset: f32, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_real_offset(offset, buffer);
			}
			
			pub fn inplace_real_scale(&mut self, factor: f32, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_real_scale(factor, buffer);
			}
					
			pub fn inplace_real_abs(&mut self, buffer: &mut DataBuffer)
			{
				self.to_gen().inplace_real_abs(buffer);
			}
			
			fn to_gen(&mut self) -> &mut $gen_type
			{
				unsafe { mem::transmute(self) }
			}
		}
	 }
}

macro_rules! define_complex_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn from_interleaved<'b>(data: &'b mut [<$name as DataVector>::E]) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: true
				}
			}
			
			pub fn from_interleaved_with_delta<'b>(data: &'b mut [<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true
				}
			}
		} 
	 }
}

macro_rules! define_complex_operations_forward {
    (from: $name:ident, to: $gen_type:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn inplace_complex_offset(&mut self, offset: Complex32, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_complex_offset(offset, buffer);
			}		
			
			// We are keeping this since scaling with a real number should be faster
			pub fn inplace_real_scale(&mut self, factor: f32, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_real_scale(factor, buffer);
			}
			
			pub fn inplace_complex_scale(&mut self, factor: Complex32, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_complex_scale(factor, buffer);
			}
			
			pub fn inplace_complex_abs(&mut self, buffer: &mut DataBuffer)
			{
				self.to_gen().inplace_complex_abs(buffer);
			}
			
			pub fn inplace_complex_abs_squared(&mut self, buffer: &mut DataBuffer)
			{
				self.to_gen().inplace_complex_abs_squared(buffer);
			}
			
			fn to_gen(&mut self) -> &mut $gen_type
			{
				unsafe { mem::transmute(self) }
			}
		} 
	 }
}

define_vector_struct!(struct DataVector32, f32);
define_real_basic_struct_members!(impl DataVector32, DataVectorDomain::Time);
define_complex_basic_struct_members!(impl DataVector32, DataVectorDomain::Frequency);

define_vector_struct!(struct RealTimeVector32, f32);
define_real_basic_struct_members!(impl RealTimeVector32, DataVectorDomain::Time);
define_real_operations_forward!(from: RealTimeVector32, to: DataVector32);

define_vector_struct!(struct RealFreqVector32, f32);
define_real_basic_struct_members!(impl RealFreqVector32, DataVectorDomain::Frequency);
define_real_operations_forward!(from: RealFreqVector32, to: DataVector32);

define_vector_struct!(struct ComplexTimeVector32, f32);
define_complex_basic_struct_members!(impl ComplexTimeVector32, DataVectorDomain::Time);
define_complex_operations_forward!(from: ComplexTimeVector32, to: DataVector32);

define_vector_struct!(struct ComplexFreqVector32, f32);
define_complex_basic_struct_members!(impl ComplexFreqVector32, DataVectorDomain::Frequency);
define_complex_operations_forward!(from: ComplexFreqVector32, to: DataVector32);
/*
define_vector_struct!(struct DataVector64, f64);
define_real_basic_struct_members!(impl DataVector64, DataVectorDomain::Time);
define_complex_basic_struct_members!(impl DataVector64, DataVectorDomain::Frequency);

define_vector_struct!(struct RealTimeVector64, f64);
define_real_basic_struct_members!(impl RealTimeVector64, DataVectorDomain::Time);
define_real_operations_forward!(from: RealTimeVector64, to: DataVector64);

define_vector_struct!(struct RealFreqVector64, f64);
define_real_basic_struct_members!(impl RealFreqVector64, DataVectorDomain::Frequency);
define_real_operations_forward!(from: RealFreqVector64, to: DataVector64);

define_vector_struct!(struct ComplexTimeVector64, f64);
define_complex_basic_struct_members!(impl ComplexTimeVector64, DataVectorDomain::Time);
define_complex_operations_forward!(from: ComplexTimeVector64, to: DataVector64);

define_vector_struct!(struct ComplexFreqVector64, f64);
define_complex_basic_struct_members!(impl ComplexFreqVector64, DataVectorDomain::Frequency);
define_complex_operations_forward!(from: ComplexFreqVector64, to: DataVector64);
*/

#[inline]
impl<'a> DataVector32<'a>
{
	pub fn inplace_complex_offset(&mut self, offset: Complex32, buffer: &mut DataBuffer) 
	{
		self.inplace_offset(&[offset.re, offset.im, offset.re, offset.im], buffer);
	}
	
	pub fn inplace_real_offset(&mut self, offset: f32, buffer: &mut DataBuffer) 
	{
		self.inplace_offset(&[offset, offset, offset, offset], buffer);
	}
	
	fn inplace_offset(&mut self, offset: &[f32; 4], buffer: &mut DataBuffer) 
	{
		let increment_vector = f32x4::load(offset, 0); 
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, DataVector32::inplace_offset_simd, increment_vector);
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
	
	pub fn inplace_real_scale(&mut self, factor: f32, buffer: &mut DataBuffer) 
	{
		let scaling_vector = f32x4::splat(factor); 
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, DataVector32::inplace_real_scale_simd, scaling_vector);
		for i in vectorization_length..data_length
		{
			array[i] = array[i] * factor;
		}
	}
		
	fn inplace_real_scale_simd(array: &mut [f32], scaling_vector: f32x4)
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let vector = f32x4::load(array, i);
			let scaled = vector * scaling_vector;
			scaled.store(array, i);
			i += 4;
		}
	}
	
	pub fn inplace_complex_scale(&mut self, factor: Complex32, buffer: &mut DataBuffer) 
	{
		let scaling_vector = f32x4::load(&[factor.re, factor.im, factor.re, factor.im], 0);
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, DataVector32::inplace_complex_scale_simd, scaling_vector);
		for i in vectorization_length..data_length
		{
			array[i] = array[i] * if i % 2 == 0 { factor.re} else {factor.im };
		}
	}
	
	fn inplace_complex_scale_simd(array: &mut [f32], scaling_vector: f32x4)
	{
		let scaling_real = f32x4::splat(scaling_vector.extract(0));
		let scaling_imag = f32x4::splat(scaling_vector.extract(1));
		let mut i = 0;
		while i < array.len()
		{
			let vector = f32x4::load(array, i);
			let parallel = scaling_real * vector;
			// There should be a shufps operation which shuffles the vector
			let vector = f32x4::new(vector.extract(1), vector.extract(0), vector.extract(3), vector.extract(2)); 
			let cross = scaling_imag * vector;
			let result = parallel.addsub(cross);
			result.store(array, i);
			i += 4;
		}
	}
	
	pub fn inplace_real_abs(&mut self, buffer: &mut DataBuffer)
	{
		let mut array = &mut self.data;
		let length = array.len();
		Chunk::execute_partial(&mut array, length, 4, buffer, DataVector32::inplace_abs_real_par);
	}
	
	fn inplace_abs_real_par(array: &mut [f32])
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].abs();
			i += 1;
		}
	}
	
	pub fn inplace_complex_abs(&mut self, buffer: &mut DataBuffer)
	{
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial(&mut array, vectorization_length, 4, buffer, DataVector32::inplace_complex_abs_simd);
		let mut i = vectorization_length;
		while i + 1 < data_length
		{
			array[i / 2] = (array[i] * array[i] + array[i + 1] * array[i + 1]).sqrt();
			i += 2;
		}
	}
	
	fn inplace_complex_abs_simd(array: &mut [f32])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let vector = f32x4::load(array, i);
			let squared = vector * vector;
			let squared_sum = squared.hadd(squared);
			let result = squared_sum.sqrt();
			result.store(array, i / 2);
			i += 4;
		}
	}
	
	pub fn inplace_complex_abs_squared(&mut self, buffer: &mut DataBuffer)
	{
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial(&mut array, vectorization_length, 4, buffer, DataVector32::inplace_complex_abs_squared_simd);
		let mut i = vectorization_length;
		while i + 1 < data_length
		{
			array[i / 2] = array[i] * array[i] + array[i + 1] * array[i + 1];
			i += 2;
		}
	}
	
	fn inplace_complex_abs_squared_simd(array: &mut [f32])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let vector = f32x4::load(array, i);
			let squared = vector * vector;
			let squared_sum = squared.hadd(squared);
			squared_sum.store(array, i / 2);
			i += 4;
		}
	}
}

#[test]
fn construct_real_time_vector_test()
{
	let mut array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let vector = RealTimeVector32::from_array(&mut array);
	assert_eq!(vector.data(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
	assert_eq!(vector.delta(), 1.0);
	assert_eq!(vector.domain(), DataVectorDomain::Time);
}

#[test]
fn add_real_one_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = RealTimeVector32::from_array(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_offset(1.0, &mut buffer);
	let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn add_real_two_test()
{
	// Test also that vector calls are possible
	let mut data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	let mut result = RealTimeVector32::from_array(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_offset(2.0, &mut buffer);
	assert_eq!(result.data, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn add_complex_test()
{
	let mut data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	let mut result = ComplexTimeVector32::from_interleaved(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_offset(Complex32::new(1.0, -1.0), &mut buffer);
	assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn multiply_real_two_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = RealTimeVector32::from_array(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_scale(2.0, &mut buffer);
	let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn multiply_complex_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = ComplexTimeVector32::from_interleaved(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_scale(Complex32::new(2.0, -3.0), &mut buffer);
	let expected = [8.0, 1.0, 18.0, -1.0, 28.0, -3.0, 38.0, -5.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn abs_real_test()
{
	let mut data = [-1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 7.0, -8.0];
	let mut result = RealTimeVector32::from_array(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_abs(&mut buffer);
	let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn abs_complex_test()
{
	let mut data = [3.0, 4.0, -3.0, 4.0, 3.0, -4.0, -3.0, -4.0];
	let mut result = ComplexTimeVector32::from_interleaved(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_abs(&mut buffer);
	// The last half is undefined, we will fix this later
	let expected = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, -3.0, -4.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn abs_complex_squared_test()
{
	let mut data = [-1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 7.0, -8.0, 9.0, 10.0];
	let mut result = ComplexTimeVector32::from_interleaved(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_abs_squared(&mut buffer);
	// The last half is undefined, we will fix this later
	let expected = [5.0, 25.0, 61.0, 113.0, 181.0, 113.0, 7.0, -8.0, 9.0, 10.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}