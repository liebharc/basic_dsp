use multicore_support::{Chunk,DataBuffer};
use simd::f32x4;
use simd::x86::sse3::Sse3F32x4;
use simd_extensions::SimdExtensions32;
use num::complex::{Complex32,Complex64};
use std::mem;
use num::traits::Float;

pub trait DataVector
{
	type E;
	fn data(&mut self, &buffer: &mut DataBuffer) -> &[Self::E];
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

#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
#[allow(dead_code)]
enum Operation32
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

macro_rules! define_vector_struct {
    (struct $name:ident,$data_type:ident) => {
		pub struct $name<'a>
		{
			data: &'a mut [$data_type],
			delta: $data_type,
			domain: DataVectorDomain,
			is_complex: bool,
			operations: Vec<Operation32>
		}
		
		#[inline]
		impl<'a> DataVector for $name<'a>
		{
			type E = $data_type;
			
			fn len(&self) -> usize 
			{
				self.data.len()
			}
			
			fn data(&mut self, buffer: &mut DataBuffer) -> &[$data_type]
			{
				// self.perfom_pending_operations(buffer);
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
				  is_complex: false,
				  operations: Vec::with_capacity(10)
				}
			}
			
			pub fn from_array_with_delta<'b>(data: &'b mut [<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  operations: Vec::with_capacity(10)
				}
			}
		}
	 }
}

macro_rules! define_generic_operations_forward {
    (from: $name:ident, to: $gen_type:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			fn perfom_pending_operations(&mut self, buffer: &mut DataBuffer) -> $name
			{
				$name::from_gen(self.to_gen().perfom_pending_operations(buffer))
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
			pub fn inplace_real_offset(&mut self, offset: <$name as DataVector>::E, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_real_offset(offset, buffer);
			}
			
			pub fn inplace_real_scale(&mut self, factor: <$name as DataVector>::E, buffer: &mut DataBuffer) 
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
			
			fn from_gen(other: $gen_type) -> $name
			{
				unsafe { mem::transmute(other) }
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
				  is_complex: true,
				  operations: Vec::with_capacity(10)
				}
			}
			
			pub fn from_interleaved_with_delta<'b>(data: &'b mut [<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  operations: Vec::with_capacity(10)
				}
			}
		} 
	 }
}

macro_rules! define_complex_operations_forward {
    (from: $name:ident, to: $gen_type:ident, complex: $complex_type:ident)
	 =>
	 { 
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn inplace_complex_offset(&mut self, offset: $complex_type, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_complex_offset(offset, buffer);
			}		
			
			// We are keeping this since scaling with a real number should be faster
			pub fn inplace_real_scale(&mut self, factor: <$name as DataVector>::E, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_real_scale(factor, buffer);
			}
				
			pub fn inplace_complex_scale(&mut self, factor: $complex_type, buffer: &mut DataBuffer) 
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
			
			fn from_gen(other: $gen_type) -> $name
			{
				unsafe { mem::transmute(other) }
			}
		} 
	 }
}

define_vector_struct!(struct DataVector32, f32);
define_real_basic_struct_members!(impl DataVector32, DataVectorDomain::Time);
define_complex_basic_struct_members!(impl DataVector32, DataVectorDomain::Frequency);

define_vector_struct!(struct RealTimeVector32, f32);
define_real_basic_struct_members!(impl RealTimeVector32, DataVectorDomain::Time);
define_generic_operations_forward!(from: RealTimeVector32, to: DataVector32);
define_real_operations_forward!(from: RealTimeVector32, to: DataVector32);

define_vector_struct!(struct RealFreqVector32, f32);
define_real_basic_struct_members!(impl RealFreqVector32, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: RealFreqVector32, to: DataVector32);
define_real_operations_forward!(from: RealFreqVector32, to: DataVector32);

define_vector_struct!(struct ComplexTimeVector32, f32);
define_complex_basic_struct_members!(impl ComplexTimeVector32, DataVectorDomain::Time);
define_generic_operations_forward!(from: ComplexTimeVector32, to: DataVector32);
define_complex_operations_forward!(from: ComplexTimeVector32, to: DataVector32, complex: Complex32);

define_vector_struct!(struct ComplexFreqVector32, f32);
define_complex_basic_struct_members!(impl ComplexFreqVector32, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: ComplexFreqVector32, to: DataVector32);
define_complex_operations_forward!(from: ComplexFreqVector32, to: DataVector32, complex: Complex32);

define_vector_struct!(struct DataVector64, f64);
define_real_basic_struct_members!(impl DataVector64, DataVectorDomain::Time);
define_complex_basic_struct_members!(impl DataVector64, DataVectorDomain::Frequency);

define_vector_struct!(struct RealTimeVector64, f64);
define_real_basic_struct_members!(impl RealTimeVector64, DataVectorDomain::Time);
define_generic_operations_forward!(from: RealTimeVector64, to: DataVector64);
define_real_operations_forward!(from: RealTimeVector64, to: DataVector64);

define_vector_struct!(struct RealFreqVector64, f64);
define_real_basic_struct_members!(impl RealFreqVector64, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: RealFreqVector64, to: DataVector64);
define_real_operations_forward!(from: RealFreqVector64, to: DataVector64);

define_vector_struct!(struct ComplexTimeVector64, f64);
define_complex_basic_struct_members!(impl ComplexTimeVector64, DataVectorDomain::Time);
define_generic_operations_forward!(from: ComplexTimeVector64, to: DataVector64);
define_complex_operations_forward!(from: ComplexTimeVector64, to: DataVector64, complex: Complex64);

define_vector_struct!(struct ComplexFreqVector64, f64);
define_complex_basic_struct_members!(impl ComplexFreqVector64, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: ComplexFreqVector64, to: DataVector64);
define_complex_operations_forward!(from: ComplexFreqVector64, to: DataVector64, complex: Complex64);

const DEFAULT_GRANUALRITY: usize = 4;

#[inline]
impl<'a> DataVector32<'a>
{
	fn perfom_pending_operations(&mut self, buffer: &mut DataBuffer)
		-> DataVector32
	{
		if self.operations.len() == 0
		{
			return DataVector32 { data: self.data, operations: self.operations.clone(), .. *self };
		}
		
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, DEFAULT_GRANUALRITY, buffer, DataVector32::perform_operations_par, &self.operations);
		DataVector32 { data: array, operations: self.operations.clone(), .. *self }
	}
	
	fn perform_operations_par(array: &mut [f32], operations: &Vec<Operation32>)
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
						let increment_vector = f32x4::load(&[value.re, value.im, value.re, value.im], 0); 
						vector = vector + increment_vector;
					}
					/*Operation32::AddVector(value) =>
					{
						// TODO
					}*/
					Operation32::MultiplyReal(value) =>
					{
						let scale_vector = f32x4::splat(value); 
						vector = vector * scale_vector;
					}
					Operation32::MultiplyComplex(value) =>
					{
						let scaling_real = f32x4::splat(value.re);
						let scaling_imag = f32x4::splat(value.im);
						let parallel = scaling_real * vector;
						// There should be a shufps operation which shuffles the vector
						let shuffled = f32x4::new(vector.extract(1), vector.extract(0), vector.extract(3), vector.extract(2)); 
						let cross = scaling_imag * shuffled;
						vector = parallel.addsub(cross);
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
						let squared = vector * vector;
						let squared_sum = squared.hadd(squared);
						vector = squared_sum;
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
	}	pub fn complex_offset(&mut self, offset: Complex32, buffer: &mut DataBuffer)
		-> DataVector32 
	{
		self.inplace_offset(&[offset.re, offset.im, offset.re, offset.im], buffer);
		DataVector32 { data: self.data, operations: self.operations.clone(), .. *self }
	}

	pub fn inplace_complex_offset(&mut self, offset: Complex32, buffer: &mut DataBuffer) 
	{
		self.operations.push(Operation32::AddComplex(offset));
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
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, DEFAULT_GRANUALRITY, buffer, DataVector32::inplace_offset_simd, increment_vector);
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
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, DEFAULT_GRANUALRITY, buffer, DataVector32::inplace_complex_scale_simd, scaling_vector);
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
	
	pub fn multi_operation_example(&mut self, buffer: &mut DataBuffer)
	{
		self.operations.push(Operation32::MultiplyComplex(Complex32::new(-1.0, 1.0)));
		self.perfom_pending_operations(buffer);
	}
	
	pub fn inplace_real_abs(&mut self, buffer: &mut DataBuffer)
	{
		let mut array = &mut self.data;
		let length = array.len();
		Chunk::execute_partial(&mut array, length, 1, buffer, DataVector32::inplace_abs_real_par);
	}
	
	fn inplace_abs_real_par<T>(array: &mut [T])
		where T : Float
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i].abs();
			i += 1;
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

#[inline]
impl<'a> DataVector64<'a>
{
	fn perfom_pending_operations(&mut self, buffer: &mut DataBuffer)
		-> DataVector64
	{
		// TODO implement
		DataVector64 { data: self.data, operations: self.operations.clone(), .. *self }
	}

	pub fn inplace_complex_offset(&mut self, offset: Complex64, buffer: &mut DataBuffer) 
	{
		self.inplace_offset(&[offset.re, offset.im], buffer);
	}
	
	pub fn inplace_real_offset(&mut self, offset: f64, buffer: &mut DataBuffer) 
	{
		self.inplace_offset(&[offset, offset], buffer);
	}
	
	fn inplace_offset(&mut self, offset: &[f64; 2], buffer: &mut DataBuffer) 
	{
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, data_length, 1, buffer, DataVector64::inplace_offset_parallel, offset);
	}
		
	fn inplace_offset_parallel(array: &mut [f64], increment_vector: &[f64;2])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			array[i] = array[i] + increment_vector[i % 2];
			i += 1;
		}
	}

	pub fn inplace_real_scale(&mut self, factor: f64, buffer: &mut DataBuffer) 
	{
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, data_length, 1, buffer, DataVector64::inplace_real_scale_par, factor);
	}
		
	fn inplace_real_scale_par(array: &mut [f64], factor: f64)
	{
		let mut i = 0;
		while i < array.len()
		{ 
			array[i] = array[i] * factor;
			i += 1;
		}
	}
	
	pub fn inplace_complex_scale(&mut self, factor: Complex64, buffer: &mut DataBuffer) 
	{
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, data_length, 1, buffer, DataVector64::inplace_complex_scale_par, factor);
	}
	
	fn inplace_complex_scale_par(array: &mut [f64], factor: Complex64)
	{
		let mut i = 0;
		while i < array.len()
		{
			let real = array[i];
			let imag = array[i + 1];
			array[i] = real * factor.re - imag * factor.im;
			array[i + 1] = real * factor.im + imag * factor.re;
			i += 2;
		}
	}
		
	pub fn inplace_real_abs(&mut self, buffer: &mut DataBuffer)
	{
		let mut array = &mut self.data;
		let length = array.len();
		Chunk::execute_partial(&mut array, length, 1, buffer, DataVector32::inplace_abs_real_par);
	}
	
	pub fn inplace_complex_abs(&mut self, buffer: &mut DataBuffer)
	{
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial(&mut array, data_length, 1, buffer, DataVector64::inplace_complex_abs_par);
	}
	
	fn inplace_complex_abs_par(array: &mut [f64])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let real = array[i];
			let imag = array[i + 1];
			array[i / 2] = (real * real + imag * imag).sqrt();
			i += 2;
		}
	}
	
	pub fn inplace_complex_abs_squared(&mut self, buffer: &mut DataBuffer)
	{
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial(&mut array, data_length, 1, buffer, DataVector64::inplace_complex_abs_squared_par);
	}
	
	fn inplace_complex_abs_squared_par(array: &mut [f64])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let real = array[i];
			let imag = array[i + 1];
			array[i / 2] = real * real + imag * imag;
			i += 2;
		}
	}
}

#[test]
fn new_syntax()
{
	let mut data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	let mut vector = DataVector32::from_interleaved(&mut data);
	let mut buffer = DataBuffer::new("test");
	let result = vector.complex_offset(Complex32::new(1.0, -1.0), &mut buffer);
	assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn construct_real_time_vector_32_test()
{
	let mut array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let vector = RealTimeVector32::from_array(&mut array);
	assert_eq!(vector.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
	assert_eq!(vector.delta(), 1.0);
	assert_eq!(vector.domain(), DataVectorDomain::Time);
}

#[test]
fn add_real_one_32_test()
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
fn add_real_two_32_test()
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
fn add_complex_32_test()
{
	let mut data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	let mut result = ComplexTimeVector32::from_interleaved(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_offset(Complex32::new(1.0, -1.0), &mut buffer);
	assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn multiply_real_two_32_test()
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
fn multiply_complex_32_test()
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
fn abs_real_32_test()
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
fn abs_complex_32_test()
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
fn abs_complex_squared_32_test()
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

#[test]
fn add_real_one_64_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector64::from_array(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_offset(1.0, &mut buffer);
	let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn multiply_real_two_64_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector64::from_array(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_scale(2.0, &mut buffer);
	let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn multiply_complex_64_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector64::from_interleaved(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_scale(Complex64::new(2.0, -3.0), &mut buffer);
	let expected = [8.0, 1.0, 18.0, -1.0, 28.0, -3.0, 38.0, -5.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}