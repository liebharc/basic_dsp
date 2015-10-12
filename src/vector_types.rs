use simd::f32x4;
use simd::x86::sse3::Sse3F32x4;
use simple_parallel::Pool;
use std::slice::ChunksMut;
use num_cpus;

#[derive(PartialEq)]
#[derive(Clone)]
#[derive(Copy)]
pub struct Complex
{
	pub real: f32,
	pub imag: f32
}

impl Complex
{
	pub fn new(real: f32, imag: f32) -> Complex
	{
		return Complex { real: real, imag: imag };
	}
}

pub struct DataBuffer
{
	// Storing the pool saves a little bit of initialization time
	pool: Pool,
	
	// TODO: buffer vectors so that they can be reused
}

impl DataBuffer
{
	#[allow(unused_variables)]
	pub fn new(name: &str) -> DataBuffer
	{
		return DataBuffer { pool: Pool::new(num_cpus::get()) };
	}
}

#[derive(Debug)]
#[derive(PartialEq)]
struct Chunk
{
	pub id: usize,
	pub start: usize,
	pub end: usize,
	pub step_size: usize
}

#[allow(dead_code)]
impl Chunk
{
	#[inline]
	fn perform_parallel_execution(array_length: usize) -> bool
	{
		array_length > 1000000
	}
	
	#[inline]
	fn partition_in_number(array: &mut [f32], array_length: usize, step_size: usize, number_of_chunks: usize) -> ChunksMut<f32>
	{
		let mut chunk_size = array_length / number_of_chunks;
		chunk_size -= chunk_size % step_size;
		if chunk_size > 0
		{
			array[0 .. array_length].chunks_mut(chunk_size)
		}
		else
		{
			array[0 .. array_length].chunks_mut(array_length)
		}	
	}
		
	#[inline]
	fn partition(array: &mut [f32], array_length: usize, step_size: usize) -> ChunksMut<f32>
	{
		Chunk::partition_in_number(array, array_length, step_size, num_cpus::get())
	}
	
	#[inline]
	fn execute<F>(array: & mut [f32], step_size: usize, buffer:  &mut DataBuffer, function: F)
		where F: Fn(&mut [f32])  + Send + 'static + Sync
	{
		let array_length = array.len();
		Chunk::execute_partial(array, array_length, step_size, buffer, function);
	}
	
	#[inline]
	fn execute_partial<F>(array: & mut [f32], array_length: usize, step_size: usize, buffer:  &mut DataBuffer, function: F)
		where F: Fn(&mut [f32]) + Send + 'static + Sync
	{
		if Chunk::perform_parallel_execution(array_length)
		{
			let chunks = Chunk::partition(array, array_length, step_size);
			let ref mut pool = buffer.pool;
			pool.for_(chunks, |chunk|
				{
					function(chunk);
				});
		}
		else
		{
			function(&mut array[0..array_length]);
		}
	}
	
	#[inline]
	fn execute_partial_with_arguments<T,F>(array: & mut [f32], array_length: usize, step_size: usize, buffer:  &mut DataBuffer, function: F, arguments: T)
		where F: Fn(& mut [f32], T) + Send + 'static + Sync, T : Sync + Copy
	{
		if Chunk::perform_parallel_execution(array_length)
		{
			let chunks = Chunk::partition(array, array_length, step_size);
			let ref mut pool = buffer.pool;
			pool.for_(chunks, |chunk|
				{
					function(chunk, arguments);
				});
		}
		else
		{
			function(&mut array[0..array_length], arguments);
		}
	}
}

pub enum DataVectorDomain {
	Time,
    Frequency
}

pub trait DataVector<'a> {
    // The element that this vector stores.
    //type Elem;
	
	fn domain() -> DataVectorDomain;
	
	fn is_complex() -> bool;
	
	fn points(&self) -> usize;
	
	fn allocated_size(&self) -> usize;
	
	fn delta(&self) -> f64;
}

pub struct RealTimeVector32<'a>
{
	data: &'a mut [f32],
	delta: f64
}

impl<'a> RealTimeVector32<'a>
{
	pub fn new<'b>(data: &'b mut [f32]) -> RealTimeVector32<'b>
	{
		RealTimeVector32 { data: data, delta: 1.0 }
	}
	
	fn len(&self) -> usize 
	{
		self.data.len()
    }
	
	pub fn data(&self) -> &[f32]
	{
		self.data
	}
	
	pub fn inplace_complex_offset(&mut self, offset: Complex, buffer: &mut DataBuffer) 
	{
		self.inplace_offset(&[offset.real, offset.imag, offset.real, offset.imag], buffer);
	}
	
	pub fn inplace_real_offset(&mut self, offset: f32, buffer: &mut DataBuffer) 
	{
		self.inplace_offset(&[offset, offset, offset, offset], buffer);
	}
	
	#[inline]
	fn inplace_offset(&mut self, offset: &[f32; 4], buffer: &mut DataBuffer) 
	{
		let increment_vector = f32x4::load(offset, 0); 
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		if Chunk::perform_parallel_execution(vectorization_length)
		{
			Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, RealTimeVector32::inplace_offset_simd, increment_vector);
		}
		else		
		{
			RealTimeVector32::inplace_offset_simd(array, increment_vector);
		}
		
		for i in vectorization_length..data_length
		{
			array[i] = array[i] + offset[i % 2];
		}
	}
		
	#[inline]
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
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, RealTimeVector32::inplace_real_scale_simd, scaling_vector);
		for i in vectorization_length..data_length
		{
			array[i] = array[i] * factor;
		}
	}
		
	#[inline]
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
	
	pub fn inplace_complex_scale(&mut self, factor: Complex, buffer: &mut DataBuffer) 
	{
		let scaling_vector = f32x4::load(&[factor.real, factor.imag, factor.real, factor.imag], 0);
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, RealTimeVector32::inplace_complex_scale_simd, scaling_vector);
		for i in vectorization_length..data_length
		{
			array[i] = array[i] * if i % 2 == 0 { factor.real} else {factor.imag };
		}
	}
	
	#[inline]	
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
		Chunk::execute_partial(&mut array, length, 4, buffer, RealTimeVector32::inplace_abs_real_par);
	}
	
	#[inline]
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
		Chunk::execute_partial(&mut array, vectorization_length, 4, buffer, RealTimeVector32::inplace_complex_abs_simd);
		let mut i = vectorization_length;
		while i + 1 < data_length
		{
			array[i / 2] = (array[i] * array[i] + array[i + 1] * array[i + 1]).sqrt();
			i += 2;
		}
	}
	
	#[inline]
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
		Chunk::execute_partial(&mut array, vectorization_length, 4, buffer, RealTimeVector32::inplace_complex_abs_squared_simd);
		let mut i = vectorization_length;
		while i + 1 < data_length
		{
			array[i / 2] = array[i] * array[i] + array[i + 1] * array[i + 1];
			i += 2;
		}
	}
	
	#[inline]
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

impl<'a> DataVector<'a> for RealTimeVector32<'a>
{
	fn domain() -> DataVectorDomain
	{
		DataVectorDomain::Time
	}
	
	fn is_complex() -> bool
	{
		false
	}
	
	fn points(&self) -> usize
	{
		self.data.len()
	}
	
	fn allocated_size(&self) -> usize
	{
		self.points()
	}
	
	fn delta(&self) -> f64
	{
		self.delta
	}
}

pub struct ComplexTimeVector32<'a>
{
	data: &'a mut [f32],
	delta: f64
}

impl<'a> ComplexTimeVector32<'a>
{
	pub fn new<'b>(data: &'b mut [f32]) -> RealTimeVector32<'b>
	{
		RealTimeVector32 { data: data, delta: 1.0 }
	}
	
	fn len(&self) -> usize 
	{
		self.data.len()
    }
	
	pub fn data(&self) -> &[f32]
	{
		self.data
	}
}

impl<'a> DataVector<'a> for ComplexTimeVector32<'a>
{
	fn domain() -> DataVectorDomain
	{
		DataVectorDomain::Time
	}
	
	fn is_complex() -> bool
	{
		true
	}
	
	fn points(&self) -> usize
	{
		self.allocated_size() / 2
	}
	
	fn allocated_size(&self) -> usize
	{
		self.data.len()
	}
	
	fn delta(&self) -> f64
	{
		self.delta
	}
}

#[test]
fn partition_array_in_tiny_pieces()
{
	let mut array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let chunks = Chunk::partition_in_number(&mut array, 8, 4, 8);
	assert_eq!(chunks.len(), 1);
	for chunk in chunks
	{
		assert_eq!(chunk.len(), 8);
	}
}

#[test]
fn partition_array()
{
	let mut array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let chunks = Chunk::partition_in_number(&mut array, 8, 4, 2);
	assert_eq!(chunks.len(), 2);
	for chunk in chunks
	{
		assert_eq!(chunk.len(), 4);
	}
}

#[test]
fn partition_array_considering_step_size()
{
	let mut array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0];
	let chunks = Chunk::partition_in_number(&mut array, 8, 4, 2);
	assert_eq!(chunks.len(), 2);
	for chunk in chunks
	{
		assert_eq!(chunk.len(), 4);
	}
}

#[test]
fn add_real_one_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = RealTimeVector32::new(&mut data);
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
	let mut result = RealTimeVector32::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_offset(2.0, &mut buffer);
	assert_eq!(result.data, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn add_complex_test()
{
	let mut data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	let mut result = RealTimeVector32::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_offset(Complex::new(1.0, -1.0), &mut buffer);
	assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn multiply_real_two_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = RealTimeVector32::new(&mut data);
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
	let mut result = RealTimeVector32::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_scale(Complex::new(2.0, -3.0), &mut buffer);
	let expected = [8.0, 1.0, 18.0, -1.0, 28.0, -3.0, 38.0, -5.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn abs_real_test()
{
	let mut data = [-1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 7.0, -8.0];
	let mut result = RealTimeVector32::new(&mut data);
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
	let mut result = RealTimeVector32::new(&mut data);
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
	let mut result = RealTimeVector32::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_abs_squared(&mut buffer);
	// The last half is undefined, we will fix this later
	let expected = [5.0, 25.0, 61.0, 113.0, 181.0, 113.0, 7.0, -8.0, 9.0, 10.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}