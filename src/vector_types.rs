use simd::f32x4;
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
	fn perform_parallel_execution(array_length: usize) -> bool
	{
		array_length > 1000000
	}
	
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
		
	fn partition(array: &mut [f32], array_length: usize, step_size: usize) -> ChunksMut<f32>
	{
		Chunk::partition_in_number(array, array_length, step_size, num_cpus::get())
	}
	
	fn execute<F>(array: & mut [f32], step_size: usize, buffer:  &mut DataBuffer, function: F)
		where F: Fn(&mut [f32])  + Send + 'static + Sync
	{
		let array_length = array.len();
		Chunk::execute_partial(array, array_length, step_size, buffer, function);
	}
	
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

pub struct DataVector<'a>
{
	pub data: &'a mut [f32],
	length: usize,
	delta: f64
}

#[allow(unused_variables)]
impl<'a> DataVector<'a>
{
	pub fn new<'b>(data: &'b mut [f32]) -> DataVector<'b>
	{
		let length = data.len();
		DataVector { data: data, length: length, delta: 1.0 }
	}
	
	pub fn len(&self) -> usize 
	{
		self.length
    }
	
	pub fn delta(&self) -> f64
	{
		self.delta
	}
	
	pub fn inplace_complex_offset(&mut self, offset: Complex, buffer: &mut DataBuffer) 
	{
		self.inplace_offset(&[offset.real, offset.imag, offset.real, offset.imag], buffer);
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
		if Chunk::perform_parallel_execution(vectorization_length)
		{
			Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, DataVector::inplace_offset_simd, increment_vector);
		}
		else		
		{
			DataVector::inplace_offset_simd(array, increment_vector);
		}
		
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
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, DataVector::inplace_real_scale_simd, scaling_vector);
				
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
	
	pub fn inplace_complex_scale(&mut self, factor: Complex, buffer: &mut DataBuffer) 
	{
		let scaling_vector = f32x4::load(&[factor.real, factor.imag, factor.real, factor.imag], 0);
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, vectorization_length, 4, buffer, DataVector::inplace_complex_scale_simd, scaling_vector);
		
		for i in vectorization_length..data_length
		{
			array[i] = array[i] * if i % 2 == 0 { factor.real} else {factor.imag };
		}
	}
		
	fn inplace_complex_scale_simd(array: &mut [f32], scaling_vector: f32x4)
	{
		let mut i = 0;
		let shuffled = f32x4::new(scaling_vector.extract(1), scaling_vector.extract(0), scaling_vector.extract(3), scaling_vector.extract(2));
		let mut temp = [0.0; 4];
		while i < array.len()
		{ 
			let vector = f32x4::load(array, i);
			let parallel = vector * scaling_vector;
			let cross = vector * shuffled;
			parallel.store(&mut temp, 0);
			array[i] = temp[0] - temp[1];
			array[i + 2] = temp[2] - temp[3];
			cross.store(&mut temp, 0);
			array[i + 1] = temp[0] + temp[1];
			array[i + 3] = temp[2] + temp[3];
			i += 4;
		}
	
		/*let mut i = 0;
		while i < array.len()
		{ 
			let real = array[i];
			let imag = array[i + 1];
			array[i] = real * factor.real - imag * factor.imag;
			array[i + 1] = real * factor.imag + imag * factor.real;
			i += 2;
		}*/
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
	let mut result = DataVector::new(&mut data);
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
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_offset(2.0, &mut buffer);
	assert_eq!(result.data, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn add_complex_test()
{
	let mut data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_offset(Complex::new(1.0, -1.0), &mut buffer);
	assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
}

#[test]
fn multiply_real_two_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector::new(&mut data);
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
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_scale(Complex::new(2.0, -3.0), &mut buffer);
	let expected = [8.0, 1.0, 18.0, -1.0, 28.0, -3.0, 38.0, -5.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}