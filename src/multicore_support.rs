use num_cpus;
use std::slice::ChunksMut;
use simple_parallel::Pool;

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

pub struct Chunk;
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
	pub fn execute<F>(array: & mut [f32], step_size: usize, buffer:  &mut DataBuffer, function: F)
		where F: Fn(&mut [f32])  + Send + 'static + Sync
	{
		let array_length = array.len();
		Chunk::execute_partial(array, array_length, step_size, buffer, function);
	}
	
	#[inline]
	pub fn execute_partial<F>(array: & mut [f32], array_length: usize, step_size: usize, buffer:  &mut DataBuffer, function: F)
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
	pub fn execute_partial_with_arguments<T,F>(array: & mut [f32], array_length: usize, step_size: usize, buffer:  &mut DataBuffer, function: F, arguments: T)
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
