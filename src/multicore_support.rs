use num_cpus;
use std::slice::ChunksMut;
use simple_parallel::Pool;
use num::traits::Float;

pub struct DataBuffer
{
	// Storing the pool saves a little bit of initialization time
	pool: &'static mut Pool,
	
	// TODO: buffer vectors so that they can be reused
}

impl DataBuffer
{
	fn get_static_pool() -> &'static mut Pool
	{
		use std::sync::{Once, ONCE_INIT};
		use std::mem::transmute;
		unsafe
		{
			static mut pool: *mut Pool = 0 as *mut Pool;
			static mut ONCE: Once = ONCE_INIT;
			ONCE.call_once(||
			{
				pool = transmute::<Box<Pool>, *mut Pool>(box Pool::new(num_cpus::get()));
			});
			
			let mut static_pool = &mut *pool;
			//Pool::new(num_cpus::get())
			static_pool
		}
	}

	#[allow(unused_variables)]
	pub fn new(name: &str) -> DataBuffer
	{
		let pool = DataBuffer::get_static_pool();
		return DataBuffer { pool: pool };
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
	fn partition_in_number<T>(array: &mut [T], array_length: usize, step_size: usize, number_of_chunks: usize) -> ChunksMut<T>
		where T : Float + Copy + Clone + Send
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
	fn partition<T>(array: &mut [T], array_length: usize, step_size: usize) -> ChunksMut<T>
		where T : Float + Copy + Clone + Send
	{
		Chunk::partition_in_number(array, array_length, step_size, num_cpus::get())
	}
	
	#[inline]
	pub fn execute<F, T>(array: & mut [T], step_size: usize, buffer:  &mut DataBuffer, function: F)
		where F: Fn(&mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send
	{
		let array_length = array.len();
		Chunk::execute_partial(array, array_length, step_size, buffer, function);
	}
	
	#[inline]
	pub fn execute_partial<F, T>(array: & mut [T], array_length: usize, step_size: usize, buffer:  &mut DataBuffer, function: F)
		where F: Fn(&mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send
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
	pub fn execute_partial_with_arguments<T,S,F>(array: & mut [T], array_length: usize, step_size: usize, buffer:  &mut DataBuffer, function: F, arguments:S)
		where F: Fn(& mut [T], S) + 'static + Sync, 
			  T: Float + Copy + Clone + Send,
			  S: Sync + Copy
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
