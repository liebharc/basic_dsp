use num_cpus;
use std::slice::ChunksMut;
use num::traits::Float;
use simple_parallel::Pool;

pub struct Chunk;
#[allow(dead_code)]
impl Chunk
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
			static_pool
		}
	}

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
		array[0 .. array_length].chunks_mut(chunk_size)
	}
		
	#[inline]
	fn partition<T>(array: &mut [T], array_length: usize, step_size: usize) -> ChunksMut<T>
		where T : Float + Copy + Clone + Send
	{
		Chunk::partition_in_number(array, array_length, step_size, num_cpus::get())
	}
	
	#[inline]
	pub fn execute<F, T>(array: &mut [T], step_size: usize, function: F)
		where F: Fn(&mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send
	{
		let array_length = array.len();
		Chunk::execute_partial(array, array_length, step_size, function);
	}
	
	#[inline]
	pub fn execute_partial<F, T>(array: &mut [T], array_length: usize, step_size: usize, function: F)
		where F: Fn(&mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send
	{
		if Chunk::perform_parallel_execution(array_length)
		{
			let chunks = Chunk::partition(array, array_length, step_size);
			let ref mut pool = Chunk::get_static_pool();
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
	pub fn execute_partial_with_arguments<T,S,F>(array: &mut [T], array_length: usize, step_size: usize, function: F, arguments:S)
		where F: Fn(&mut [T], S) + 'static + Sync, 
			  T: Float + Copy + Clone + Send,
			  S: Sync + Copy
	{
		if Chunk::perform_parallel_execution(array_length)
		{
			let chunks = Chunk::partition(array, array_length, step_size);
			let ref mut pool = Chunk::get_static_pool();
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
	
	#[inline]
	pub fn execute_with_temp<F, T>(array: &mut [T], step_size: usize, temp: &mut [T], temp_step_size: usize, function: F)
		where F: Fn(&[T], &mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send
	{
		let array_length = array.len();
		let temp_length = temp.len();
		Chunk::execute_partial_with_temp(array, array_length, step_size, temp, temp_length, temp_step_size, function);
	}
	
	#[inline]
	pub fn execute_partial_with_temp<F, T>(array: &mut [T], array_length: usize, step_size: usize, temp: &mut [T], temp_length: usize, temp_step_size: usize, function: F)
		where F: Fn(&[T], &mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send
	{
		if Chunk::perform_parallel_execution(array_length)
		{
			let chunks = Chunk::partition(array, array_length, step_size);
			let temps = Chunk::partition(temp, temp_length, temp_step_size);
			let ref mut pool = Chunk::get_static_pool();
			pool.for_(chunks.zip(temps), |chunk|
				{
					function(chunk.0, chunk.1);
				});
		}
		else
		{
			function(&array[0..array_length], &mut temp[0..temp_length]);
		}
	}
	
	#[inline]
	pub fn execute_partial_with_temp_and_arguments<T,S,F>(array: &mut [T], array_length: usize, step_size: usize, temp: &mut [T], temp_length: usize, temp_step_size: usize, function: F, arguments:S)
		where F: Fn(&[T], &mut [T], S) + 'static + Sync, 
			  T: Float + Copy + Clone + Send,
			  S: Sync + Copy
	{
		if Chunk::perform_parallel_execution(array_length)
		{
			let chunks = Chunk::partition(array, array_length, step_size);
			let temps = Chunk::partition(temp, temp_length, temp_step_size);
			let ref mut pool = Chunk::get_static_pool();
			pool.for_(chunks.zip(temps), |chunk|
				{
					function(chunk.0, chunk.1, arguments);
				});
		}
		else
		{
			function(&mut array[0..array_length], temp, arguments);
		}
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
