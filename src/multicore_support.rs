use num_cpus;
use std::slice::ChunksMut;
use num::traits::Float;
use simple_parallel::Pool;
use std::ops::Range;

pub struct Chunk;
#[allow(dead_code)]
impl Chunk
{
	fn get_static_pool() -> &'static mut Pool
	{
		use std::mem::transmute;
		use std::sync::{Once, ONCE_INIT};
		unsafe
		{
			static mut pool: *mut Pool = 0 as *mut Pool;
			static mut ONCE: Once = ONCE_INIT;
			ONCE.call_once(||
			{
				pool = transmute::<Box<Pool>, *mut Pool>(box Pool::new(num_cpus::get() * 2));
			});
			
			let mut static_pool = &mut *pool;
			static_pool
		}
	}
	
	/*fn get_static_pool() -> Pool
	{
		Pool::new(num_cpus::get())
	}*/

	#[inline]
	fn perform_parallel_execution(array_length: usize) -> bool
	{
		array_length > 1000000
	}
	
	#[inline]
	fn partition_in_number<T>(array: &mut [T], array_length: usize, step_size: usize, number_of_chunks: usize) -> ChunksMut<T>
		where T : Float + Copy + Clone + Send
	{
		let chunk_size = Chunk::calc_chunk_size(array_length, step_size, number_of_chunks);
		array[0 .. array_length].chunks_mut(chunk_size)
	}
	
	#[inline]
	fn calc_chunk_size(array_length: usize, step_size: usize, number_of_chunks: usize) -> usize
	{
		let mut chunk_size = (array_length as f64/ number_of_chunks as f64).ceil() as usize;
		let remainder = chunk_size % step_size;
		if remainder > 0
		{
			chunk_size += step_size - chunk_size % step_size;
		}
		
		chunk_size
	}
	
	#[inline]
	fn partition_in_ranges(array_length: usize, step_size: usize, number_of_chunks: usize) -> Vec<Range<usize>>
	{
		let chunk_size = Chunk::calc_chunk_size(array_length, step_size, number_of_chunks);
		let mut ranges = Vec::with_capacity(number_of_chunks);
		let mut sum = 0;
		for i in 0..number_of_chunks {
			let new_sum = if i < number_of_chunks - 1 { sum + chunk_size } else { array_length };
			ranges.push(Range { start: sum, end: new_sum - 1 });
			sum = new_sum;
		} 
		
		ranges
	}
		
	#[inline]
	fn partition<T>(array: &mut [T], array_length: usize, step_size: usize) -> ChunksMut<T>
		where T : Float + Copy + Clone + Send + Sync
	{
		Chunk::partition_in_number(array, array_length, step_size, num_cpus::get() * 2)
	}
	
	#[inline]
	pub fn execute<F, T>(array: &mut [T], step_size: usize, function: F)
		where F: Fn(&mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send + Sync
	{
		let array_length = array.len();
		Chunk::execute_partial(array, array_length, step_size, function);
	}
	
	#[inline]
	pub fn execute_partial<F, T>(array: &mut [T], array_length: usize, step_size: usize, function: F)
		where F: Fn(&mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send + Sync
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
	pub fn execute_partial_with_arguments<T,S,F>(array: &mut [T], array_length: usize, step_size: usize, arguments:S, function: F)
		where F: Fn(&mut [T], S) + 'static + Sync, 
			  T: Float + Copy + Clone + Send + Sync,
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
			  T : Float + Copy + Clone + Send + Sync
	{
		let array_length = array.len();
		let temp_length = temp.len();
		Chunk::execute_partial_with_temp(array, array_length, step_size, temp, temp_length, temp_step_size, function);
	}
	
	#[inline]
	pub fn execute_partial_with_temp<F, T>(array: &mut [T], array_length: usize, step_size: usize, temp: &mut [T], temp_length: usize, temp_step_size: usize, function: F)
		where F: Fn(&[T], &mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send + Sync
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
	pub fn execute_partial_with_temp_and_arguments<T,S,F>(array: &mut [T], array_length: usize, step_size: usize, temp: &mut [T], temp_length: usize, temp_step_size: usize, arguments:S, function: F)
		where F: Fn(&[T], &mut [T], S) + 'static + Sync, 
			  T: Float + Copy + Clone + Send + Sync,
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
	
	#[inline]
	pub fn execute_original_to_target<F, T>(original: &[T], original_length: usize, original_step: usize, target: &mut [T], target_length: usize, target_step: usize, function: F)
		where F: Fn(&[T], Range<usize>, &mut [T]) + 'static + Sync,
			  T : Float + Copy + Clone + Send + Sync
	{
		if Chunk::perform_parallel_execution(target_length)
		{
			let chunks = Chunk::partition(target, target_length, target_step);
			let ranges = Chunk::partition_in_ranges(original_length, original_step, chunks.len());
			let ref mut pool = Chunk::get_static_pool();
			pool.for_(chunks.zip(ranges), |chunk|
				{
					function(original, chunk.1, chunk.0);
				});
		}
		else
		{
			function(original, Range { start: 0, end: original_length }, &mut target[0..target_length]);
		}
	}
	
	#[inline]
	pub fn execute_original_to_target_with_arguments<T,S,F>(original: &[T], original_length: usize, original_step: usize, target: &mut [T], target_length: usize, target_step: usize, arguments:S, function: F)
		where F: Fn(&[T], Range<usize>, &mut [T], S) + 'static + Sync,
			  T : Float + Copy + Clone + Send + Sync,
			  S: Sync + Copy
	{
		if Chunk::perform_parallel_execution(target_length)
		{
			let chunks = Chunk::partition(target, target_length, target_step);
			let ranges = Chunk::partition_in_ranges(original_length, original_step, chunks.len());
			let ref mut pool = Chunk::get_static_pool();
			pool.for_(chunks.zip(ranges), |chunk|
				{
					function(original, chunk.1, chunk.0, arguments);
				});
		}
		else
		{
			function(original, Range { start: 0, end: original_length }, &mut target[0..target_length], arguments);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::ops::Range;
	
	#[test]
	fn partition_array()
	{
		let mut array = [0.0; 256];
		let chunks = Chunk::partition_in_number(&mut array, 256, 4, 2);
		assert_eq!(chunks.len(), 2);
		for chunk in chunks
		{
			assert_eq!(chunk.len(), 128);
		}
	}
	
	#[test]
	fn partition_array_8_cores()
	{
		let mut array = [0.0; 1023];
		let chunks = Chunk::partition_in_number(&mut array, 1023, 4, 8);
		assert_eq!(chunks.len(), 8);
		let mut i = 0;
		for chunk in chunks
		{
			let expected = if i >= 7 { 127 } else { 128 };
			assert_eq!(chunk.len(), expected);
			i += 1;
		}
	}
	
	#[test]
	fn partitionin_ranges()
	{
		let ranges = Chunk::partition_in_ranges(1023, 4, 2);
		assert_eq!(ranges.len(), 2);
		assert_eq!(ranges[0], Range { start: 0, end: 511 });
		assert_eq!(ranges[1], Range { start: 512, end: 1022 });
	}
}