use num_cpus;
use std::slice::{Chunks, ChunksMut};
use num::traits::Float;
use simple_parallel::Pool;
use std::ops::Range;
use std::sync::Mutex;
use std::mem;
use super::RealNumber;

/// Indicates how complex an operation is and determines how many cores 
/// will be used since operations with smaller complexity are memory bus bound
/// and not CPU bound
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Complexity {
	Small,
    Medium,
    Large
}

/// Holds parameters which specifiy how multiple cores are used
/// to execute an operation.
#[derive(Debug)] 
#[repr(C)]  
pub struct MultiCoreSettings {
    /// All operations will be limited to not create more threads than specified here
    pub core_limit: usize,
    
    /// Indicates whether the temp arrays of a vector should already be allocated during
    /// construction
    pub early_temp_allocation: bool
    // TODO: Specify and use options such as core/thread limits
}

impl MultiCoreSettings {
    /// Creates multi core settings with default values
    pub fn default() -> MultiCoreSettings {
        // Initialize the pool
        Chunk::init_static_pool();
        // Half because we assume hyper threading and that we will keep a core so busy
        // that hyper threading isn't of any use
        Self::new(num_cpus::get() / 2, true)
    }
    
    /// Creates multi core settings with the given values.
    pub fn new(core_limit: usize, early_temp_allocation: bool) -> MultiCoreSettings {
        // Initialize the pool
        Chunk::init_static_pool();
        MultiCoreSettings {
            core_limit: if core_limit >= 1 { core_limit } else { 1 }, 
            early_temp_allocation: early_temp_allocation
        }
    }
}

impl Clone for MultiCoreSettings {
    fn clone(&self) -> Self {
        MultiCoreSettings {
            core_limit: self.core_limit,
            early_temp_allocation: self.early_temp_allocation
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.core_limit = source.core_limit;
    }
}

static mut POOL: *mut Pool = 0 as *mut Pool;

/// Contains logic which helps to perform an operation
/// in parallel by dividing an array into chunks.
pub struct Chunk;
impl Chunk
{
    fn init_static_pool() {
        use std::mem::transmute;
		use std::sync::{Once, ONCE_INIT};
		unsafe
		{
			static mut ONCE: Once = ONCE_INIT;
			ONCE.call_once(||
			{
				POOL = transmute::<Box<Pool>, *mut Pool>(Box::new(Pool::new(num_cpus::get())));
			});
		}
    }
    
    /// Gives access to the thread pool singleton
	fn get_static_pool() -> &'static mut Pool
	{
		unsafe
		{
			let mut static_pool = &mut *POOL;
			static_pool
		}
	}

    /// Figures out how many threads make use for the an operation with the given complexity on 
    /// an array with the given size. 
    ///
    /// This method tries to balance the expected performance gain vs. CPU utilization since there is in most cases
    /// no point to keep all CPU cores busy only to get 5 to 10% performance gain.
    /// The expected performance gain is roughly estimated based on three factors:
    /// 1. More cores improves the calculation speed according to Amdahl's law (`https://en.wikipedia.org/wiki/Amdahl's_law`)
    /// 2. Spawning a thread consumes time and so the array length must be large enough to so that the expected performance
    ///    gain justifies the effort to spawn a thread/invoke the thread pool. 
    /// 3. The CPU is not the only resource and if one of the other resources is a bottleneck then Amdahl's law won't be applicable.
    ///    The memory bus speed is limited (20GB/s in case of a typical 2015 consumer laptop in the price range of $1000) and
    ///    for operations which only require a few CPU cycles already one or two cores will process data faster than the 
    ///    memory bus is able to provide and to transport back. Using more cores then only creates heat but no performance benefit.
	#[inline]
	fn determine_number_of_chunks(array_length: usize, complexity: Complexity, settings: &MultiCoreSettings) -> usize
	{
        let mut cores = num_cpus::get();
        if cores > settings.core_limit {
            cores = settings.core_limit;
        }
        if complexity == Complexity::Large || cores == 1 {
            cores
        }
        else if complexity == Complexity::Small  {
            if array_length < 500000 {
                1
            }
            else {
                if cores >= 2 {
                    2  
                } else {
                    1
                }
            }
        }
        else { // complexity == medium
            if array_length < 10000 {
                1
            }
            else if array_length < 50000 {
                if cores >= 2 {
                    2  
                } else {
                    1
                }
            }
            else {
                cores
            }
        }
	}
    
    /// Partitions an array into the given number of chunks. It makes sure that all chunks have the same size
    /// and so it will happen that some elements at the end of the array are not part of any chunk. 
	#[inline]
	fn partition<T>(array: &[T], array_length: usize, step_size: usize, number_of_chunks: usize) -> Chunks<T>
		where T : Float + Copy + Clone + Send
	{
		let chunk_size = Chunk::calc_chunk_size(array_length, step_size, number_of_chunks);
		array[0 .. array_length].chunks(chunk_size)
	}
	
    /// Partitions an array into the given number of chunks. It makes sure that all chunks have the same size
    /// and so it will happen that some elements at the end of the array are not part of any chunk. 
	#[inline]
	fn partition_mut<T>(array: &mut [T], array_length: usize, step_size: usize, number_of_chunks: usize) -> ChunksMut<T>
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
	
    /// This function returns the ranges which correspond to the chunks generated by `partition_in_number`. 
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
	
    /// Executes the given function on the first `array_length` elements of the given array in parallel and passes
    /// the argument to all function calls.
	#[inline]
	pub fn execute_partial_with_arguments<T,S,F>(
            complexity: Complexity, 
            settings: &MultiCoreSettings, 
            array: &mut [T], array_length: usize, step_size: usize, 
            arguments:S, ref function: F)
		where F: Fn(&mut [T], S) + 'static + Sync, 
			  T: RealNumber,
			  S: Sync + Copy
	{
		let number_of_chunks = Chunk::determine_number_of_chunks(array_length, complexity, settings);
		if number_of_chunks > 1
		{
			let chunks = Chunk::partition_mut(array, array_length, step_size, number_of_chunks);
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
    
    /// Executes the given function on the all elements of the array in parallel. A result is
    /// returned for each chunk.
	#[inline]
	pub fn get_a_fold_b<F, T, R>(
            complexity: Complexity, 
            settings: &MultiCoreSettings, 
            a: &[T], a_len: usize, a_step: usize, 
            b: &[T], b_len: usize, b_step: usize, 
            function: F) -> Vec<R>
		where F: Fn(&[T], Range<usize>, &[T]) -> R + 'static + Sync,
			  T: Float + Copy + Clone + Send + Sync,
              R: Send
	{
		let number_of_chunks = Chunk::determine_number_of_chunks(a_len, complexity, settings);
		if number_of_chunks > 1
		{
			let chunks = Chunk::partition(b, b_len, b_step, number_of_chunks);
			let ranges = Chunk::partition_in_ranges(a_len, a_step, chunks.len());
			let ref mut pool = Chunk::get_static_pool();
            let result = Vec::with_capacity(chunks.len());
            let stack_array = Mutex::new(result);
            pool.for_(chunks.zip(ranges), |chunk|
                {   
                    let r = function(a, chunk.1, chunk.0);
                    stack_array.lock().unwrap().push(r);
                });
            let mut guard = stack_array.lock().unwrap();
            mem::replace(&mut guard, Vec::new())
		}
		else
		{
			let result = function(a, Range { start: 0, end: a_len }, &b[0..b_len]);
            vec![result]
		}
	}
    
    /// Executes the given function on the all elements of the array in parallel. A result is
    /// returned for each chunk.
	#[inline]
	pub fn get_chunked_results_with_arguments<F, S, T, R>(
            complexity: Complexity, 
            settings: &MultiCoreSettings, 
            a: &[T], a_len: usize, a_step: usize, 
            arguments:S, function: F) -> Vec<R>
		where F: Fn(&[T], Range<usize>, S) -> R + 'static + Sync,
			  T: Float + Copy + Clone + Send + Sync,
              R: Send,
              S: Sync + Copy
	{
		let number_of_chunks = Chunk::determine_number_of_chunks(a_len, complexity, settings);
		if number_of_chunks > 1
		{
			let chunks = Chunk::partition(a, a_len, a_step, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(a_len, a_step, chunks.len());
			let ref mut pool = Chunk::get_static_pool();
            let result = Vec::with_capacity(chunks.len());
            let stack_array = Mutex::new(result);
            pool.for_(chunks.zip(ranges), |chunk|
                {   
                    let r = function(chunk.0, chunk.1, arguments);
                    stack_array.lock().unwrap().push(r);
                });
            let mut guard = stack_array.lock().unwrap();
            mem::replace(&mut guard, Vec::new())
		}
		else
		{
			let result = function(&a[0..a_len], Range { start: 0, end: a_len }, arguments);
            vec![result]
		}
	}
    
    /// Executes the given function on the all elements of the array in parallel and passes
    /// the argument to all function calls.. Results are intended to be stored in the target array.
	#[inline]
	pub fn execute_original_to_target_with_arguments<T,S,F>(
            complexity: Complexity, 
            settings: &MultiCoreSettings, 
            original: &[T], original_length: usize, original_step: usize, 
            target: &mut [T], target_length: usize, target_step: usize, 
            arguments: S, ref function: F)
		where F: Fn(&[T], Range<usize>, &mut [T], S) + 'static + Sync,
			  T : Float + Copy + Clone + Send + Sync,
			  S: Sync + Copy
	{
		let number_of_chunks = Chunk::determine_number_of_chunks(original_length, complexity, settings);
		if number_of_chunks > 1
		{
			let chunks = Chunk::partition_mut(target, target_length, target_step, number_of_chunks);
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
		let chunks = Chunk::partition_mut(&mut array, 256, 4, 2);
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
		let chunks = Chunk::partition_mut(&mut array, 1023, 4, 8);
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