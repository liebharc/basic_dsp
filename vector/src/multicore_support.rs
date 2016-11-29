use num_cpus;
use std::slice::{Chunks, ChunksMut};
use num::traits::Float;
use crossbeam;
use std::ops::Range;
use std::sync::{Mutex, Arc};
use std::mem;
use super::RealNumber;
use std::iter::Iterator;

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
    Large,
}

/// Holds parameters which specify how multiple cores are used
/// to execute an operation.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct MultiCoreSettings {
    /// All operations will be limited to not create more threads than specified here
    pub core_limit: usize,

    /// Indicates whether the temp arrays of a vector should already be allocated during
    /// construction
    pub early_temp_allocation: bool,
}

impl MultiCoreSettings {
    /// Creates multi core settings with default values
    pub fn default() -> MultiCoreSettings {
        // Half because we assume hyper threading and that we will keep a core so busy
        // that hyper threading isn't of any use
        Self::new(num_cpus::get() / 2, false)
    }

    /// Creates multi core settings with the given values.
    pub fn new(core_limit: usize, early_temp_allocation: bool) -> MultiCoreSettings {
        MultiCoreSettings {
            core_limit: if core_limit >= 1 { core_limit } else { 1 },
            early_temp_allocation: early_temp_allocation,
        }
    }
}

/// Contains logic which helps to perform an operation
/// in parallel by dividing an array into chunks.
pub struct Chunk;
impl Chunk {
    /// Figures out how many threads make use for the an operation with the given complexity on
    /// an array with the given size.
    ///
    /// This method tries to balance the expected performance gain vs. CPU utilization
    /// since there is in most cases
    /// no point to keep all CPU cores busy only to get 5 to 10% performance gain.
    /// The expected performance gain is roughly estimated based on three factors:
    /// 1. More cores improves the calculation speed according to Amdahl's law
    ///    (`https://en.wikipedia.org/wiki/Amdahl's_law`)
    /// 2. Spawning a thread consumes time and so the array length must be large
    ///    enough to so that the expected performance
    ///    gain justifies the effort to spawn a thread/invoke the thread pool.
    /// 3. The CPU is not the only resource and if one of the other resources is a
    ///    bottleneck then Amdahl's law won't be applicable.
    ///    The memory bus speed is limited (20GB/s in case of a typical 2015 consumer
    ///    laptop in the price range of $1000) and
    ///    for operations which only require a few CPU cycles already one or two cores
    ///    will process data faster than the
    ///    memory bus is able to provide and to transport back. Using more cores then
    ///    only creates heat but no performance benefit.
    #[inline]
    fn determine_number_of_chunks(array_length: usize,
                                  complexity: Complexity,
                                  settings: &MultiCoreSettings)
                                  -> usize {
        let mut cores = num_cpus::get();
        if cores > settings.core_limit {
            cores = settings.core_limit;
        }
        
        if cores == 1 {
            cores
        } else if complexity == Complexity::Small {
            // Recent tests seem to indicate that even for large arrays it never makes
            // sense to spawn threads for trivial instructions. A single CPU core with SIMD
            // is already fast enough to occupy the max memory bandwidth
            1
        } else if complexity == Complexity::Medium {
            if array_length < 50000 {
                1
            } else if array_length < 100000 {
                if cores >= 2 { 2 } else { 1 }
            } else {
                cores
            }
        } else if array_length < 30000 {
            // complexity == Complexity::Large
            1
        } else { 
            cores 
        }
    }

    /// Partitions an array into the given number of chunks. It makes sure that all chunks
    /// have the same size
    /// and so it will happen that some elements at the end of the array are not part of any chunk.
    #[inline]
    fn partition<T>(array: &[T], step_size: usize, number_of_chunks: usize) -> Chunks<T>
        where T: Copy + Clone + Send + Sync
    {
        let chunk_size = Chunk::calc_chunk_size(array.len(), step_size, number_of_chunks);
        array[0..array.len()].chunks(chunk_size)
    }

    /// Partitions an array into the given number of chunks. It makes sure that all chunks
    /// have the same size
    /// and so it will happen that some elements at the end of the array are not part of any chunk.
    #[inline]
    fn partition_mut<T>(array: &mut [T], step_size: usize, number_of_chunks: usize) -> ChunksMut<T>
        where T: Copy + Clone + Send
    {
        let chunk_size = Chunk::calc_chunk_size(array.len(), step_size, number_of_chunks);
        array.chunks_mut(chunk_size)
    }

    #[inline]
    fn calc_chunk_size(array_length: usize, step_size: usize, number_of_chunks: usize) -> usize {
        let mut chunk_size = (array_length as f64 / number_of_chunks as f64).ceil() as usize;
        let remainder = chunk_size % step_size;
        if remainder > 0 {
            chunk_size += step_size - chunk_size % step_size;
        }

        chunk_size
    }

    /// This function returns the ranges which correspond to the chunks generated by
    /// partition_in_number`.
    #[inline]
    fn partition_in_ranges(array_length: usize,
                           step_size: usize,
                           number_of_chunks: usize)
                           -> Vec<Range<usize>> {
        let chunk_size = Chunk::calc_chunk_size(array_length, step_size, number_of_chunks);
        let mut ranges = Vec::with_capacity(number_of_chunks);
        let mut sum = 0;
        for i in 0..number_of_chunks {
            let new_sum = if i < number_of_chunks - 1 {
                sum + chunk_size
            } else {
                array_length
            };
            ranges.push(Range {
                start: sum,
                end: new_sum,
            });
            sum = new_sum;
        }

        ranges
    }

    /// Executes the given function on the first `array_length` elements of the given array in
    /// parallel and passes the argument to all function calls.
    #[inline]
    pub fn execute_partial<'a, T, S, F>(complexity: Complexity,
                                        settings: &MultiCoreSettings,
                                        array: &mut [T],
                                        step_size: usize,
                                        arguments: S,
                                        ref function: F)
        where F: Fn(&mut [T], S) + 'a + Sync,
              T: RealNumber,
              S: Sync + Copy + Send
    {
        let array_length = array.len();
        let number_of_chunks =
            Chunk::determine_number_of_chunks(array_length, complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition_mut(array, step_size, number_of_chunks);
            crossbeam::scope(|scope| {
                for chunk in chunks {
                    scope.spawn(move || {
                        function(chunk, arguments);
                    });
                }
            });
        } else {
            function(array, arguments);
        }
    }

    /// Executes the given function on the first `array_length` elements of the given list of
    /// arrays in parallel and passes the argument to all function calls.
    #[inline]
    pub fn execute_partial_multidim<'a, T, S, F>(complexity: Complexity,
                                                 settings: &MultiCoreSettings,
                                                 array: &mut [&mut [T]],
                                                 range: Range<usize>,
                                                 step_size: usize,
                                                 arguments: S,
                                                 ref function: F)
        where F: Fn(&mut Vec<&mut [T]>, Range<usize>, S) + 'a + Sync,
              T: RealNumber,
              S: Sync + Copy + Send
    {
        let dimensions = array.len();
        let number_of_chunks =
            Chunk::determine_number_of_chunks(range.end - range.start, complexity, settings);
        if number_of_chunks > 1 {
            let ranges =
                Chunk::partition_in_ranges(range.end - range.start, step_size, number_of_chunks);
            // As an example: If this function is called with 2 arrays,
            // and each array has 1000 elements and is split into 4 chunks then
            // this:
            // let matrix = array.iter_mut().map(|a| {
            //    Chunk::partition_mut(a, array_length, step_size, number_of_chunks)
            // }).collect();
            // would give us a layout like this:
            // Vector with 2 elements * 4 chunks * each chunk with 250 elements (mutable)
            //
            // But what we want to have is:
            // 4 chunks * a vector of 2 elements * 250 elements in each vector (mutable)
            //
            // We achieve this by a) flatten the structure (flat_map) so that we get:
            // array1 chunk1
            // array1 chunk2
            // array1 chunk3
            // array1 chunk4
            // array2 chunk1
            // array2 chunk2
            // array2 chunk3
            // array2 chunk4
            //
            // and b) creating a nested vector structure so that we can push
            // the elements on it in the correct order
            let mut flat_layout: Vec<&mut [T]> = array.iter_mut()
                .flat_map(|a| {
                    Chunk::partition_mut(&mut a[range.start..range.end],
                                         step_size,
                                         number_of_chunks)
                })
                .collect();

            let mut reorganized = Vec::with_capacity(number_of_chunks);
            for _ in 0..number_of_chunks {
                reorganized.push(Vec::with_capacity(dimensions));
            }
            let mut i = flat_layout.len();
            while i > 0 {
                let elem = flat_layout.pop().unwrap();
                reorganized[i % number_of_chunks].push(elem);
                i -= 1;
            }

            crossbeam::scope(|scope| {
                for chunk in reorganized.iter_mut().zip(ranges) {
                    scope.spawn(move || {
                        function(chunk.0, chunk.1, arguments);
                    });
                }
            });
        } else {
            let mut shortened: Vec<&mut [T]> =
                array.iter_mut().map(|a| &mut a[range.start..range.end]).collect();
            function(&mut shortened, range, arguments);
        }
    }

    /// Executes the given function on the all elements of the array and also tells the function
    /// on which range/chunk it operates on.
    #[inline]
    pub fn execute_with_range<'a, T, S, F>(complexity: Complexity,
                                           settings: &MultiCoreSettings,
                                           array: &mut [T],
                                           step_size: usize,
                                           arguments: S,
                                           ref function: F)
        where F: Fn(&mut [T], Range<usize>, S) + 'a + Sync,
              T: Copy + Clone + Send + Sync,
              S: Sync + Copy + Send
    {
        let array_length = array.len();
        let number_of_chunks =
            Chunk::determine_number_of_chunks(array_length, complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition_mut(array, step_size, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(array_length, step_size, chunks.len());
            crossbeam::scope(|scope| {
                for chunk in chunks.zip(ranges) {
                    scope.spawn(move || {
                        function(chunk.0, chunk.1, arguments);
                    });
                }
            });
        } else {
            function(array,
                     Range {
                         start: 0,
                         end: array_length,
                     },
                     arguments);
        }
    }

    /// Executes the given function on an unspecified number and size of chunks on the array and
    /// returns the result of each chunk.
    #[inline]
    pub fn map_on_array_chunks<'a, T, S, F, R>(complexity: Complexity,
                                               settings: &MultiCoreSettings,
                                               array: &[T],
                                               step_size: usize,
                                               arguments: S,
                                               ref function: F)
                                               -> Vec<R>
        where F: Fn(&[T], Range<usize>, S) -> R + 'a + Sync,
              T: Copy + Clone + Send + Sync,
              S: Sync + Copy + Send,
              R: Send
    {
        let array_len = array.len();
        let number_of_chunks = Chunk::determine_number_of_chunks(array.len(), complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition(array, step_size, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(array_len, step_size, chunks.len());
            let result = Vec::with_capacity(chunks.len());
            let stack_array = Arc::new(Mutex::new(result));
            crossbeam::scope(|scope| {
                for chunk in chunks.zip(ranges) {
                    let stack_array = stack_array.clone();
                    scope.spawn(move || {
                        let r = function(chunk.0, chunk.1, arguments);
                        stack_array.lock().unwrap().push(r);
                    });
                }
            });
            let mut guard = stack_array.lock().unwrap();
            mem::replace(&mut guard, Vec::new())
        } else {
            let result = function(array,
                                  Range {
                                      start: 0,
                                      end: array_len,
                                  },
                                  arguments);
            vec![result]
        }
    }

    /// Executes the given function on the all elements of the array and also tells the
    /// function on which range/chunk it operates on.
    ///
    /// This function will chunk the array into an even number and pass every
    /// call to `function` two chunks. The two chunks will always be symmetric
    /// around 0. This allows `function` to make use of symmetry properties of the
    /// underlying data or the argument.
    #[inline]
    pub fn execute_sym_pairs_with_range<'a, T, S, F>(complexity: Complexity,
                                                     settings: &MultiCoreSettings,
                                                     array: &mut [T],
                                                     step_size: usize,
                                                     arguments: S,
                                                     ref function: F)
        where F: Fn(&mut &mut [T],
                    &Range<usize>,
                    &mut &mut [T],
                    &Range<usize>,
                    S) + 'a + Sync,
              T: Copy + Clone + Send + Sync,
              S: Sync + Copy + Send
    {
        let array_length = array.len();
        let number_of_chunks =
            2 * Chunk::determine_number_of_chunks(array_length, complexity, settings);
        if number_of_chunks > 2 {
            let chunks = Chunk::partition_mut(array, step_size, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(array_length, step_size, chunks.len());
            let mut i = 0;
            let (mut chunks1, mut chunks2): (Vec<_>, Vec<_>) = chunks.partition(|_c| {
                i += 1;
                i <= number_of_chunks / 2
            });
            i = 0;
            let (ranges1, ranges2): (Vec<_>, Vec<_>) = ranges.iter().partition(|_r| {
                i += 1;
                i <= number_of_chunks / 2
            });
            let chunks2 = chunks2.iter_mut().rev();
            let ranges2 = ranges2.iter().rev();
            let zipped1 = chunks1.iter_mut().zip(ranges1);
            let zipped2 = chunks2.zip(ranges2);
            crossbeam::scope(|scope| {
                for chunk in zipped1.zip(zipped2) {
                    scope.spawn(move || {
                        let (pair1, pair2) = chunk;
                        function(pair1.0, pair1.1, pair2.0, pair2.1, arguments);
                    });
                }
            });
        } else {
            let mut chunks = Chunk::partition_mut(array, step_size, number_of_chunks);
            let mut chunks1 = chunks.next().unwrap();
            let len1 = chunks1.len();
            let mut chunks2 = chunks.next().unwrap();
            function(&mut chunks1,
                     &Range {
                         start: 0,
                         end: len1,
                     },
                     &mut chunks2,
                     &Range {
                         start: len1,
                         end: array_length,
                     },
                     arguments);
        }
    }

    /// Executes the given function on the all elements of the array in parallel. A result is
    /// returned for each chunk.
    #[inline]
    pub fn get_a_fold_b<'a, F, T, R>(complexity: Complexity,
                                     settings: &MultiCoreSettings,
                                     a: &[T],
                                     a_step: usize,
                                     b: &[T],
                                     b_step: usize,
                                     ref function: F)
                                     -> Vec<R>
        where F: Fn(&[T], Range<usize>, &[T]) -> R + 'a + Sync,
              T: Float + Copy + Clone + Send + Sync,
              R: Send
    {
        let a_len = a.len();
        let b_len = b.len();
        let number_of_chunks = Chunk::determine_number_of_chunks(a_len, complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition(b, b_step, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(a_len, a_step, chunks.len());
            let result = Vec::with_capacity(chunks.len());
            let stack_array = Arc::new(Mutex::new(result));
            crossbeam::scope(|scope| {
                for chunk in chunks.zip(ranges) {
                    let stack_array = stack_array.clone();
                    scope.spawn(move || {
                        let r = function(a, chunk.1, chunk.0);
                        stack_array.lock().unwrap().push(r);
                    });
                }
            });
            let mut guard = stack_array.lock().unwrap();
            mem::replace(&mut guard, Vec::new())
        } else {
            let result = function(a,
                                  Range {
                                      start: 0,
                                      end: a_len,
                                  },
                                  &b[0..b_len]);
            vec![result]
        }
    }

    /// Executes the given function on the all elements of the array in parallel. A result is
    /// returned for each chunk.
    #[inline]
    pub fn get_chunked_results<'a, F, S, T, R>(complexity: Complexity,
                                               settings: &MultiCoreSettings,
                                               a: &[T],
                                               a_step: usize,
                                               arguments: S,
                                               ref function: F)
                                               -> Vec<R>
        where F: Fn(&[T], Range<usize>, S) -> R + 'a + Sync,
              T: Float + Copy + Clone + Send + Sync,
              R: Send,
              S: Sync + Copy + Send
    {
        let a_len = a.len();
        let number_of_chunks = Chunk::determine_number_of_chunks(a_len, complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition(a, a_step, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(a_len, a_step, chunks.len());
            let result = Vec::with_capacity(chunks.len());
            let stack_array = Arc::new(Mutex::new(result));
            crossbeam::scope(|scope| {
                for chunk in chunks.zip(ranges) {
                    let stack_array = stack_array.clone();
                    scope.spawn(move || {
                        let r = function(chunk.0, chunk.1, arguments);
                        stack_array.lock().unwrap().push(r);
                    });
                }
            });
            let mut guard = stack_array.lock().unwrap();
            mem::replace(&mut guard, Vec::new())
        } else {
            let result = function(&a[0..a_len],
                                  Range {
                                      start: 0,
                                      end: a_len,
                                  },
                                  arguments);
            vec![result]
        }
    }

    /// Executes the given function on the all elements of the array in parallel and passes
    /// the argument to all function calls.. Results are intended to be stored in the target array.
    #[inline]
    pub fn from_src_to_dest<'a, T, S, F>(complexity: Complexity,
                                         settings: &MultiCoreSettings,
                                         original: &[T],
                                         original_step: usize,
                                         target: &mut [T],
                                         target_step: usize,
                                         arguments: S,
                                         ref function: F)
        where F: Fn(&[T], Range<usize>, &mut [T], S) + 'a + Sync,
              T: Float + Copy + Clone + Send + Sync,
              S: Sync + Copy + Send
    {
        let original_length = original.len();
        let number_of_chunks =
            Chunk::determine_number_of_chunks(original_length, complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition_mut(target, target_step, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(original_length, original_step, chunks.len());

            crossbeam::scope(|scope| {
                for chunk in chunks.zip(ranges) {
                    scope.spawn(move || {
                        function(original, chunk.1, chunk.0, arguments);
                    });
                }
            });
        } else {
            function(original,
                     Range {
                         start: 0,
                         end: original_length,
                     },
                     target,
                     arguments);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    #[test]
    fn partition_array() {
        let mut array = [0.0; 256];
        let chunks = Chunk::partition_mut(&mut array, 4, 2);
        assert_eq!(chunks.len(), 2);
        for chunk in chunks {
            assert_eq!(chunk.len(), 128);
        }
    }

    #[test]
    fn partition_array_8_cores() {
        let mut array = [0.0; 1023];
        let chunks = Chunk::partition_mut(&mut array, 4, 8);
        assert_eq!(chunks.len(), 8);
        let mut i = 0;
        for chunk in chunks {
            let expected = if i >= 7 { 127 } else { 128 };
            assert_eq!(chunk.len(), expected);
            i += 1;
        }
    }

    #[test]
    fn partitionin_ranges() {
        let ranges = Chunk::partition_in_ranges(1023, 4, 2);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0],
                   Range {
                       start: 0,
                       end: 512,
                   });
        assert_eq!(ranges[1],
                   Range {
                       start: 512,
                       end: 1023,
                   });
    }
}
