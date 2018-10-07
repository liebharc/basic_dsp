#![cfg_attr(feature = "cargo-clippy", allow(clippy::toplevel_ref_arg))]

use super::Complexity;
use crossbeam;
use inline_vector::InlineVector;
use linreg::linear_regression;
use num_cpus;
use num_traits::FromPrimitive;
use numbers::*;
use std::iter::Iterator;
use std::mem;
use std::ops::Range;
use std::slice::{Chunks, ChunksMut};
use std::sync::{Arc, Mutex};
use time;

/// Holds parameters which specify how multiple cores are used
/// to execute an operation.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct MultiCoreSettings {
    /// All operations will be limited to not create more threads than specified here
    pub core_limit: usize,
}

lazy_static! {
    static ref NUMBER_OF_CORES: usize = num_cpus::get();
}

impl MultiCoreSettings {
    /// Creates multi core settings with default values
    pub fn default() -> MultiCoreSettings {
        Self::single_threaded()
    }

    /// Creates multi core settings so that no thread will be spawned.
    pub fn single_threaded() -> MultiCoreSettings {
        Self::new(1)
    }

    /// Creates multi core so that threads will be spawned if this appears to be beneficial.
    pub fn parallel() -> MultiCoreSettings {
        if *NUMBER_OF_CORES == 1 {
            Self::new(1)
        } else if *NUMBER_OF_CORES == 2 {
            // If two cores are available then use both of them if multi-threading is requested
            Self::new(2)
        } else {
            // Half because we assume hyper threading and that we will keep a core so busy
            // that hyper threading isn't of any use
            Self::new(*NUMBER_OF_CORES / 2)
        }
    }

    /// Creates multi core settings with the given values.
    pub fn new(core_limit: usize) -> MultiCoreSettings {
        MultiCoreSettings {
            core_limit: if core_limit >= 1 { core_limit } else { 1 },
        }
    }
}

/// Calibration information which determines when multi threaded code will be used.
#[derive(Debug, PartialEq)]
struct Calibration {
    med_dual_core_threshold: usize,
    med_multi_core_threshold: usize,
    large_dual_core_threshold: usize,
    large_multi_core_threshold: usize,
    duration_cal_routine: f64,
    cal_routine_result_code: u32,
}

/// Limits a value between min and max
fn limit(value: usize, min: usize, max: usize) -> usize {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn measure(number_of_cores: usize, data: &mut Vec<f64>) -> Result<f64, u32> {
    let start = time::precise_time_ns();
    Chunk::execute_on_chunks(&mut data[..], 1, (), number_of_cores, move |array, _arg| {
        for v in array {
            *v = v.sin();
        }
    });
    let result = (time::precise_time_ns() - start) as f64;
    if result < 1000.0 {
        Err(1) // Likely the compiler optimized our benchmark code away so we have to work with defaults
    } else {
        Ok(result)
    }
}

/// Fits data to the equation `y = m * x`
fn proportional_regression(x: &[f64], y: &[f64]) -> Option<f64> {
    assert!(x.len() == y.len());
    let mut sum = 0.0;
    for (x, y) in x.iter().zip(y) {
        sum += *y / *x;
    }

    let result = sum / x.len() as f64;
    if result.is_nan() || result.is_infinite() {
        None
    } else {
        Some(result)
    }
}

/// Calculates the intersection of two lines `m1 * x + b1 = m2 * x + b2`
fn intersection(m1: f64, b1: f64, m2: f64, b2: f64) -> Option<usize> {
    usize::from_f64(((b2 - b1) / (m1 - m2)).round().abs())
}

/// Runs a couple of benchmarks to obtain the multi core thresholds.
///
/// The algorithm assumes that there is a linear relation between the number of elements
/// in a vector and the execution speed of the loops (which make most the basic_dsp code).
/// It then estimates the linear relation using linear regression for different number of cores
/// and finally calculations the intersections between those lines.
///
/// It also adds some less reasonable values to the calculation so that if in doubt no threads
/// are spawned.
fn attempt_calibrate(number_of_cores: usize) -> Result<Calibration, u32> {
    if number_of_cores == 1 {
        return Err(2); // In this case the calibration results don't matter
    }

    let mut size = 10000;
    let step = 5000;
    let iterations = 10;

    let mut sizes = Vec::with_capacity(iterations);
    let mut ono_thread = Vec::with_capacity(iterations);
    let mut two_threads = Vec::with_capacity(iterations);
    let mut max_threads = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        sizes.push(size as f64);
        let mut data = vec![1.0; size];
        ono_thread.push(try!(measure(1, &mut data)));
        let two_threads_result = try!(measure(2, &mut data));
        two_threads.push(two_threads_result);
        if number_of_cores > 2 {
            max_threads.push(try!(measure(number_of_cores, &mut data)));
        } else {
            max_threads.push(two_threads_result);
        }

        size += step;
    }

    let ono_thread = proportional_regression(&sizes, &ono_thread);
    let two_threads = linear_regression::<f64, f64, f64>(&sizes, &two_threads);
    let max_threads = linear_regression::<f64, f64, f64>(&sizes, &max_threads);
    if two_threads.is_none() || max_threads.is_none() {
        return Err(3); // If curve fitting fails then work with defaults
    }

    let t1_m = ono_thread.unwrap();
    let t1_b = 0.0; // due to asssumed proportional relation
    let (t2_m, t2_b) = two_threads.unwrap();
    let (tmax_m, tmax_b) = max_threads.unwrap();

    let med_dual_core_threshold_res = intersection(t1_m, t1_b, t2_m, t2_b);
    let med_multi_core_threshold_res = intersection(t1_m, t1_b, tmax_m, tmax_b);
    if med_dual_core_threshold_res.is_none() || med_multi_core_threshold_res.is_none() {
        return Err(4);
    }

    let dual_core_min = 5_000;
    let dual_core_max = 100_000;
    let multi_core_max = 200_000;

    // The multiplication factor is the less well reasoned part in this equiation. It should only help
    // to avoid that threads are spawned too aggressively.
    let med_dual_core_threshold = limit(
        2 * med_dual_core_threshold_res.unwrap(),
        dual_core_min,
        dual_core_max,
    );
    let med_multi_core_threshold = limit(
        2 * med_multi_core_threshold_res.unwrap(),
        med_dual_core_threshold + 1,
        multi_core_max,
    );
    let large_dual_core_threshold = limit(
        med_dual_core_threshold_res.unwrap(),
        dual_core_min,
        dual_core_max,
    );
    let large_multi_core_threshold = limit(
        med_multi_core_threshold_res.unwrap(),
        large_dual_core_threshold + 1,
        multi_core_max,
    );
    Ok(Calibration {
        med_dual_core_threshold,
        med_multi_core_threshold,
        large_dual_core_threshold,
        large_multi_core_threshold,
        duration_cal_routine: 0.0,  // Will be set by the calling function
        cal_routine_result_code: 0, // Success
    })
}

fn calibrate(number_of_cores: usize) -> Calibration {
    let start = time::precise_time_s();
    match attempt_calibrate(number_of_cores) {
        Ok(mut calibration) => {
            calibration.duration_cal_routine = time::precise_time_s() - start;
            calibration
        }
        Err(err_code) => Calibration {
            med_dual_core_threshold: 50_000,
            med_multi_core_threshold: 100_000,
            large_dual_core_threshold: 20_000,
            large_multi_core_threshold: 30_000,
            duration_cal_routine: time::precise_time_s() - start,
            cal_routine_result_code: err_code,
        },
    }
}

lazy_static! {
    static ref CALIBRATION: Calibration = calibrate(*NUMBER_OF_CORES);
}

/// Prints debug information about the calibration. The calibration determines when the library
/// will start to spawn threads. If a calibration hasn't been performed yet
/// than calling this function will trigger the calibration.
///
/// The returned value might change in format and content at any time.
pub fn print_calibration() -> String {
    format!("{:?}", *CALIBRATION)
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
    fn determine_number_of_chunks(
        array_length: usize,
        complexity: Complexity,
        settings: MultiCoreSettings,
    ) -> usize {
        if settings.core_limit == 1 {
            settings.core_limit
        } else if complexity == Complexity::Small {
            // Recent tests seem to indicate that even for large arrays it never makes
            // sense to spawn threads for trivial instructions. A single CPU core with SIMD
            // is already fast enough to occupy the max memory bandwidth
            1
        } else if complexity == Complexity::Medium {
            let calibration = &CALIBRATION;
            if array_length < calibration.med_dual_core_threshold {
                1
            } else if array_length < calibration.med_multi_core_threshold {
                2
            } else {
                settings.core_limit
            }
        } else {
            // Complexity::Large
            let calibration = &CALIBRATION;
            if array_length < calibration.large_dual_core_threshold {
                1
            } else if array_length < calibration.large_multi_core_threshold {
                2
            } else {
                settings.core_limit
            }
        }
    }

    /// Partitions an array into the given number of chunks. It makes sure that all chunks
    /// have the same size
    /// and so it will happen that some elements at the end of the array are not part of any chunk.
    #[inline]
    fn partition<T>(array: &[T], step_size: usize, number_of_chunks: usize) -> Chunks<T>
    where
        T: Copy + Clone + Send + Sync,
    {
        let chunk_size = Chunk::calc_chunk_size(array.len(), step_size, number_of_chunks);
        array[0..array.len()].chunks(chunk_size)
    }

    /// Partitions an array into the given number of chunks. It makes sure that all chunks
    /// have the same size
    /// and so it will happen that some elements at the end of the array are not part of any chunk.
    #[inline]
    fn partition_mut<T>(array: &mut [T], step_size: usize, number_of_chunks: usize) -> ChunksMut<T>
    where
        T: Copy + Clone + Send,
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
    fn partition_in_ranges(
        array_length: usize,
        step_size: usize,
        number_of_chunks: usize,
    ) -> Vec<Range<usize>> {
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
    pub fn execute_partial<'a, T, S, F>(
        complexity: Complexity,
        settings: MultiCoreSettings,
        array: &mut [T],
        step_size: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&mut [T], S) + 'a + Sync,
        T: RealNumber,
        S: Sync + Copy + Send,
    {
        let array_length = array.len();
        let number_of_chunks =
            Chunk::determine_number_of_chunks(array_length, complexity, settings);
        if number_of_chunks > 1 {
            Chunk::execute_on_chunks(array, step_size, arguments, number_of_chunks, function);
        } else {
            function(array, arguments);
        }
    }

    fn execute_on_chunks<'a, T, S, F>(
        array: &mut [T],
        step_size: usize,
        arguments: S,
        number_of_chunks: usize,
        ref function: F,
    ) where
        F: Fn(&mut [T], S) + 'a + Sync,
        T: RealNumber,
        S: Sync + Copy + Send,
    {
        let chunks = Chunk::partition_mut(array, step_size, number_of_chunks);
        crossbeam::scope(|scope| {
            for chunk in chunks {
                scope.spawn(move || {
                    function(chunk, arguments);
                });
            }
        });
    }

    /// Executes the given function on the first `array_length` elements of the given list of
    /// arrays in parallel and passes the argument to all function calls.
    #[inline]
    pub fn execute_partial_multidim<'a, T, S, F>(
        complexity: Complexity,
        settings: MultiCoreSettings,
        array: &mut [&mut [T]],
        range: Range<usize>,
        step_size: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&mut InlineVector<&mut [T]>, Range<usize>, S) + 'a + Sync,
        T: RealNumber,
        S: Sync + Copy + Send,
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
            let mut flat_layout: Vec<&mut [T]> = array
                .iter_mut()
                .flat_map(|a| {
                    Chunk::partition_mut(
                        &mut a[range.start..range.end],
                        step_size,
                        number_of_chunks,
                    )
                })
                .collect();

            let mut reorganized = Vec::with_capacity(number_of_chunks);
            for _ in 0..number_of_chunks {
                reorganized.push(InlineVector::with_capacity(dimensions));
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
            let mut shortened = InlineVector::with_capacity(range.end - range.start);
            for n in array.iter_mut().map(|a| &mut a[range.start..range.end]) {
                shortened.push(n);
            }
            function(&mut shortened, range, arguments);
        }
    }

    /// Executes the given function on the all elements of the array and also tells the function
    /// on which range/chunk it operates on.
    #[inline]
    pub fn execute_with_range<'a, T, S, F>(
        complexity: Complexity,
        settings: MultiCoreSettings,
        array: &mut [T],
        step_size: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&mut [T], Range<usize>, S) + 'a + Sync,
        T: Copy + Clone + Send + Sync,
        S: Sync + Copy + Send,
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
            function(
                array,
                Range {
                    start: 0,
                    end: array_length,
                },
                arguments,
            );
        }
    }

    /// Executes the given function on an unspecified number and size of chunks on the array and
    /// returns the result of each chunk.
    #[inline]
    pub fn map_on_array_chunks<'a, T, S, F, R>(
        complexity: Complexity,
        settings: MultiCoreSettings,
        array: &[T],
        step_size: usize,
        arguments: S,
        ref function: F,
    ) -> InlineVector<R>
    where
        F: Fn(&[T], Range<usize>, S) -> R + 'a + Sync,
        T: Copy + Clone + Send + Sync,
        S: Sync + Copy + Send,
        R: Send,
    {
        let array_len = array.len();
        let number_of_chunks = Chunk::determine_number_of_chunks(array.len(), complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition(array, step_size, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(array_len, step_size, chunks.len());
            let result = InlineVector::with_capacity(chunks.len());
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
            mem::replace(&mut guard, InlineVector::empty())
        } else {
            let result = function(
                array,
                Range {
                    start: 0,
                    end: array_len,
                },
                arguments,
            );
            InlineVector::with_elem(result)
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
    pub fn execute_sym_pairs_with_range<'a, T, S, F>(
        complexity: Complexity,
        settings: MultiCoreSettings,
        array: &mut [T],
        step_size: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&mut &mut [T], &Range<usize>, &mut &mut [T], &Range<usize>, S) + 'a + Sync,
        T: Copy + Clone + Send + Sync,
        S: Sync + Copy + Send,
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
            function(
                &mut chunks1,
                &Range {
                    start: 0,
                    end: len1,
                },
                &mut chunks2,
                &Range {
                    start: len1,
                    end: array_length,
                },
                arguments,
            );
        }
    }

    /// Executes the given function on the all elements of the array in parallel. A result is
    /// returned for each chunk.
    #[inline]
    pub fn get_a_fold_b<'a, F, T, R>(
        complexity: Complexity,
        settings: MultiCoreSettings,
        a: &[T],
        a_step: usize,
        b: &[T],
        b_step: usize,
        ref function: F,
    ) -> InlineVector<R>
    where
        F: Fn(&[T], Range<usize>, &[T]) -> R + 'a + Sync,
        T: Float + Copy + Clone + Send + Sync,
        R: Send,
    {
        let a_len = a.len();
        let b_len = b.len();
        let number_of_chunks = Chunk::determine_number_of_chunks(a_len, complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition(b, b_step, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(a_len, a_step, chunks.len());
            let result = InlineVector::with_capacity(chunks.len());
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
            mem::replace(&mut guard, InlineVector::empty())
        } else {
            let result = function(
                a,
                Range {
                    start: 0,
                    end: a_len,
                },
                &b[0..b_len],
            );
            InlineVector::with_elem(result)
        }
    }

    /// Executes the given function on the all elements of the array in parallel. A result is
    /// returned for each chunk.
    #[inline]
    pub fn get_chunked_results<'a, F, S, T, R>(
        complexity: Complexity,
        settings: MultiCoreSettings,
        a: &[T],
        a_step: usize,
        arguments: S,
        ref function: F,
    ) -> InlineVector<R>
    where
        F: Fn(&[T], Range<usize>, S) -> R + 'a + Sync,
        T: Float + Copy + Clone + Send + Sync,
        R: Send,
        S: Sync + Copy + Send,
    {
        let a_len = a.len();
        let number_of_chunks = Chunk::determine_number_of_chunks(a_len, complexity, settings);
        if number_of_chunks > 1 {
            let chunks = Chunk::partition(a, a_step, number_of_chunks);
            let ranges = Chunk::partition_in_ranges(a_len, a_step, chunks.len());
            let result = InlineVector::with_capacity(chunks.len());
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
            mem::replace(&mut guard, InlineVector::empty())
        } else {
            let result = function(
                &a[0..a_len],
                Range {
                    start: 0,
                    end: a_len,
                },
                arguments,
            );
            InlineVector::with_elem(result)
        }
    }

    /// Executes the given function on the all elements of the array in parallel and passes
    /// the argument to all function calls.. Results are intended to be stored in the target array.
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    #[inline]
    pub fn from_src_to_dest<'a, T, S, F>(
        complexity: Complexity,
        settings: MultiCoreSettings,
        original: &[T],
        original_step: usize,
        target: &mut [T],
        target_step: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&[T], Range<usize>, &mut [T], S) + 'a + Sync,
        T: Float + Copy + Clone + Send + Sync,
        S: Sync + Copy + Send,
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
            function(
                original,
                Range {
                    start: 0,
                    end: original_length,
                },
                target,
                arguments,
            );
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
        assert_eq!(ranges[0], Range { start: 0, end: 512 });
        assert_eq!(
            ranges[1],
            Range {
                start: 512,
                end: 1023,
            }
        );
    }

    #[test]
    fn calibration_test() {
        let cal = calibrate(4);
        assert!(cal.med_dual_core_threshold > 0);
        assert!(cal.med_multi_core_threshold > cal.med_dual_core_threshold);
        assert!(cal.large_multi_core_threshold > 0);
    }
}
