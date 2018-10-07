use super::Complexity;
use inline_vector::InlineVector;
use numbers::*;
use std::iter::Iterator;
use std::ops::Range;
use std::slice::ChunksMut;

/// Empty struct since multicore support is disabled
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct MultiCoreSettings;

impl MultiCoreSettings {
    /// Creates multi core settings with default values
    pub fn default() -> MultiCoreSettings {
        MultiCoreSettings
    }

    /// Creates multi core settings so that no thread will be spawned.
    pub fn single_threaded() -> MultiCoreSettings {
        MultiCoreSettings
    }

    /// Creates multi core settings. The argument will be ignored
    /// since multi threading is disabled. The method only exists
    /// to stay compatible with the multi-threaded version of this lib.
    pub fn new(_: usize) -> MultiCoreSettings {
        MultiCoreSettings
    }
}

/// Simplified version which has the same interface as the the `threading version`.
pub struct Chunk;
impl Chunk {
    /// Executes the given function on the first `array_length` elements of the given array in
    /// parallel and passes the argument to all function calls.
    #[inline]
    pub fn execute_partial<'a, T, S, F>(
        _: Complexity,
        _: MultiCoreSettings,
        array: &mut [T],
        _: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&mut [T], S) + 'a + Sync,
        T: RealNumber,
        S: Sync + Copy + Send,
    {
        function(array, arguments);
    }

    /// Executes the given function on the first `array_length` elements of the given list of
    /// arrays in parallel and passes the argument to all function calls.
    #[inline]
    pub fn execute_partial_multidim<'a, T, S, F>(
        _: Complexity,
        _: MultiCoreSettings,
        array: &mut [&mut [T]],
        range: Range<usize>,
        _: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&mut InlineVector<&mut [T]>, Range<usize>, S) + 'a + Sync,
        T: RealNumber,
        S: Sync + Copy + Send,
    {
        let mut shortened: InlineVector<&mut [T]> = array
            .iter_mut()
            .map(|a| &mut a[range.start..range.end])
            .collect();
        function(&mut shortened, range, arguments);
    }

    /// Executes the given function on the all elements of the array and also tells the function
    /// on which range/chunk it operates on.
    #[inline]
    pub fn execute_with_range<'a, T, S, F>(
        _: Complexity,
        _: MultiCoreSettings,
        array: &mut [T],
        _: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&mut [T], Range<usize>, S) + 'a + Sync,
        T: Copy + Clone + Send + Sync,
        S: Sync + Copy + Send,
    {
        let array_length = array.len();
        function(
            array,
            Range {
                start: 0,
                end: array_length,
            },
            arguments,
        );
    }

    /// Executes the given function on an unspecified number and size of chunks on the array and
    /// returns the result of each chunk.
    #[inline]
    pub fn map_on_array_chunks<'a, T, S, F, R>(
        _: Complexity,
        _: MultiCoreSettings,
        array: &[T],
        _: usize,
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

    /// Executes the given function on the all elements of the array and also tells the
    /// function on which range/chunk it operates on.
    ///
    /// This function will chunk the array into an even number and pass every
    /// call to `function` two chunks. The two chunks will always be symmetric
    /// around 0. This allows `function` to make use of symmetry properties of the
    /// underlying data or the argument.
    #[inline]
    pub fn execute_sym_pairs_with_range<'a, T, S, F>(
        _: Complexity,
        _: MultiCoreSettings,
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
        let mut chunks = Chunk::partition_mut(array, step_size, 2);
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

    /// Executes the given function on the all elements of the array in parallel. A result is
    /// returned for each chunk.
    #[inline]
    pub fn get_a_fold_b<'a, F, T, R>(
        _: Complexity,
        _: MultiCoreSettings,
        a: &[T],
        _: usize,
        b: &[T],
        _: usize,
        ref function: F,
    ) -> InlineVector<R>
    where
        F: Fn(&[T], Range<usize>, &[T]) -> R + 'a + Sync,
        T: Float + Copy + Clone + Send + Sync,
        R: Send,
    {
        let a_len = a.len();
        let b_len = b.len();
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

    /// Executes the given function on the all elements of the array in parallel. A result is
    /// returned for each chunk.
    #[inline]
    pub fn get_chunked_results<'a, F, S, T, R>(
        _: Complexity,
        _: MultiCoreSettings,
        a: &[T],
        _: usize,
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

    /// Executes the given function on the all elements of the array in parallel and passes
    /// the argument to all function calls.. Results are intended to be stored in the target array.
    #[inline]
    pub fn from_src_to_dest<'a, T, S, F>(
        _: Complexity,
        _: MultiCoreSettings,
        original: &[T],
        _: usize,
        target: &mut [T],
        _: usize,
        arguments: S,
        ref function: F,
    ) where
        F: Fn(&[T], Range<usize>, &mut [T], S) + 'a + Sync,
        T: Float + Copy + Clone + Send + Sync,
        S: Sync + Copy + Send,
    {
        let original_length = original.len();
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
}

/// Prints debug information about the calibration. Returns a constant as threading is deactivated.
pub fn print_calibration() -> String {
    "No threading".to_string()
}
