use {array_to_complex, array_to_complex_mut};
use multicore_support::{Chunk, MultiCoreSettings, Complexity};
use numbers::*;
use std::ops::*;
use std::mem;
use std::cmp;
use std::result;
use simd_extensions::*;
use std::fmt;
use std;

mod requirements;
pub use self::requirements::*;
mod to_from_vec_conversions;
pub use self::to_from_vec_conversions::*;
mod support_core;
pub use self::support_core::*;
#[cfg(any(feature="std", test))]
mod support_std;
#[cfg(any(feature="std", test))]
pub use self::support_std::*;
mod vec_impl_and_indexers;
pub use self::vec_impl_and_indexers::*;
mod complex;
pub use self::complex::*;
mod real;
pub use self::real::*;
mod time_freq;
pub use self::time_freq::*;
pub mod combined_ops;
mod rededicate_and_relations;
pub use self::rededicate_and_relations::*;
mod checks_and_results;
pub use self::checks_and_results::*;
mod general;
pub use self::general::*;
mod buffer;
pub use self::buffer::*;

/// Result for operations which transform a type (most commonly the type is a vector).
/// On success the transformed type is returned.
/// On failure it contains an error reason and the original type with with invalid data
/// which still can be used in order to avoid memory allocation.
pub type TransRes<T> = result::Result<T, (ErrorReason, T)>;

/// Void/nothing in case of success or a reason in case of an error.
pub type VoidResult = result::Result<(), ErrorReason>;

/// Scalar result or a reason in case of an error.
pub type ScalarResult<T> = result::Result<T, ErrorReason>;

/// The domain of a data vector
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum DataDomain {
    /// Time domain, the x-axis is in [s].
    Time,
    /// Frequency domain, the x-axis is in [Hz].
    Frequency,
}

/// Number space (real or complex) information.
pub trait NumberSpace: fmt::Debug + cmp::PartialEq + Clone {
    fn is_complex(&self) -> bool;

    /// For implementations which track meta data
    /// at runtime this method may be implemented to transition
    /// between different states. For all other implementations
    /// they may leave this empty.
    fn to_complex(&mut self);

    /// See `to_complex` for more details.
    fn to_real(&mut self);
}

/// Domain (time or frequency) information.
pub trait Domain: fmt::Debug + cmp::PartialEq + Clone {
    fn domain(&self) -> DataDomain;

    /// See `to_complex` for some details.
    fn to_freq(&mut self);

    /// See `to_complex` for some details.
    fn to_time(&mut self);
}

/// Trait for types containing real data.
pub trait RealNumberSpace: NumberSpace {}

/// Trait for types containing complex data.
pub trait ComplexNumberSpace: NumberSpace {}

/// Trait for types containing time domain data.
pub trait TimeDomain: Domain {}

/// Trait for types containing frequency domain data.
pub trait FrequencyDomain: Domain {}

/// Marker for types containing real data.
#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
pub struct RealData;
impl NumberSpace for RealData {
    fn is_complex(&self) -> bool {
        false
    }
    fn to_complex(&mut self) {}
    fn to_real(&mut self) {}
}
impl RealNumberSpace for RealData {}

/// Marker for types containing complex data.
#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
pub struct ComplexData;
impl NumberSpace for ComplexData {
    fn is_complex(&self) -> bool {
        true
    }
    fn to_complex(&mut self) {}
    fn to_real(&mut self) {}
}
impl ComplexNumberSpace for ComplexData {}

/// Marker for types containing real or complex data.
#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
pub struct RealOrComplexData {
    pub is_complex_current: bool,
}
impl NumberSpace for RealOrComplexData {
    fn is_complex(&self) -> bool {
        self.is_complex_current
    }

    fn to_complex(&mut self) {
        self.is_complex_current = true;
    }

    fn to_real(&mut self) {
        self.is_complex_current = false;
    }
}
impl RealNumberSpace for RealOrComplexData {}
impl ComplexNumberSpace for RealOrComplexData {}

/// Marker for types containing time data.
#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
pub struct TimeData;
impl Domain for TimeData {
    fn domain(&self) -> DataDomain {
        DataDomain::Time
    }
    fn to_freq(&mut self) {}
    fn to_time(&mut self) {}
}
impl TimeDomain for TimeData {}

/// Marker for types containing frequency data.
#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
pub struct FrequencyData;
impl Domain for FrequencyData {
    fn domain(&self) -> DataDomain {
        DataDomain::Frequency
    }
    fn to_time(&mut self) {}
    fn to_freq(&mut self) {}
}
impl FrequencyDomain for FrequencyData {}

/// Marker for types containing time or frequency data.
#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
pub struct TimeOrFrequencyData {
    pub domain_current: DataDomain,
}
impl Domain for TimeOrFrequencyData {
    fn domain(&self) -> DataDomain {
        self.domain_current
    }

    fn to_freq(&mut self) {
        self.domain_current = DataDomain::Frequency;
    }

    fn to_time(&mut self) {
        self.domain_current = DataDomain::Time;
    }
}

impl TimeDomain for TimeOrFrequencyData {}
impl FrequencyDomain for TimeOrFrequencyData {}

/// Expresses at compile time that two classes could potentially represent the same number space or domain. 
pub trait PosEq<O> { }
impl PosEq<RealData> for RealData { }
impl PosEq<RealOrComplexData> for RealData { }
impl PosEq<RealData> for RealOrComplexData { }
impl PosEq<ComplexData> for RealOrComplexData { }
impl PosEq<RealOrComplexData> for RealOrComplexData { }
impl PosEq<ComplexData> for ComplexData { }
impl PosEq<RealOrComplexData> for ComplexData { }

impl PosEq<TimeData> for TimeData { }
impl PosEq<TimeOrFrequencyData> for TimeData { }
impl PosEq<TimeData> for TimeOrFrequencyData { }
impl PosEq<FrequencyData> for TimeOrFrequencyData { }
impl PosEq<TimeOrFrequencyData> for TimeOrFrequencyData { }
impl PosEq<FrequencyData> for FrequencyData { }
impl PosEq<TimeOrFrequencyData> for FrequencyData { }
    
/// A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing
/// (DSP) operations.
///
/// Vectors come in different flavors:
///
/// 1. Time or Frequency domain
/// 2. Real or Complex numbers
/// 3. 32bit or 64bit floating point numbers
///
/// The first two flavors define meta information about the vector and provide compile time
/// information what operations are available with the given vector and how this will transform
/// the vector. This makes sure that some invalid operations are already discovered at compile
/// time. In case that this isn't desired or the information about the vector isn't known at
/// compile time there are the generic [`DataVec32`](type.DataVec32.html) and
/// [`DataVec64`](type.DataVec64.html) vectors available.
///
/// 32bit and 64bit flavors trade performance and memory consumption against accuracy.
/// 32bit vectors are roughly two times faster than 64bit vectors for most operations.
/// But remember that you should benchmark first before you give away accuracy for performance
/// unless however you are sure that 32bit accuracy is certainly good enough.
pub struct DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    /// The underlying storage. `self.len()` should be called to find out how many
    /// elements in `data` contain valid data.
    pub data: S,
    delta: T,
    domain: D,
    number_space: N,
    valid_len: usize,
    multicore_settings: MultiCoreSettings,
}

/// Holds meta data about a type.
#[derive(Clone)]
#[derive(Copy)]
pub struct TypeMetaData<T, N, D> {
    delta: T,
    domain: D,
    number_space: N,
    multicore_settings: MultiCoreSettings,
}

impl<S, T, N, D> fmt::Debug for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          D: Domain,
          N: NumberSpace
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "DspVec {{ points: {}, domain: {:?}, number_space: {:?} }}",
               self.valid_len,
               self.domain,
               self.number_space)
    }
}

/// Rounds a length so that it always devides by the length of a SIMD
/// register. This function assumes that `Reg32::len() > Reg64::len()`.
fn round_len(len: usize) -> usize {
    ((len + Reg32::len() - 1) / Reg32::len()) * Reg32::len()
}

/// Swaps the halves of two arrays. The `forward` paremeter
/// specifies what should happen then the `data.len()` is odd.
/// The function should always produce the same results as `fft_shift`
/// and `ifft_shift` in GNU Octave.
fn swap_array_halves<T>(data: &mut [T], forward: bool)
    where T: Copy
{
    let len = data.len();
    if len % 2 == 0 {
        let (lo, up) = data.split_at_mut(len / 2);
        for (lo, up) in lo.iter_mut().zip(up.iter_mut()) {
            mem::swap(lo, up);
        }
    } else {
        let step = if forward { len / 2 } else { len / 2 + 1 };
        let mut temp = data[0];
        let mut pos = step;
        for _ in 0..len {
            let pos_new = (pos + step) % len;
            unsafe { mem::swap(&mut temp, data.get_unchecked_mut(pos)) };
            pos = pos_new;
        }
    }
}

/// Creates an interleaved array slice from a complex array.
fn complex_to_array<T>(complex: &[Complex<T>]) -> &[T]
    where T: RealNumber
{
    use std::slice;
    use std::mem;
    unsafe {
        let len = complex.len();
        let trans: &[T] = mem::transmute(complex);
        slice::from_raw_parts(&trans[0] as *const T, len * 2)
    }
}

/// Creates an interleaved array slice from a complex array.
fn complex_to_array_mut<T>(complex: &mut [Complex<T>]) -> &mut [T]
    where T: RealNumber
{
    use std::slice;
    use std::mem;
    unsafe {
        let len = complex.len();
        let trans: &mut [T] = mem::transmute(complex);
        slice::from_raw_parts_mut(&mut trans[0] as *mut T, len * 2)
    }
}

impl<S, T, N, D> DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
	/// Executes a real function.
    #[inline]
    fn pure_real_operation<A, F>(&mut self, op: F, argument: A, complexity: Complexity)
        where A: Sync + Copy + Send,
              F: Fn(T, A) -> T + 'static + Sync
    {
        {
            let len = self.valid_len;
            let array = self.data.to_slice_mut();
            Chunk::execute_partial(complexity,
                                   &self.multicore_settings,
                                   &mut array[0..len],
                                   1,
                                   argument,
                                   move |array, argument| {
                                       for num in array {
                                           *num = op(*num, argument);
                                       }
                                   });
        }
    }

	/// Executes a complex function.
    #[inline]
    fn pure_complex_operation<A, F>(&mut self, op: F, argument: A, complexity: Complexity)
        where A: Sync + Copy + Send,
              F: Fn(Complex<T>, A) -> Complex<T> + 'static + Sync
    {
        {
            let len = self.valid_len;
            let array = self.data.to_slice_mut();
            Chunk::execute_partial(complexity,
                                   &self.multicore_settings,
                                   &mut array[0..len],
                                   2,
                                   argument,
                                   move |array, argument| {
                                       let array = array_to_complex_mut(array);
                                       for num in array {
                                           *num = op(*num, argument);
                                       }
                                   });
        }
    }

	/// Executes a real function with SIMD optimization.
    #[inline]
    fn simd_real_operation<A, F, G>(&mut self,
                                    simd_op: F,
                                    scalar_op: G,
                                    argument: A,
                                    complexity: Complexity)
        where A: Sync + Copy + Send,
              F: Fn(T::Reg, A) -> T::Reg + 'static + Sync,
              G: Fn(T, A) -> T + 'static + Sync
    {
        {
            let data_length = self.valid_len;
            let array = self.data.to_slice_mut();
            let (scalar_left, scalar_right, vectorization_length) =
                T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
            if vectorization_length > 0 {
                Chunk::execute_partial(complexity,
                                       &self.multicore_settings,
                                       &mut array[scalar_left..vectorization_length],
                                       T::Reg::len(),
                                       argument,
                                       move |array, argument| {
                                           let array = T::Reg::array_to_regs_mut(array);
                                           for reg in array {
                                               *reg = simd_op(*reg, argument);
                                           }
                                       });
            }
            for num in &mut array[0..scalar_left] {
                *num = scalar_op(*num, argument);
            }
            for num in &mut array[scalar_right..data_length] {
                *num = scalar_op(*num, argument);
            }
        }
    }

	/// Executes a complex function with SIMD optimization.
    #[inline]
    fn simd_complex_operation<A, F, G>(&mut self,
                                       simd_op: F,
                                       scalar_op: G,
                                       argument: A,
                                       complexity: Complexity)
        where A: Sync + Copy + Send,
              F: Fn(T::Reg, A) -> T::Reg + 'static + Sync,
              G: Fn(Complex<T>, A) -> Complex<T> + 'static + Sync
    {
        {
            let data_length = self.valid_len;
            let array = self.data.to_slice_mut();
            let (scalar_left, scalar_right, vectorization_length) =
                T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
            if vectorization_length > 0 {
                Chunk::execute_partial(complexity,
                                       &self.multicore_settings,
                                       &mut array[scalar_left..vectorization_length],
                                       T::Reg::len(),
                                       argument,
                                       move |array, argument| {
                                           let array = T::Reg::array_to_regs_mut(array);
                                           for reg in array {
                                               *reg = simd_op(*reg, argument);
                                           }
                                       });
            }
            {
                let array = array_to_complex_mut(&mut array[0..scalar_left]);
                for num in array {
                    *num = scalar_op(*num, argument);
                }
            }
            {
                let array = array_to_complex_mut(&mut array[scalar_right..data_length]);
                for num in array {
                    *num = scalar_op(*num, argument);
                }
            }
        }
    }

	/// Executes a function which converts a complex array into a real array.
    #[inline]
    fn pure_complex_to_real_operation<A, F, B>(&mut self,
                                               buffer: &mut B,
                                               op: F,
                                               argument: A,
                                               complexity: Complexity)
        where A: Sync + Copy + Send,
              F: Fn(Complex<T>, A) -> T + 'static + Sync,
              B: for<'a> Buffer<'a, S, T>
    {
        {
            let data_length = self.len();
            let mut result = buffer.borrow(data_length / 2);
            {
                let array = self.data.to_slice_mut();
                let temp = result.to_slice_mut();
                Chunk::from_src_to_dest(complexity,
                                        &self.multicore_settings,
                                        &array[0..data_length],
                                        2,
                                        &mut temp[0..data_length / 2],
                                        1,
                                        argument,
                                        move |array, range, target, argument| {
                    let array = array_to_complex(&array[range.start..range.end]);
                    for pair in array.iter().zip(target) {
                        let (src, dest) = pair;
                        *dest = op(*src, argument);
                    }
                });
                self.valid_len = data_length / 2;
            }
            result.trade(&mut self.data);
        }
    }

	/// Executes a function which converts a complex array into a real array. Conversion is applied in-place.
    #[inline]
    fn pure_complex_to_real_operation_inplace<A, F>(&mut self, op: F, argument: A)
        where A: Sync + Copy + Send,
              F: Fn(Complex<T>, A) -> T + 'static + Sync
    {
        {
            let data_length = self.len();
            let array = self.data.to_slice_mut();
            for i in 0..data_length / 2 {
                let input = Complex::new(array[2 * i], array[2 * i + 1]);
                array[i] = op(input, argument);
            }

            self.valid_len /= 2;
        }
    }

	/// Executes a function which converts a complex array into a real array.
    #[inline]
    fn simd_complex_to_real_operation<A, F, G, B>(&mut self,
                                                  buffer: &mut B,
                                                  simd_op: F,
                                                  scalar_op: G,
                                                  argument: A,
                                                  complexity: Complexity)
        where A: Sync + Copy + Send,
              F: Fn(T::Reg, A) -> T::Reg + 'static + Sync,
              G: Fn(Complex<T>, A) -> T + 'static + Sync,
              B: for<'a> Buffer<'a, S, T>
    {
        let data_length = self.len();
        let mut result = buffer.borrow(data_length / 2);
        {
            let array = self.data.to_slice_mut();
            let temp = result.to_slice_mut();
            let (scalar_left, scalar_right, vectorization_length) =
                T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
            if vectorization_length > 0 {
                Chunk::from_src_to_dest(complexity,
                                        &self.multicore_settings,
                                        &array[scalar_left..vectorization_length],
                                        T::Reg::len(),
                                        &mut temp[scalar_left / 2..vectorization_length / 2],
                                        T::Reg::len() / 2,
                                        argument,
                                        move |array, range, target, argument| {
                    let array = T::Reg::array_to_regs(&array[range.start..range.end]);
                    let mut j = 0;
                    for reg in array {
                        let result = simd_op(*reg, argument);
                        result.store_half_unchecked(target, j);
                        j += T::Reg::len() / 2;
                    }
                });
            }
            {
                let array = array_to_complex(&array[0..scalar_left]);
                for pair in array.iter().zip(&mut temp[0..scalar_left / 2]) {
                    let (src, dest) = pair;
                    *dest = scalar_op(*src, argument);
                }
            }
            {
                let array = array_to_complex(&array[scalar_right..data_length]);
                for pair in array.iter().zip(&mut temp[scalar_right / 2..data_length / 2]) {
                    let (src, dest) = pair;
                    *dest = scalar_op(*src, argument);
                }
            }

            self.valid_len /= 2;
        }
        result.trade(&mut self.data);
    }

	/// Forwards calls to `swap_array_halves`.
    #[inline]
    fn swap_halves_priv(&mut self, forward: bool) {
        let len = self.len();
        if len == 0 {
            return;
        }

        if self.is_complex() {
            let data = self.data.to_slice_mut();
            let data = array_to_complex_mut(&mut data[0..len]);
            swap_array_halves(data, forward);
        } else {
            let data = self.data.to_slice_mut();
            swap_array_halves(&mut data[0..len], forward);
        }
    }

	/// Multiplies the vector with a window function.
    #[inline]
    fn multiply_window_priv<TT, CMut, FA, F>(&mut self,
                                             is_symmetric: bool,
                                             convert_mut: CMut,
                                             function_arg: FA,
                                             fun: F)
        where CMut: Fn(&mut [T]) -> &mut [TT],
              FA: Copy + Sync + Send,
              F: Fn(FA, usize, usize) -> TT + 'static + Sync,
              TT: Zero + Mul<Output = TT> + Copy + Send + Sync + From<T>
    {
        if !is_symmetric {
            {
                let len = self.len();
                let points = self.points();
                let data = self.data.to_slice_mut();
                let converted = convert_mut(&mut data[0..len]);
                Chunk::execute_with_range(Complexity::Medium,
                                          &self.multicore_settings,
                                          converted,
                                          1,
                                          function_arg,
                                          move |array, range, arg| {
                    let mut j = range.start;
                    for num in array {
                        *num = (*num) * fun(arg, j, points);
                        j += 1;
                    }
                });
            }
        } else {
            {
                let len = self.len();
                let points = self.points();
                let data = self.data.to_slice_mut();
                let converted = convert_mut(&mut data[0..len]);
                Chunk::execute_sym_pairs_with_range(Complexity::Medium,
                                                    &self.multicore_settings,
                                                    converted,
                                                    1,
                                                    function_arg,
                                                    move |array1, range, array2, _, arg| {
                    assert!(array1.len() >= array2.len());
                    let mut j = range.start;
                    let len1 = array1.len();
                    let len_diff = len1 - array2.len();
                    {
                        let iter1 = array1.iter_mut();
                        let iter2 = array2.iter_mut().rev();
                        for (num1, num2) in iter1.zip(iter2) {
                            let arg = fun(arg, j, points);
                            *num1 = (*num1) * arg;
                            *num2 = (*num2) * arg;
                            j += 1;
                        }
                    }
                    for num1 in &mut array1[len1 - len_diff..len1] {
                        let arg = fun(arg, j, points);
                        *num1 = (*num1) * arg;
                        j += 1;
                    }
                });
            }
        }
    }
}

/// Buffer borrow type for `NoTradeBufferBurrow`.
pub struct NoTradeBufferBurrow<'a, T: RealNumber + 'a> {
    data: &'a mut [T],
}

impl<'a, T: RealNumber> Deref for NoTradeBufferBurrow<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.data
    }
}

impl<'a, T: RealNumber> DerefMut for NoTradeBufferBurrow<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.data
    }
}

impl<'a, S: ToSliceMut<T>, T: RealNumber> BufferBorrow<S, T> for NoTradeBufferBurrow<'a, T> {
    fn trade(self, _: &mut S) {
        // No trade
    }
}

/// For internal use only. A buffer which doesn't implement the `swap` routine. Swapping is a no-op in this
/// implementation. This can be useful in cases where an implementation will do the swap step on its own.
struct NoTradeBuffer<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    data: S,
    data_type: std::marker::PhantomData<T>,
}

impl<S, T> NoTradeBuffer<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Creates a new buffer from a storage type. The buffer will internally hold
    /// its storage for it's complete life time.
    pub fn new(storage: S) -> NoTradeBuffer<S, T> {
        NoTradeBuffer {
            data: storage,
            data_type: std::marker::PhantomData,
        }
    }
}

impl<'a, S, T> Buffer<'a, S, T> for NoTradeBuffer<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber + 'a
{
    type Borrow = NoTradeBufferBurrow<'a, T>;

    fn borrow(&'a mut self, len: usize) -> Self::Borrow {
        if self.data.len() < len {
            panic!("NoTradeBuffer: Out of memory");
        }

        NoTradeBufferBurrow { data: &mut self.data.to_slice_mut()[0..len] }
    }

    fn alloc_len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::swap_array_halves;

    #[test]
    fn swap_halves_even_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        swap_array_halves(&mut v, false);
        assert_eq!(&v[..], &[5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn swap_halves_odd_foward_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        swap_array_halves(&mut v, true);
        assert_eq!(&v[..], &[6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn swap_halves_odd_inverse_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        swap_array_halves(&mut v, false);
        assert_eq!(&v[..], &[5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0]);
    }
}
