use RealNumber;
use multicore_support::{
    Chunk,
    MultiCoreSettings,
    Complexity};
use num::complex::Complex;
use std::mem;
use std::cmp;
use std::result;
use simd_extensions::*;
 use std::fmt::Debug;

mod requirements;
pub use self::requirements::*;
mod to_from_vec_conversions;
pub use self::to_from_vec_conversions::*;
mod vec_impl_and_indexers;
pub use self::vec_impl_and_indexers::*;
mod complex_to_real;
pub use self::complex_to_real::*;
mod rededicate_and_relations;
pub use self::rededicate_and_relations::*;
mod checks_and_results;
pub use self::checks_and_results::*;
mod elementary;
pub use self::elementary::*;
mod trigonometry;
pub use self::trigonometry::*;
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
    /// Time domain, the x-axis is in [s]
    Time,
    /// Frequency domain, the x-axis in [Hz]
    Frequency
}

/// Number space (real or complex) information.
pub trait NumberSpace : Debug + cmp::PartialEq {
    fn is_complex(&self) -> bool;
}

/// Domain (time or frequency) information.
pub trait Domain : Debug + cmp::PartialEq {
    fn domain(&self) -> DataDomain;
}

/// Trait for types containing real data.
pub trait RealNumberSpace : NumberSpace {
    /// For implementations which track meta data
    /// at runtime this method may be implemented to transition
    /// between different states. For all other implementations
    /// they may leave this empty.
    fn to_complex(&mut self);
}

/// TWorait for types containing complex data.
pub trait ComplexNumberSpace : NumberSpace {
    /// See `RealNumberSpace` for some more details.
    fn to_real(&mut self);
}

/// Trait for types containing time domain data.
pub trait TimeDomain {
    /// See `RealNumberSpace` for some more details.
    fn to_freq(&mut self);
}

/// Trait for types containing frequency domain data.
pub trait FrequencyDomain {
    /// See `RealNumberSpace` for some more details.
    fn to_time(&mut self);
}

/// Marker for types containing real data.
#[derive(Debug)]
#[derive(PartialEq)]
pub struct RealData;
impl NumberSpace for RealData {
    fn is_complex(&self) -> bool { false }
}
impl RealNumberSpace for RealData {
    fn to_complex(&mut self) { }
}

/// Marker for types containing complex data.
#[derive(Debug)]
#[derive(PartialEq)]
pub struct ComplexData;
impl NumberSpace for ComplexData {
    fn is_complex(&self) -> bool { true }
}
impl ComplexNumberSpace for ComplexData {
    fn to_real(&mut self) { }
}

/// Marker for types containing real or complex data.
#[derive(Debug)]
#[derive(PartialEq)]
pub struct RealOrComplexData {
    is_complex_current: bool
}
impl NumberSpace for RealOrComplexData {
    fn is_complex(&self) -> bool { self.is_complex_current }
}
impl RealNumberSpace for RealOrComplexData {
    fn to_complex(&mut self) {
        self.is_complex_current = true;
    }
}
impl ComplexNumberSpace for RealOrComplexData {
    fn to_real(&mut self) {
        self.is_complex_current = false;
    }
}

/// Marker for types containing time data.
#[derive(Debug)]
#[derive(PartialEq)]
pub struct TimeData;
impl Domain for TimeData {
    fn domain(&self) -> DataDomain { DataDomain::Time }
}
impl TimeDomain for TimeData {
    fn to_freq(&mut self) { }
 }

/// Marker for types containing frequency data.
#[derive(Debug)]
#[derive(PartialEq)]
pub struct FrequencyData;
impl Domain for FrequencyData {
    fn domain(&self) -> DataDomain { DataDomain::Frequency }
}
impl FrequencyDomain for FrequencyData {
    fn to_time(&mut self) { }
}

/// Marker for types containing time or frequency data.
#[derive(Debug)]
#[derive(PartialEq)]
pub struct TimeOrFrequencyData {
    domain_current: DataDomain
}
impl Domain for TimeOrFrequencyData {
    fn domain(&self) -> DataDomain { self.domain_current }
}

impl TimeDomain for TimeOrFrequencyData {
    fn to_freq(&mut self) {
        self.domain_current = DataDomain::Frequency;
    }
}
impl FrequencyDomain for TimeOrFrequencyData {
    fn to_time(&mut self) {
        self.domain_current = DataDomain::Time;
    }
}

/// A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations.
/// All data vector operations consume the vector they operate on and return a new vector. A consumed vector
/// must not be accessed again.
///
/// Vectors come in different flavors:
///
/// 1. Time or Frequency domain
/// 2. Real or Complex numbers
/// 3. 32bit or 64bit floating point numbers
///
/// The first two flavors define meta information about the vector and provide compile time information what
/// operations are available with the given vector and how this will transform the vector. This makes sure that
/// some invalid operations are already discovered at compile time. In case that this isn't desired or the information
/// about the vector isn't known at compile time there are the generic [`DataVec32`](type.DataVec32.html) and [`DataVec64`](type.DataVec64.html) vectors
/// available.
///
/// 32bit and 64bit flavors trade performance and memory consumption against accuracy. 32bit vectors are roughly
/// two times faster than 64bit vectors for most operations. But remember that you should benchmark first
/// before you give away accuracy for performance unless however you are sure that 32bit accuracy is certainly good
/// enough.
#[derive(Debug)]
pub struct DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain {
    data: S,
    delta: T,
    domain: D,
    number_space: N,
    valid_len: usize,
    multicore_settings: MultiCoreSettings
}

/// A vector with real numbers in time domain.
pub type RealTimeVec<S, T> = DspVec<S, T, RealData, TimeData>;
/// A vector with real numbers in frequency domain.
pub type RealFreqVec<S, T> = DspVec<S, T, RealData, FrequencyData>;
/// A vector with complex numbers in time domain.
pub type ComplexTimeVec<S, T> = DspVec<S, T, ComplexData, TimeData>;
/// A vector with complex numbers in frequency domain.
pub type ComplexFreqVec<S, T> = DspVec<S, T, ComplexData, FrequencyData>;
/// A vector with no information about number space or domain at compile time.
pub type GenDspVec<S, T> = DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>;

fn array_to_complex<T>(array: &[T]) -> &[Complex<T>] {
    unsafe {
        let len = array.len();
        if len % 2 != 0 {
            panic!("Argument must have an even length");
        }
        let trans: &[Complex<T>] = mem::transmute(array);
        &trans[0 .. len / 2]
    }
}

fn array_to_complex_mut<T>(array: &mut [T]) -> &mut [Complex<T>] {
    unsafe {
        let len = array.len();
        if len % 2 != 0 {
            panic!("Argument must have an even length");
        }
        let trans: &mut [Complex<T>] = mem::transmute(array);
        &mut trans[0 .. len / 2]
    }
}

impl<S, T, N, D> DspVec<S, T, N, D> where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain {
    #[inline]
    fn pure_real_operation<A, F>(&mut self, op: F, argument: A, complexity: Complexity)
        where A: Sync + Copy + Send,
            F: Fn(T, A) -> T + 'static + Sync {
        {
            let len = self.valid_len;
            let mut array = self.data.to_slice_mut();
            Chunk::execute_partial(
                complexity, &self.multicore_settings,
                &mut array[0..len], 1, argument,
                move|array, argument| {
                    for num in array {
                        *num = op(*num, argument);
                    }
            });
        }
    }

    #[inline]
    fn pure_complex_operation<A, F>(&mut self, op: F, argument: A, complexity: Complexity)
        where A: Sync + Copy + Send,
            F: Fn(Complex<T>, A) -> Complex<T> + 'static + Sync {
        {
            let len = self.valid_len;
            let mut array = self.data.to_slice_mut();
            Chunk::execute_partial(
                complexity, &self.multicore_settings,
                &mut array[0..len], 2, argument,
                move|array, argument| {
                    let array = array_to_complex_mut(array);
                    for num in array {
                        *num = op(*num, argument);
                    }
            });
        }
    }

    #[inline]
    fn simd_real_operation<A, F, G>(&mut self, simd_op: F, scalar_op: G, argument: A, complexity: Complexity)
        where A: Sync + Copy + Send,
                F: Fn(T::Reg, A) -> T::Reg + 'static + Sync,
                G: Fn(T, A) -> T + 'static + Sync {
        {
            let data_length = self.valid_len;
            let mut array = self.data.to_slice_mut();
            let (scalar_left, scalar_right, vectorization_length) = T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
            if vectorization_length > 0 {
                Chunk::execute_partial(
                    complexity, &self.multicore_settings,
                    &mut array[scalar_left..vectorization_length], T::Reg::len(), argument,
                    move |array, argument| {
                    let array = T::Reg::array_to_regs_mut(array);
                    for reg in array {
                        *reg = simd_op(*reg, argument);
                    }
                });
            }
            for num in &mut array[0..scalar_left]
            {
                *num = scalar_op(*num, argument);
            }
            for num in &mut array[scalar_right..data_length]
            {
                *num = scalar_op(*num, argument);
            }
        }
    }

    #[inline]
    fn simd_complex_operation<A, F, G>(&mut self, simd_op: F, scalar_op: G, argument: A, complexity: Complexity)
        where A: Sync + Copy + Send,
                F: Fn(T::Reg, A) -> T::Reg + 'static + Sync,
                G: Fn(Complex<T>, A) -> Complex<T> + 'static + Sync {
        {
            let data_length = self.valid_len;
            let mut array = self.data.to_slice_mut();
            let (scalar_left, scalar_right, vectorization_length) = T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
            if vectorization_length > 0 {
                Chunk::execute_partial(
                    complexity, &self.multicore_settings,
                    &mut array[scalar_left..vectorization_length], T::Reg::len(), argument,
                    move |array, argument| {
                    let array = T::Reg::array_to_regs_mut(array);
                    for reg in array {
                        *reg = simd_op(*reg, argument);
                    }
                });
            }
            {
                let array = array_to_complex_mut(&mut array[0..scalar_left]);
                for num in array
                {
                    *num = scalar_op(*num, argument);
                }
            }
            {
                let array = array_to_complex_mut(&mut array[scalar_right..data_length]);
                for num in array
                {
                    *num = scalar_op(*num, argument);
                }
            }
        }
    }

    #[inline]
    fn pure_complex_to_real_operation<A, F, B>(&mut self, buffer: &mut B, op: F, argument: A, complexity: Complexity)
        where A: Sync + Copy + Send,
            F: Fn(Complex<T>, A) -> T + 'static + Sync,
            B: Buffer<S, T> {
        {
            let data_length = self.len();
            let mut result = buffer.get(data_length / 2);
            {
                let mut array = self.data.to_slice_mut();
                let mut temp = result.to_slice_mut();
                Chunk::from_src_to_dest(
                    complexity, &self.multicore_settings,
                    &mut array[0..data_length], 2,
                    &mut temp[0..data_length / 2], 1, argument,
                    move |array, range, target, argument| {
                        let array = array_to_complex(&array[range.start..range.end]);
                        for pair in array.iter().zip(target) {
                            let (src, dest) = pair;
                            *dest = op(*src, argument);
                        }
                });
                self.valid_len = data_length / 2;
            }
            mem::swap(&mut self.data, &mut result);
            buffer.free(result);
        }
    }

    #[inline]
    fn simd_complex_to_real_operation<A, F, G, B>(&mut self, buffer: &mut B, simd_op: F, scalar_op: G, argument: A, complexity: Complexity)
        where A: Sync + Copy + Send,
              F: Fn(T::Reg, A) -> T::Reg + 'static + Sync,
              G: Fn(Complex<T>, A) -> T + 'static + Sync,
              B: Buffer<S, T> {
        let data_length = self.len();
        let mut result = buffer.get(data_length / 2);
        {
            let mut array = self.data.to_slice_mut();
            let mut temp = result.to_slice_mut();
            let (scalar_left, scalar_right, vectorization_length) = T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
            if vectorization_length > 0 {
                Chunk::from_src_to_dest(
                complexity, &self.multicore_settings,
                &mut array[scalar_left..vectorization_length], T::Reg::len(),
                &mut temp[scalar_left/2..vectorization_length/2], T::Reg::len() / 2, argument,
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
                for pair in array.iter().zip(&mut temp[0..scalar_left/2]) {
                    let (src, dest) = pair;
                    *dest = scalar_op(*src, argument);
                }
            }
            {
                let array = array_to_complex(&array[scalar_right..data_length]);
                for pair in array.iter().zip(&mut temp[scalar_right/2..data_length/2]) {
                    let (src, dest) = pair;
                    *dest = scalar_op(*src, argument);
                }
            }

            self.valid_len = self.valid_len / 2;
        }
        mem::swap(&mut self.data, &mut result);
        buffer.free(result);
    }
}
