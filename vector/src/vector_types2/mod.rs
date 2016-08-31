use RealNumber;

mod requirements;
pub use self::requirements::*;
mod to_from_vec_conversions;
pub use self::to_from_vec_conversions::*;
mod vec_impl_and_indexers;
pub use self::vec_impl_and_indexers::*;
use vector_types::{
    DataVecDomain};
use multicore_support::MultiCoreSettings;
use std::marker::PhantomData;

/// Marker trait for types containing real data.
pub trait RealDataMarker { }

/// Marker trait for types containing complex data.
pub trait ComplexDataMarker { }

/// Marker trait for types containing time domain data.
pub trait TimeDataMarker { }

/// Marker trait for types containing frequency domain data.
pub trait FrequencyDataMarker { }

/// Marker for types containing real data.
pub struct RealData { }
impl RealDataMarker for RealData { }

/// Marker for types containing complex data.
pub struct ComplexData { }
impl ComplexDataMarker for ComplexData { }

/// Marker for types containing real or complex data.
pub struct RealOrComplexData { }
impl RealDataMarker for RealOrComplexData { }
impl ComplexDataMarker for RealOrComplexData { }

/// Marker for types containing time data.
pub struct TimeData { }
impl TimeDataMarker for TimeData { }

/// Marker for types containing frequency data.
pub struct FrequencyData { }
impl FrequencyDataMarker for FrequencyData { }

/// Marker for types containing time or frequency data.
pub struct TimeOrFrequencyData { }
impl TimeDataMarker for TimeOrFrequencyData { }
impl FrequencyDataMarker for TimeOrFrequencyData { }

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
pub struct DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber {
    data: S,
    delta: T,
    domain: DataVecDomain,
    is_complex: bool,
    valid_len: usize,
    multicore_settings: MultiCoreSettings,
    _static_number_space: PhantomData<*const N>,
    _static_domain: PhantomData<*const D>
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
