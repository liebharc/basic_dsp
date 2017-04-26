//! Conversions to and from vectors which serve as constructors.
use numbers::*;
use super::{DataDomain, NumberSpace, Domain, DspVec, GenDspVec,
            RealTimeVec, RealFreqVec, ComplexTimeVec, ComplexFreqVec, RealData, ComplexData,
            TimeData, FrequencyData, ToSlice,
            TypeMetaData};
use multicore_support::MultiCoreSettings;
use std::convert::From;

/// Conversion from a generic data type into a dsp vector which tracks
/// its meta information (domain and number space)
/// only at runtime. See `ToRealVector` and
/// `ToComplexVector` for alternatives which track most of the meta data
/// with the type system and therefore avoid runtime errors.
pub trait ToDspVector<T>: Sized + ToSlice<T>
    where T: RealNumber
{
    /// Create a new generic vector.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    fn to_gen_dsp_vec(self, is_complex: bool, domain: DataDomain) -> GenDspVec<Self, T>;

    /// Create a new vector from the given meta data. The meta data can be
    /// retrieved from an existing vector. If no existing vector is available
    /// then one of the other constructor methods should be used.
    fn to_dsp_vec<N, D>(self, meta_data: &TypeMetaData<T, N, D>) -> DspVec<Self, T, N, D>
        where N: NumberSpace, D: Domain;
}

/// Conversion from a generic data type into a dsp vector with real data.
pub trait ToRealVector<T>: Sized + ToSlice<T>
    where T: RealNumber
{
    /// Create a new vector in real number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_time_vec(self) -> RealTimeVec<Self, T>;

    /// Create a new vector in real number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_freq_vec(self) -> RealFreqVec<Self, T>;
}

/// Conversion from a generic data type into a dsp vector with complex data.
pub trait ToComplexVector<S, T>
    where S: Sized + ToSlice<T>,
          T: RealNumber
{
    /// Create a new vector in complex number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    fn to_complex_time_vec(self) -> ComplexTimeVec<S, T>;

    /// Create a new vector in complex number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    fn to_complex_freq_vec(self) -> ComplexFreqVec<S, T>;
}

/// Retrieves the underlying storage from a vector.
pub trait FromVector<T>
    where T: RealNumber
{
    /// Type of the underlying storage of a vector.
    type Output;

    /// Gets the underlying storage and the number of elements which
    /// contain valid.
    fn get(self) -> (Self::Output, usize);

    /// Gets the underlying slice of a vector.
    fn to_slice(&self) -> &[T];
}

impl<S, T, N, D> FromVector<T> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    type Output = S;

    fn get(self) -> (Self::Output, usize) {
        let len = self.valid_len;
        (self.data, len)
    }

    fn to_slice(&self) -> &[T] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        &slice[0..len]
    }
}

impl<S, T> From<S> for RealTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    fn from(mut data: S) -> Self {
        let len = data.len();
        let alloc = data.alloc_len();
        data.try_resize(alloc).expect("Expanding to alloc_len should always work");
        RealTimeVec {
            data: data,
            delta: T::one(),
            domain: TimeData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<S, T> From<S> for ComplexTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    fn from(mut data: S) -> Self {
        let len = data.len();
        let alloc = data.alloc_len();
        data.try_resize(alloc).expect("Expanding to alloc_len should always work");
        ComplexTimeVec {
            data: data,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}
impl<S, T> From<S> for RealFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    fn from(mut data: S) -> Self {
        let len = data.len();
        let alloc = data.alloc_len();
        data.try_resize(alloc).expect("Expanding to alloc_len should always work");
        RealFreqVec {
            data: data,
            delta: T::one(),
            domain: FrequencyData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<S, T> From<S> for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    fn from(mut data: S) -> Self {
        let len = data.len();
        let alloc = data.alloc_len();
        data.try_resize(alloc).expect("Expanding to alloc_len should always work");
        ComplexFreqVec {
            data: data,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<S, T, N, D> Clone for DspVec<S, T, N, D>
    where S: ToSlice<T> + Clone,
          T: RealNumber,
          N: NumberSpace + Clone,
          D: Domain + Clone
{
    fn clone(&self) -> Self {
        DspVec {
            data: self.data.clone(),
            delta: self.delta,
            domain: self.domain.clone(),
            number_space: self.number_space.clone(),
            valid_len: self.valid_len,
            multicore_settings: self.multicore_settings,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.data = source.data.clone();
        self.delta = source.delta;
        self.domain = source.domain.clone();
        self.number_space = source.number_space.clone();
        self.valid_len = source.valid_len;
        self.multicore_settings = source.multicore_settings;
    }
}
