//! Conversions to and from vectors which serve as constructors.
use super::{
    ComplexFreqVec, ComplexTimeVec, DataDomain, Domain, DspVec, GenDspVec, NumberSpace,
    RealFreqVec, RealTimeVec, ToSlice, TypeMetaData,
};
use crate::meta;
use crate::multicore_support::MultiCoreSettings;
use crate::numbers::*;
use std::convert::From;

/// Conversion from a generic data type into a dsp vector which tracks
/// its meta information (domain and number space)
/// only at runtime. See `ToRealVector` and
/// `ToComplexVector` for alternatives which track most of the meta data
/// with the type system and therefore avoid runtime errors.
pub trait ToDspVector<T>: Sized + ToSlice<T>
where
    T: RealNumber,
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
    where
        N: NumberSpace,
        D: Domain;
}

/// Conversion from a generic data type into a dsp vector with real data.
pub trait ToRealVector<T>: Sized + ToSlice<T>
where
    T: RealNumber,
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
where
    S: Sized + ToSlice<T>,
    T: RealNumber,
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

/// Retrieves the underlying storage from a vector. Returned value will always hold floating point numbers.
pub trait FromVectorFloat<T>
where
    T: RealNumber,
{
    /// Type of the underlying storage of a vector.
    type Output;

    /// Gets the underlying storage and the number of elements which
    /// contain valid data. In case of complex vectors the values are returned real-imag pairs. 
    /// Refer to [`Into`](https://doc.rust-lang.org/std/convert/trait.Into.html) or [`FromVector`](trait.FromVector.html#)
    /// for a method which returns the data of complex vectors in a different manner.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// use basic_dsp_vector::*;
    /// # use num_complex::Complex;
    /// let v: Vec<Complex<f64>> = vec!(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let v: DspVec<Vec<f64>, f64, meta::Complex, meta::Time> = v.to_complex_time_vec();
    /// // Note that the resulting type is always a vector or floats and not a vector of complex numbers
    /// let (v, p): (Vec<f64>, usize) = v.getf();
    /// assert_eq!(4, p);
    /// assert_eq!(vec!(1.0, 2.0, 3.0, 4.0), v);
    /// ```
    fn getf(self) -> (Self::Output, usize);
}

/// Retrieves the underlying storage from a vector.
///
/// If you are working with `std::vec::Vec` then it's recommended to use  
/// [`Into`](https://doc.rust-lang.org/std/convert/trait.Into.html#tymethod.into) instead of this one, as
/// it's more straightforward to use.
pub trait FromVector<T>
where
    T: RealNumber,
{
    /// Type of the underlying storage of a vector.
    type Output;

    /// If you are working with `std::vec::Vec` then it's recommended to use  
    /// [`Into`](https://doc.rust-lang.org/std/convert/trait.Into.html#tymethod.into) instead of this one, as
    /// it's more straightforward to use.
    ///
    /// Gets the underlying storage and the number of elements which
    /// contain valid data. Therefore a caller should only use the first `valid data` elements from the storage. 
    /// The remaining elements (if there are any) might have been allocated during the calculations but contain 
    /// no useful information.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// use basic_dsp_vector::*;
    /// # use num_complex::Complex;
    /// let v: Vec<Complex<f64>> = vec!(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let v: DspVec<Vec<f64>, f64, meta::Complex, meta::Time> = v.to_complex_time_vec();
    /// let (v, p): (Vec<Complex<f64>>, usize) = v.get();
    /// assert_eq!(2, p);
    /// assert_eq!(vec!(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)), v);
    /// ```
    fn get(self) -> (Self::Output, usize);
}

impl<S, T, N, D> FromVectorFloat<T> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = S;

    fn getf(self) -> (Self::Output, usize) {
        let len = self.valid_len;
        (self.data, len)
    }
}

impl<S, T, D> FromVector<T> for DspVec<S, T, meta::Real, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = S;

    fn get(self) -> (Self::Output, usize) {
        let len = self.valid_len;
        (self.data, len)
    }
}

impl<S, T> From<S> for RealTimeVec<S, T>
where
    S: ToSlice<T>,
    T: RealNumber,
{
    fn from(mut data: S) -> Self {
        let len = data.len();
        let alloc = data.alloc_len();
        data.try_resize(alloc)
            .expect("Expanding to alloc_len should always work");
        RealTimeVec {
            data,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<S, T> From<S> for ComplexTimeVec<S, T>
where
    S: ToSlice<T>,
    T: RealNumber,
{
    fn from(mut data: S) -> Self {
        let len = data.len();
        let alloc = data.alloc_len();
        data.try_resize(alloc)
            .expect("Expanding to alloc_len should always work");
        ComplexTimeVec {
            data,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}
impl<S, T> From<S> for RealFreqVec<S, T>
where
    S: ToSlice<T>,
    T: RealNumber,
{
    fn from(mut data: S) -> Self {
        let len = data.len();
        let alloc = data.alloc_len();
        data.try_resize(alloc)
            .expect("Expanding to alloc_len should always work");
        RealFreqVec {
            data,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<S, T> From<S> for ComplexFreqVec<S, T>
where
    S: ToSlice<T>,
    T: RealNumber,
{
    fn from(mut data: S) -> Self {
        let len = data.len();
        let alloc = data.alloc_len();
        data.try_resize(alloc)
            .expect("Expanding to alloc_len should always work");
        ComplexFreqVec {
            data,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<S, T, N, D> Clone for DspVec<S, T, N, D>
where
    S: ToSlice<T> + Clone,
    T: RealNumber,
    N: NumberSpace + Clone,
    D: Domain + Clone,
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
