//! Conversions to and from vectors which serve as constructors.
use RealNumber;
use super::{
    DataDomain,
    NumberSpace,
    Domain,
    DspVec, GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec,
    ToSlice};
use multicore_support::MultiCoreSettings;
use std::marker::PhantomData;

/// Conversion from a generic data type into a dsp vector which tracks
/// its meta information (domain and number space)
/// only at runtime. See `ToRealVector` and
/// `ToComplexVector` for alternatives which track most of the meta data
/// with the type system and therefore avoid runtime errors.
pub trait ToDspVector<T> : Sized + ToSlice<T>
    where T: RealNumber {
    /// Create a new generic vector.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    fn to_gen_dsp_vec(self, domain: DataDomain, is_complex: bool) -> GenDspVec<Self, T>;
}

/// Conversion from a generic data type into a dsp vector with real data.
pub trait ToRealVector<T> : Sized + ToSlice<T>
    where T: RealNumber {
    /// Create a new vector in real number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_time_vec(self) -> RealTimeVec<Self, T>;

    /// Create a new vector in real number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_freq_vec(self) -> RealFreqVec<Self, T>;
}

/// Conversion from a generic data type into a dsp vector with complex data.
pub trait ToComplexVector<T> : Sized + ToSlice<T>
    where T: RealNumber {
    /// Create a new vector in complex number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T>;

    /// Create a new vector in complex number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T>;
}

/// Retrieves the underlying storage from a vector.
pub trait FromVector<T>
    where T: RealNumber {
    /// Type of the underlying storage of a vector.
    type Output;

    /// Gets the underlying storage and the number of elements which
    /// contain valid.
    fn get(self) -> (Self::Output, usize);

    /// Gets the underlying slice of a vector.
    fn to_slice(&self) -> &[T];
}

impl<T> ToDspVector<T> for Vec<T>
    where T: RealNumber {
    fn to_gen_dsp_vec(self, domain: DataDomain, is_complex: bool) -> GenDspVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 && is_complex {
            len = 0;
        }
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: domain,
            is_complex: is_complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<T> ToRealVector<T> for Vec<T>
    where T: RealNumber {
    fn to_real_time_vec(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Time,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Frequency,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<T> ToComplexVector<T> for Vec<T>
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Time,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Frequency,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<'a, T> ToDspVector<T> for &'a [T]
    where T: RealNumber {
    fn to_gen_dsp_vec(self, domain: DataDomain, is_complex: bool) -> GenDspVec<Self, T> {
        let len = self.len();
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: domain,
            is_complex: is_complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<'a, T> ToRealVector<T> for &'a [T]
    where T: RealNumber {
    fn to_real_time_vec(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Time,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Frequency,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<'a, T> ToComplexVector<T> for &'a[T]
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Time,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Frequency,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<'a, T> ToDspVector<T> for &'a mut [T]
    where T: RealNumber {
    fn to_gen_dsp_vec(self, domain: DataDomain, is_complex: bool) -> GenDspVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 && is_complex {
            len = 0;
        }
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: domain,
            is_complex: is_complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<'a, T> ToRealVector<T> for &'a mut [T]
    where T: RealNumber {
    fn to_real_time_vec(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Time,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Frequency,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<'a, T> ToComplexVector<T> for &'a mut [T]
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Time,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Frequency,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<T> ToDspVector<T> for Box<[T]>
    where T: RealNumber {
    fn to_gen_dsp_vec(self, domain: DataDomain, is_complex: bool) -> GenDspVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 && is_complex {
            len = 0;
        }
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: domain,
            is_complex: is_complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<T> ToRealVector<T> for Box<[T]>
    where T: RealNumber {
    fn to_real_time_vec(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Time,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Frequency,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<T> ToComplexVector<T> for Box<[T]>
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Time,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: DataDomain::Frequency,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
            _static_number_space: PhantomData,
            _static_domain: PhantomData
        }
    }
}

impl<S, T, N, D> FromVector<T> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain {
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
