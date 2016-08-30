//! Conversions to and from vectors which serve as constructors.
use RealNumber;
use vector_types::{
    DataVecDomain};
use super::{
    GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec,
    ToSlice};
    use multicore_support::MultiCoreSettings;

/// Conversion from a generic data type into a dsp vector which tracks
/// its meta information (domain and number space)
/// only at runtime. See `ToRealVector` and
/// `ToComplexVector` for alternatives which track most of the meta data
/// with the type system and therefore avoid runtime errors.
trait ToDspVector<T> : Sized + ToSlice<T>
    where T: RealNumber {
    /// Create a new generic vector.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_gen_dsp(self, domain: DataVecDomain, is_complex: bool) -> GenDspVec<Self, T>;
}

/// Conversion from a generic data type into a dsp vector with real data.
trait ToRealVector<T> : Sized + ToSlice<T>
    where T: RealNumber {
    /// Create a new vector in real number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_time(self) -> RealTimeVec<Self, T>;

    /// Create a new vector in real number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_freq(self) -> RealFreqVec<Self, T>;
}

/// Conversion from a generic data type into a dsp vector with complex data.
trait ToComplexVector<T> : Sized + ToSlice<T>
    where T: RealNumber {
    /// Create a new vector in complex number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_complex_time(self) -> ComplexTimeVec<Self, T>;

    /// Create a new vector in complex number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_complex_freq(self) -> ComplexFreqVec<Self, T>;
}

/// Retrieves the underlying storage from a vector.
trait FromVector<T>
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
    fn to_gen_dsp(self, domain: DataVecDomain, is_complex: bool) -> GenDspVec<Self, T> {
        let len = self.len();
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: domain,
            is_complex: is_complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToRealVector<T> for Vec<T>
    where T: RealNumber {
    fn to_real_time(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Time,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_real_freq(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Frequency,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToComplexVector<T> for Vec<T>
    where T: RealNumber {
    fn to_complex_time(self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Time,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_complex_freq(self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Frequency,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToDspVector<T> for &'a [T]
    where T: RealNumber {
    fn to_gen_dsp(self, domain: DataVecDomain, is_complex: bool) -> GenDspVec<Self, T> {
        let len = self.len();
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: domain,
            is_complex: is_complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToRealVector<T> for &'a [T]
    where T: RealNumber {
    fn to_real_time(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Time,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_real_freq(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Frequency,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToComplexVector<T> for &'a[T]
    where T: RealNumber {
    fn to_complex_time(self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Time,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_complex_freq(self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Frequency,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToDspVector<T> for &'a mut [T]
    where T: RealNumber {
    fn to_gen_dsp(self, domain: DataVecDomain, is_complex: bool) -> GenDspVec<Self, T> {
        let len = self.len();
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: domain,
            is_complex: is_complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToRealVector<T> for &'a mut [T]
    where T: RealNumber {
    fn to_real_time(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Time,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_real_freq(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Frequency,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToComplexVector<T> for &'a mut [T]
    where T: RealNumber {
    fn to_complex_time(self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Time,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_complex_freq(self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Frequency,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToDspVector<T> for Box<[T]>
    where T: RealNumber {
    fn to_gen_dsp(self, domain: DataVecDomain, is_complex: bool) -> GenDspVec<Self, T> {
        let len = self.len();
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: domain,
            is_complex: is_complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToRealVector<T> for Box<[T]>
    where T: RealNumber {
    fn to_real_time(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Time,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_real_freq(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Frequency,
            is_complex: false,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToComplexVector<T> for Box<[T]>
    where T: RealNumber {
    fn to_complex_time(self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Time,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_complex_freq(self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: DataVecDomain::Frequency,
            is_complex: true,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

macro_rules! define_vector_conversions {
    ($($name:ident),*) => {
        $(
            impl<S, T> FromVector<T> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
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
        )*
    }
}
define_vector_conversions!(
    GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec);
