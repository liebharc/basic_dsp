//! Conversions to and from vectors which serve as constructors.
use RealNumber;
use num::{Complex, Zero};
use std::result;
use super::{
    round_len,
    DataDomain,
    NumberSpace,
    Domain, ErrorReason,
    DspVec, GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec,
    RealData, ComplexData, RealOrComplexData,
    TimeData, FrequencyData, TimeOrFrequencyData,
    ToSlice};
use multicore_support::MultiCoreSettings;
use std::convert::From;

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
pub trait ToComplexVector<S, T>
    where S: Sized + ToSlice<T>,
          T: RealNumber {
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

/// Conversion from two instances of a generic data type into a dsp vector with complex data.
pub trait InterleaveToVector<T> : ToSlice<T>
    where T: RealNumber {
    /// Create a new vector in real number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn interleave_to_complex_time_vec(&self, other: &Self) -> result::Result<ComplexTimeVec<Vec<T>, T>, ErrorReason>;

    /// Create a new vector in real number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn interleave_to_complex_freq_vec(&self, other: &Self) -> result::Result<ComplexFreqVec<Vec<T>, T>, ErrorReason>;
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
            domain: TimeOrFrequencyData { domain_current: domain },
            number_space: RealOrComplexData { is_complex_current: is_complex },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToRealVector<T> for Vec<T>
    where T: RealNumber {
    fn to_real_time_vec(mut self) -> RealTimeVec<Self, T> {
        let len = self.len();
        expand_to_full_capacity(&mut self);
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_real_freq_vec(mut self) -> RealFreqVec<Self, T> {
        let len = self.len();
        expand_to_full_capacity(&mut self);
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToComplexVector<Vec<T>, T> for Vec<T>
    where T: RealNumber {
    fn to_complex_time_vec(mut self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        expand_to_full_capacity(&mut self);
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_complex_freq_vec(mut self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        expand_to_full_capacity(&mut self);
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToComplexVector<Vec<T>, T> for Vec<Complex<T>>
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<Vec<T>, T> {
        let len = self.len();
        let vec = complex_vec_to_interleaved_vec(self);
        ComplexTimeVec {
            data: vec,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: 2 * len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Vec<T>, T> {
        let len = self.len();
        let vec = complex_vec_to_interleaved_vec(self);
        ComplexFreqVec {
            data: vec,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: 2 * len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: TimeOrFrequencyData { domain_current: domain },
            number_space: RealOrComplexData { is_complex_current: is_complex },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: TimeData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToComplexVector<&'a[T], T> for &'a[T]
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToComplexVector<&'a [T], T> for &'a [Complex<T>]
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<&'a [T], T> {
        let array = complex_to_array(self);
        let len = array.len();
        ComplexTimeVec {
            data: array,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<&'a [T], T> {
        let array = complex_to_array(self);
        let len = array.len();
        ComplexFreqVec {
            data: array,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: TimeOrFrequencyData { domain_current: domain },
            number_space: RealOrComplexData { is_complex_current: is_complex },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: TimeData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToComplexVector<&'a mut [T], T> for &'a mut [T]
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<'a, T> ToComplexVector<&'a mut [T], T> for &'a mut [Complex<T>]
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<&'a mut [T], T> {
        let array = complex_to_array_mut(self);
        let len = array.len();
        ComplexTimeVec {
            data: array,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<&'a mut [T], T> {
        let array = complex_to_array_mut(self);
        let len = array.len();
        ComplexFreqVec {
            data: array,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: TimeOrFrequencyData { domain_current: domain },
            number_space: RealOrComplexData { is_complex_current: is_complex },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: TimeData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<T> ToComplexVector<Box<[T]>, T> for Box<[T]>
    where T: RealNumber {
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default()
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

impl<S, T> From<S> for RealTimeVec<S, T>
    where S: ToSlice<T>,
        T: RealNumber {
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
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<S, T> From<S> for ComplexTimeVec<S, T>
    where S: ToSlice<T>,
        T: RealNumber {
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
            multicore_settings: MultiCoreSettings::default()
        }
    }
}
impl<S, T> From<S> for RealFreqVec<S, T>
    where S: ToSlice<T>,
        T: RealNumber {
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
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<S, T> From<S> for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
        T: RealNumber {
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
            multicore_settings: MultiCoreSettings::default()
        }
    }
}

impl<Type, T> InterleaveToVector<T> for Type
    where Type: ToSlice<T>,
          T: RealNumber {
    fn interleave_to_complex_time_vec(&self, other: &Self) -> result::Result<ComplexTimeVec<Vec<T>, T>, ErrorReason> {
        if self.len() != other.len() {
            return Err(ErrorReason::InputMustHaveTheSameSize);
        }

        let rounded_len = round_len(self.len() + other.len());
        let mut data = Vec::with_capacity(rounded_len);

        let len = self.len();
        let real = self.to_slice();
        let imag  = other.to_slice();
        for i in 0 .. len {
            data.push(real[i]);
            data.push(imag[i]);
        }

        let data_length = data.len();

        Ok(ComplexTimeVec {
            data: data,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: data_length,
            multicore_settings: MultiCoreSettings::default()
        })
    }

    fn interleave_to_complex_freq_vec(&self, other: &Self) -> result::Result<ComplexFreqVec<Vec<T>, T>, ErrorReason> {
        if self.len() != other.len() {
            return Err(ErrorReason::InputMustHaveTheSameSize);
        }

        let rounded_len = round_len(self.len() + other.len());
        let mut data = Vec::with_capacity(rounded_len);

        let len = self.len();
        let real = self.to_slice();
        let imag  = other.to_slice();
        for i in 0 .. len {
            data.push(real[i]);
            data.push(imag[i]);
        }

        let data_length = data.len();

        Ok(ComplexFreqVec {
            data: data,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: data_length,
            multicore_settings: MultiCoreSettings::default()
        })
    }
}

impl<S, T, N, D> Clone for DspVec<S, T, N, D>
    where S: ToSlice<T> + Clone,
          T: RealNumber,
          N: NumberSpace + Clone,
          D: Domain + Clone {
    fn clone(&self) -> Self {
        DspVec
        {
          data: self.data.clone(),
          delta: self.delta.clone(),
          domain: self.domain.clone(),
          number_space: self.number_space.clone(),
          valid_len: self.valid_len,
          multicore_settings: self.multicore_settings.clone()
        }
    }

    fn clone_from(&mut self, source: &Self) {
         self.data = source.data.clone();
         self.delta = source.delta.clone();
         self.domain = source.domain.clone();
         self.number_space = source.number_space.clone();
         self.valid_len = source.valid_len;
         self.multicore_settings = source.multicore_settings.clone();
    }
}

fn complex_to_array<T>(complex: &[Complex<T>]) -> &[T]
    where T: RealNumber {
    use std::slice;
    use std::mem;
    let data = unsafe {
        let len = complex.len();
        let trans: &[T] = mem::transmute(complex);
        slice::from_raw_parts(&trans[0] as *const T, len * 2)
    };
    data
}

fn complex_to_array_mut<T>(complex: &mut [Complex<T>]) -> &mut [T]
    where T: RealNumber {
    use std::slice;
    use std::mem;
    let data = unsafe {
        let len = complex.len();
        let mut trans: &mut [T] = mem::transmute(complex);
        slice::from_raw_parts_mut(&mut trans[0] as *mut T, len * 2)
    };
    data
}

fn expand_to_full_capacity<T>(vec: &mut Vec<T>)
    where T: Zero {
    while vec.len() < vec.capacity() {
        vec.push(T::zero());
    }
}

fn complex_vec_to_interleaved_vec<T>(mut vec: Vec<Complex<T>>) -> Vec<T>
    where T: RealNumber {
    use std::mem;

    expand_to_full_capacity(&mut vec);
    let boxed = vec.into_boxed_slice();
    let len = boxed.len();
    let data = unsafe {
        let mut trans: Box<[T]> = mem::transmute(boxed);
        let vec = Vec::<T>::from_raw_parts(&mut trans[0] as *mut T, len * 2, len * 2);
        mem::forget(trans); // TODO memory leak?
        vec
    };
    data
 }

#[cfg(test)]
mod tests {
    use num::complex::Complex32;
    use super::complex_vec_to_interleaved_vec;

    #[test]
    fn complex_vec_to_interleaved_vec_test() {
        let complex = vec!(Complex32::new(0.0, 0.0); 5);
        let real = complex_vec_to_interleaved_vec(complex);
        assert_eq!(real.len(), 10);
        assert_eq!(&real[..], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

}
