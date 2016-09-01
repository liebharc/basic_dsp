//! Specifies the conversions between data types.
use RealNumber;
use super::{
    DspVec, GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec,
    ToSlice, NumberSpace, Domain,
    RealData, ComplexData, RealOrComplexData,
    TimeData, FrequencyData, TimeOrFrequencyData
};

/// This trait allows to change a data type. The operations will
/// convert a type to a different one and set `self.len()` to zero.
/// However `self.allocated_len()` will remain unchanged. The use case for this
/// is to allow to reuse the memory of a vector for different operations.
///
/// If a type should always be converted without any checks then the `RededicateForceOps`
/// trait provides option for that.
pub trait RededicateOps<Other> : RededicateForceOps<Other> {
    /// Make `Other` a `Self`.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::{ComplexFreqVector32, ComplexTimeVector32, ComplexVectorOps, RededicateOps, DataVec, DataVecDomain};
    /// let complex = ComplexFreqVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let real = complex.phase().expect("Ignoring error handling in examples");
    /// let complex = ComplexTimeVector32::rededicate_from(real);
    /// assert_eq!(true, complex.is_complex());
    /// assert_eq!(DataVecDomain::Time, complex.domain());
    /// assert_eq!(0, complex.len());
    /// assert_eq!(2, complex.allocated_len());
    /// ```
    fn rededicate_from(origin: Other) -> Self;
}

/// This trait allows to change a data type and performs the Conversion
/// without any checks. `RededicateOps` provides the same functionality
/// but performs runtime checks to avoid that data is interpreted the wrong
/// way.
pub trait RededicateForceOps<Other> {
    /// Make `Other` a `Self` without performing any checks.
    fn rededicate_from_force(origin: Other) -> Self;
}

/// Specifies what the the result is if a type is transformed to real numbers.
pub trait ToRealResult {
    type RealResult;
}

/// Specifies what the the result is if a type is transformed to complex numbers.
pub trait ToComplexResult {
    type ComplexResult;
}

pub trait ToTimeResult {
    /// Specifies what the the result is if a type is transformed to time domain.
    type TimeResult;
}

/// Specifies what the the result is if a type is transformed to frequency domain.
pub trait ToFreqResult {
    type FreqResult;
}

impl<S, T> ToRealResult for ComplexTimeVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber {
    type RealResult = RealTimeVec<S, T>;
}

impl<S, T> ToRealResult for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type RealResult = RealFreqVec<S, T>;
}

impl<S, T> ToRealResult for GenDspVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type RealResult = GenDspVec<S, T>;
}

impl<S, T> ToComplexResult for RealTimeVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type ComplexResult = ComplexTimeVec<S, T>;
}

impl<S, T> ToComplexResult for RealFreqVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type ComplexResult = ComplexFreqVec<S, T>;
}

impl<S, T> ToComplexResult for GenDspVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type ComplexResult = GenDspVec<S, T>;
}

impl<S, T> ToTimeResult for RealFreqVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type TimeResult = RealTimeVec<S, T>;
}

impl<S, T> ToTimeResult for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type TimeResult = ComplexTimeVec<S, T>;
}

impl<S, T> ToTimeResult for GenDspVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type TimeResult = GenDspVec<S, T>;
}

impl<S, T> ToFreqResult for RealTimeVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type FreqResult = RealFreqVec<S, T>;
}

impl<S, T> ToFreqResult for ComplexTimeVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type FreqResult = ComplexFreqVec<S, T>;
}

impl<S, T> ToFreqResult for GenDspVec<S, T>
    where S: ToSlice<T>,
      T: RealNumber  {
    type FreqResult = GenDspVec<S, T>;
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for RealTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain {

    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        RealTimeVec {
            data: origin.data,
            delta: origin.delta,
            domain: TimeData,
            number_space: RealData,
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings
        }
    }
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for RealFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain {

    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        RealFreqVec {
            data: origin.data,
            delta: origin.delta,
            domain: FrequencyData,
            number_space: RealData,
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings
        }
    }
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for ComplexTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain {

    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        ComplexTimeVec {
            data: origin.data,
            delta: origin.delta,
            domain: TimeData,
            number_space: ComplexData,
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings
        }
    }
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain {

    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        ComplexFreqVec {
            data: origin.data,
            delta: origin.delta,
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings
        }
    }
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for GenDspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain {

    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        let domain = origin.domain();
        let is_complex = origin.is_complex();
        GenDspVec {
            data: origin.data,
            delta: origin.delta,
            domain: TimeOrFrequencyData { domain_current: domain },
            number_space: RealOrComplexData { is_complex_current: is_complex },
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings
        }
    }
}
