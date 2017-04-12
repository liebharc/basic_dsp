//! Specifies the conversions between data types.
use traits::*;
use super::{DspVec, GenDspVec, MetaData, RealTimeVec, RealFreqVec, ResizeOps, ComplexTimeVec,
            ComplexFreqVec, ToSlice, NumberSpace, Domain, DataDomain, RealData, ComplexData,
            RealOrComplexData, TimeData, FrequencyData, TimeOrFrequencyData};

/// This trait allows to change a data type. The operations will
/// convert a type to a different one and set `self.len()` to zero.
/// However `self.allocated_len()` will remain unchanged. The use case for this
/// is to allow to reuse the memory of a vector for different operations.
///
/// If a type should always be converted without any checks then the `RededicateForceOps`
/// trait provides option for that.
pub trait RededicateOps<Other>: RededicateForceOps<Other> {
    /// Make `Other` a `Self`.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let complex = vec!(1.0, 2.0, 3.0, 4.0).to_complex_freq_vec();
    /// let real = complex.phase();
    /// let complex = ComplexTimeVec32::rededicate_from(real);
    /// assert_eq!(true, complex.is_complex());
    /// assert_eq!(DataDomain::Time, complex.domain());
    /// assert_eq!(0, complex.len());
    /// assert_eq!(4, complex.alloc_len());
    /// ```
    fn rededicate_from(origin: Other) -> Self;
}

/// This trait allows to change a data type and performs the Conversion
/// without any checks. `RededicateOps` provides the same functionality
/// but performs runtime checks to avoid that data is interpreted the wrong
/// way.
///
/// In almost all cases this trait shouldn't be used directly.
pub trait RededicateForceOps<Other> {
    /// Make `Other` a `Self` without performing any checks.
    fn rededicate_from_force(origin: Other) -> Self;

    /// Make `Other` a `Self` without performing any checks.
    ///
    /// Try to set the domain and number space. There is no guarantee
    /// that this will succeed, since some rededication targets only
    /// support one domain and number space value. Failures will
    /// be silenty ignored (which is by design).
    fn rededicate_with_runtime_data(origin: Other, is_complex: bool, domain: DataDomain) -> Self;
}

/// This trait allows to change a data type. The operations will
/// convert a type to a different one and set `self.len()` to zero.
/// However `self.allocated_len()` will remain unchanged. The use case for this
/// is to allow to reuse the memory of a vector for different operations.
pub trait RededicateToOps<Other> {
    /// Make `Selfr` a `SelOther`.
    fn rededicate(self) -> Other;
}

/// Specifies what the the result is if a type is transformed to real numbers.
pub trait ToRealResult {
    type RealResult;
}

/// Specifies what the the result is if a type is transformed to complex numbers.
pub trait ToComplexResult {
    type ComplexResult;
}

/// Specifies what the the result is if a type is transformed to time domain.
pub trait ToTimeResult {
    /// Specifies what the the result is if a type is transformed to time domain.
    type TimeResult;
}

/// Specifies what the the result is if a type is transformed to frequency domain.
pub trait ToFreqResult {
    type FreqResult;
}

/// Specifies what the the result is if a type is transformed to real numbers in time domain.
pub trait ToRealTimeResult {
    type RealTimeResult;
}

impl<S, T> ToRealResult for ComplexTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type RealResult = RealTimeVec<S, T>;
}

impl<S, T> ToRealResult for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type RealResult = RealFreqVec<S, T>;
}

impl<S, T> ToRealResult for GenDspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type RealResult = GenDspVec<S, T>;
}

impl<S, T> ToComplexResult for RealTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type ComplexResult = ComplexTimeVec<S, T>;
}

impl<S, T> ToComplexResult for RealFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type ComplexResult = ComplexFreqVec<S, T>;
}

impl<S, T> ToComplexResult for GenDspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type ComplexResult = GenDspVec<S, T>;
}

impl<S, T> ToTimeResult for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type TimeResult = ComplexTimeVec<S, T>;
}

impl<S, T> ToTimeResult for GenDspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type TimeResult = GenDspVec<S, T>;
}

impl<S, T> ToFreqResult for RealTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type FreqResult = ComplexFreqVec<S, T>;
}

impl<S, T> ToFreqResult for ComplexTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type FreqResult = ComplexFreqVec<S, T>;
}

impl<S, T> ToFreqResult for GenDspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type FreqResult = GenDspVec<S, T>;
}

impl<S, T> ToRealTimeResult for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type RealTimeResult = RealTimeVec<S, T>;
}

impl<S, T> ToRealTimeResult for GenDspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber
{
    type RealTimeResult = GenDspVec<S, T>;
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for RealTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        RealTimeVec {
            data: origin.data,
            delta: origin.delta,
            domain: TimeData,
            number_space: RealData,
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings,
        }
    }

    fn rededicate_with_runtime_data(origin: DspVec<S, T, N, D>, _: bool, _: DataDomain) -> Self {
        Self::rededicate_from_force(origin)
    }
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for RealFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        RealFreqVec {
            data: origin.data,
            delta: origin.delta,
            domain: FrequencyData,
            number_space: RealData,
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings,
        }
    }

    fn rededicate_with_runtime_data(origin: DspVec<S, T, N, D>, _: bool, _: DataDomain) -> Self {
        Self::rededicate_from_force(origin)
    }
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for ComplexTimeVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        ComplexTimeVec {
            data: origin.data,
            delta: origin.delta,
            domain: TimeData,
            number_space: ComplexData,
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings,
        }
    }

    fn rededicate_with_runtime_data(origin: DspVec<S, T, N, D>, _: bool, _: DataDomain) -> Self {
        Self::rededicate_from_force(origin)
    }
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for ComplexFreqVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        ComplexFreqVec {
            data: origin.data,
            delta: origin.delta,
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings,
        }
    }

    fn rededicate_with_runtime_data(origin: DspVec<S, T, N, D>, _: bool, _: DataDomain) -> Self {
        Self::rededicate_from_force(origin)
    }
}

impl<S, T, N, D> RededicateForceOps<DspVec<S, T, N, D>> for GenDspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn rededicate_from_force(origin: DspVec<S, T, N, D>) -> Self {
        let domain = origin.domain();
        let is_complex = origin.is_complex();
        GenDspVec {
            data: origin.data,
            delta: origin.delta,
            domain: TimeOrFrequencyData { domain_current: domain },
            number_space: RealOrComplexData { is_complex_current: is_complex },
            valid_len: origin.valid_len,
            multicore_settings: origin.multicore_settings,
        }
    }

    fn rededicate_with_runtime_data(origin: DspVec<S, T, N, D>,
                                    is_complex: bool,
                                    domain: DataDomain)
                                    -> Self {
        let mut result = Self::rededicate_from_force(origin);
        result.number_space.is_complex_current = is_complex;
        result.domain.domain_current = domain;
        result
    }
}

impl<U, V> RededicateOps<U> for V
    where V: RededicateForceOps<U> + MetaData + ResizeOps,
          U: MetaData
{
    fn rededicate_from(origin: U) -> Self {
        let is_complex = origin.is_complex();
        let domain = origin.domain();
        let mut result = Self::rededicate_from_force(origin);
        if result.is_complex() != is_complex && result.domain() != domain {
            result.resize(0).expect("Setting size to 0 should always succeed");
        }
        result
    }
}

impl<T, U> RededicateToOps<U> for T
    where U: RededicateOps<T>
{
    fn rededicate(self) -> U {
        U::rededicate_from(self)
    }
}
