use super::{
    ComplexData, ComplexFreqVec, ComplexTimeVec, DataDomain, Domain, DspVec, FrequencyData,
    GenDspVec, NumberSpace, RealData, RealFreqVec, RealOrComplexData, RealTimeVec, TimeData,
    TimeOrFrequencyData, ToSlice, TypeMetaData,
};
use super::{ToComplexVector, ToDspVector, ToRealVector};
use multicore_support::MultiCoreSettings;
/// ! Support for types in Rust std
use numbers::*;
use vector_types::vec_impl_and_indexers::Vector;

/// Conversion from a generic data type into a dsp vector which tracks
/// its meta information (domain and number space)
/// only at runtime. See `ToRealVector` and
/// `ToComplexVector` for alternatives which track most of the meta data
/// with the type system and therefore avoid runtime errors.
///
/// The resulting vector may use multi-threading for its processing.
pub trait ToDspVectorPar<T>: Sized + ToSlice<T>
where
    T: RealNumber,
{
    /// Create a new generic vector.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    ///
    /// The resulting vector may use multi-threading for its processing.
    fn to_gen_dsp_vec_par(self, is_complex: bool, domain: DataDomain) -> GenDspVec<Self, T>;

    /// Create a new vector from the given meta data. The meta data can be
    /// retrieved from an existing vector. If no existing vector is available
    /// then one of the other constructor methods should be used.
    ///
    /// The resulting vector may use multi-threading for its processing.
    fn to_dsp_vec_par<N, D>(self, meta_data: &TypeMetaData<T, N, D>) -> DspVec<Self, T, N, D>
    where
        N: NumberSpace,
        D: Domain;
}

/// Conversion from a generic data type into a dsp vector with real data.
///
/// The resulting vector may use multi-threading for its processing.
pub trait ToRealVectorPar<T>: Sized + ToSlice<T>
where
    T: RealNumber,
{
    /// Create a new vector in real number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// The resulting vector may use multi-threading for its processing.
    fn to_real_time_vec_par(self) -> RealTimeVec<Self, T>;

    /// Create a new vector in real number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// The resulting vector may use multi-threading for its processing.
    fn to_real_freq_vec_par(self) -> RealFreqVec<Self, T>;
}

/// Conversion from a generic data type into a dsp vector with complex data.
///
/// The resulting vector may use multi-threading for its processing.
pub trait ToComplexVectorPar<S, T>
where
    S: Sized + ToSlice<T>,
    T: RealNumber,
{
    /// Create a new vector in complex number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    fn to_complex_time_vec_par(self) -> ComplexTimeVec<S, T>;

    /// Create a new vector in complex number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex vectors with an odd length the resulting value will have a zero length.
    fn to_complex_freq_vec_par(self) -> ComplexFreqVec<S, T>;
}

impl<Type: ToDspVector<T>, T: RealNumber> ToDspVectorPar<T> for Type {
    fn to_gen_dsp_vec_par(
        self,
        is_complex: bool,
        domain: DataDomain,
    ) -> DspVec<Self, T, RealOrComplexData, TimeOrFrequencyData> {
        let mut vec = self.to_gen_dsp_vec(is_complex, domain);
        vec.set_multicore_settings(MultiCoreSettings::parallel());
        vec
    }

    fn to_dsp_vec_par<N, D>(self, meta_data: &TypeMetaData<T, N, D>) -> DspVec<Self, T, N, D>
    where
        N: NumberSpace,
        D: Domain,
    {
        let mut vec = self.to_dsp_vec(meta_data);
        vec.set_multicore_settings(MultiCoreSettings::parallel());
        vec
    }
}

impl<Type: ToRealVector<T>, T: RealNumber> ToRealVectorPar<T> for Type {
    fn to_real_time_vec_par(self) -> DspVec<Self, T, RealData, TimeData> {
        let mut vec = self.to_real_time_vec();
        vec.set_multicore_settings(MultiCoreSettings::parallel());
        vec
    }

    fn to_real_freq_vec_par(self) -> DspVec<Self, T, RealData, FrequencyData> {
        let mut vec = self.to_real_freq_vec();
        vec.set_multicore_settings(MultiCoreSettings::parallel());
        vec
    }
}

impl<Type: ToComplexVector<S, T> + Sized + ToSlice<T>, S: Sized + ToSlice<T>, T: RealNumber>
    ToComplexVectorPar<S, T> for Type
{
    fn to_complex_time_vec_par(self) -> DspVec<S, T, ComplexData, TimeData> {
        let mut vec = self.to_complex_time_vec();
        vec.set_multicore_settings(MultiCoreSettings::parallel());
        vec
    }

    fn to_complex_freq_vec_par(self) -> DspVec<S, T, ComplexData, FrequencyData> {
        let mut vec = self.to_complex_freq_vec();
        vec.set_multicore_settings(MultiCoreSettings::parallel());
        vec
    }
}

#[cfg(test)]
mod tests {
    use num_cpus;
    use vector_types::*;

    #[test]
    fn single_threaded_vector() {
        let data = vec![0.0; 6];
        let vector = data.to_complex_time_vec();
        assert_eq!(vector.get_multicore_settings().core_limit, 1);
    }

    #[test]
    fn parallel_vector() {
        let data = vec![0.0; 6];
        let vector = data.to_complex_time_vec_par();
        assert!(vector.get_multicore_settings().core_limit > 1 || num_cpus::get() == 1);
    }
}
