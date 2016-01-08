use super::definitions::{
    DataVectorDomain,
    ErrorReason,
    DataVector,
    VecResult};
use RealNumber;
use super::{
    GenericDataVector,
    GenericVectorOperations,
    ComplexVectorOperations,
    ComplexTimeVector,
    ComplexFreqVector};
use super::convolution_impl::Convolution;

/// Cross-correlation of data vectors. See also https://en.wikipedia.org/wiki/Cross-correlation
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions are consistent.
/// However the actual implementation is lacking tests.
/// # Failures
/// VecResult may report the following `ErrorReason` members:
/// 
/// 1. `VectorMustBeComplex`: if `self` is in real number space.
/// 3. `VectorMetaDataMustAgree`: in case `self` and `function` are not in the same number space and same domain.
pub trait CrossCorrelation<T> : DataVector<T> 
    where T : RealNumber {
    /// Calculates the correlation between `self` and `other`.
    fn correlate(self, other: &Self) -> VecResult<Self>;
}

macro_rules! define_correlation_impl {
    ($($data_type:ident);*) => {
        $( 
            impl CrossCorrelation<$data_type> for GenericDataVector<$data_type> {
                fn correlate(self, other: &Self) -> VecResult<Self> {
                    assert_complex!(self);
                    assert_meta_data!(self, other);
                    if self.domain == DataVectorDomain::Time {
                        let points = self.points();
                        self.complex_conj()
                        .and_then(|v|v.convolve(other, 1.0, points))
                    } else {
                        self.complex_conj()
                        .and_then(|v|v.multiply_vector(&other))
                    }
                }
            }
        )*
    }
}
define_correlation_impl!(f32; f64);

macro_rules! define_correlation_forward {
    ($($name:ident, $data_type:ident);*) => {
        $( 
            impl CrossCorrelation<$data_type> for $name<$data_type> {
                fn correlate(self, other: &Self) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().correlate(other.to_gen_borrow()))
                }
            }
        )*
    }
}

define_correlation_forward!(
    ComplexTimeVector, f32; ComplexTimeVector, f64;
    ComplexFreqVector, f32; ComplexFreqVector, f64
);