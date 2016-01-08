use super::definitions::{
    DataVector,
    VecResult};
use RealNumber;
use super::{
    GenericDataVector,
    RealTimeVector,
    RealFreqVector,
    ComplexTimeVector,
    ComplexFreqVector};

/// Cross-correlation of data vectors. See also https://en.wikipedia.org/wiki/Cross-correlation
pub trait CrossCorrelation<T> : DataVector<T> 
    where T : RealNumber {
    /// Calculates the correlation between `self` and `other`. Correlation is internally always
    /// done in frequency domain, however the result is stored in the same domain as `self` was
    /// before the operation was performed.
    fn correlate(self, other: &Self) -> VecResult<Self>;
}

macro_rules! define_correlation_impl {
    ($($data_type:ident);*) => {
        $( 
            impl CrossCorrelation<$data_type> for GenericDataVector<$data_type> {
                fn correlate(self, _other: &Self) -> VecResult<Self> {
                    panic!("Panic")
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
    RealTimeVector, f32; RealTimeVector, f64;
    ComplexTimeVector, f32; ComplexTimeVector, f64;
    RealFreqVector, f32; RealFreqVector, f64;
    ComplexFreqVector, f32; ComplexFreqVector, f64
);