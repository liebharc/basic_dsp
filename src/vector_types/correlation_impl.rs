use super::definitions::DataVector;
use RealNumber;

/// Cross-correlation of data vectors. See also https://en.wikipedia.org/wiki/Cross-correlation
pub trait CrossCorrelation<T> : DataVector<T> 
    where T : RealNumber {
    /// Calculates the correlation between `self` and `other`. Correlation is internally always
    /// done in frequency domain, however the result is stored in the same domain as `self` was
    /// before the operation was performed.
    fn correlate(self, other: &Self) -> Self
}