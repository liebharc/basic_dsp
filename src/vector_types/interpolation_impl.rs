use super::definitions::{
    DataVector,
    VecResult};
use RealNumber;
use conv_types::RealTimeConvFunction;

/// Provides a interpolation operation for data vectors.
pub trait Interpolation<T> : DataVector<T> 
    where T : RealNumber {
    /// Interpolates `self` with the convolution function `function`.
    fn interpolate(self, function: RealTimeConvFunction<T>, interpolation_factor: T, delay: T) -> VecResult<Self>;
}