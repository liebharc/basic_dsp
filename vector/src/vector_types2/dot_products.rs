/// An operation which multiplies each vector element with a constant
pub trait DotProductOps<T> : Sized
    where T: Sized {
    type SumResult;
    /// Calculates the dot product of self and factor. Self and factor remain unchanged.
    fn dot_product(&self, factor: &Self) -> Self::SumResult;
}
