/// Operations which allow to iterate over the vector and to derive results
/// or to change the vector.
pub trait VectorIter<T> : Sized
    where T: Sized {
    /// Transforms all vector elements using the function `map`.
    fn map_inplace<A, F>(self, argument: A, map: F) -> TransRes<Self>
        where A: Sync + Copy + Send,
              F: Fn(T, usize, A) -> T + 'static + Sync;

    /// Transforms all vector elements using the function `map` and then aggregates
    /// all the results with `aggregate`. `aggregate` must be a commutativity and associativity;
    /// that's because there is no guarantee that the numbers will be aggregated in any deterministic order.
    fn map_aggregate<A, FMap, FAggr, R>(
            &self,
            argument: A,
            map: FMap,
            aggregate: FAggr) -> ScalarResult<R>
        where A: Sync + Copy + Send,
              FMap: Fn(T, usize, A) -> R + 'static + Sync,
              FAggr: Fn(R, R) -> R + 'static + Sync + Send,
              R: Send;
}
