use RealNumber;
use num::Complex;
use std::sync::Arc;
use multicore_support::*;
use super::super::{array_to_complex, array_to_complex_mut, ScalarResult, ErrorReason, DspVec,
                   ToSlice, ToSliceMut, MetaData, Domain, RealNumberSpace, ComplexNumberSpace};

/// Operations which allow to iterate over the vector and to derive results
/// or to change the vector.
pub trait MapInplaceOps<T>: Sized
    where T: Sized
{
    /// Transforms all vector elements using the function `map`.
    fn map_inplace<'a, A, F>(&mut self, argument: A, map: F)
        where A: Sync + Copy + Send,
              F: Fn(T, usize, A) -> T + 'a + Sync;
}

/// Operations which allow to iterate over the vector and to derive results
/// or to change the vector.
pub trait MapInplaceNoArgsOps<T>: Sized
    where T: Sized
{
    /// Transforms all vector elements using the function `map`.
    fn map_inplace<F>(&mut self, map: F) where F: Fn(T, usize) -> T + 'static + Sync + Send;
}

pub trait MapAggregateOps<T, R>: Sized
    where T: Sized,
          R: Send
{
    type Output;
    /// Transforms all vector elements using the function `map` and then aggregates
    /// all the results with `aggregate`. `aggregate` must be a commutativity and associativity;
    /// that's because there is no guarantee that the numbers will
    /// be aggregated in any deterministic order.
    fn map_aggregate<'a, A, FMap, FAggr>(&self,
                                         argument: A,
                                         map: FMap,
                                         aggregate: FAggr)
                                         -> Self::Output
        where A: Sync + Copy + Send,
              FMap: Fn(T, usize, A) -> R + 'a + Sync,
              FAggr: Fn(R, R) -> R + 'a + Sync + Send;
}

impl<S, T, N, D> MapInplaceOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    fn map_inplace<'a, A, F>(&mut self, argument: A, map: F)
        where A: Sync + Copy + Send,
              F: Fn(T, usize, A) -> T + 'a + Sync
    {
        if self.is_complex() {
            self.valid_len = 0;
            return;
        }

        let mut array = self.data.to_slice_mut();
        let length = array.len();
        Chunk::execute_with_range(Complexity::Small,
                                  &self.multicore_settings,
                                  &mut array[0..length],
                                  1,
                                  argument,
                                  move |array, range, argument| {
            let mut i = range.start;
            for num in array {
                *num = map(*num, i, argument);
                i += 1;
            }
        });
    }
}

impl<S, T, N, D, R> MapAggregateOps<T, R> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain,
          R: Send
{
    type Output = ScalarResult<R>;
    fn map_aggregate<'a, A, FMap, FAggr>(&self,
                                         argument: A,
                                         map: FMap,
                                         aggregate: FAggr)
                                         -> ScalarResult<R>
        where A: Sync + Copy + Send,
              FMap: Fn(T, usize, A) -> R + 'a + Sync,
              FAggr: Fn(R, R) -> R + 'a + Sync + Send
    {
        let aggregate = Arc::new(aggregate);
        let mut result = {
            if self.is_complex() {
                return Err(ErrorReason::InputMustBeReal);
            }

            let array = self.data.to_slice();
            let length = array.len();
            if length == 0 {
                return Err(ErrorReason::InputMustNotBeEmpty);
            }
            let aggregate = aggregate.clone();
            Chunk::map_on_array_chunks(Complexity::Small,
                                       &self.multicore_settings,
                                       &array[0..length],
                                       1,
                                       argument,
                                       move |array, range, argument| {
                let aggregate = aggregate.clone();
                let mut i = range.start;
                let mut sum: Option<R> = None;
                for num in array {
                    let res = map(*num, i, argument);
                    sum = match sum {
                        None => Some(res),
                        Some(s) => Some(aggregate(s, res)),
                    };
                    i += 1;
                }
                sum
            })
        };
        let aggregate = aggregate.clone();
        // Would be nicer if we could use iter().fold(..) but we need
        // the value of R and not just a reference so we can't user an iter
        let mut only_valid_options = Vec::with_capacity(result.len());
        for _ in 0..result.len() {
            let elem = result.pop().unwrap();
            match elem {
                None => (),
                Some(e) => only_valid_options.push(e),
            };
        }

        if only_valid_options.len() == 0 {
            return Err(ErrorReason::InputMustNotBeEmpty);
        }
        let mut aggregated = only_valid_options.pop().unwrap();
        for _ in 0..only_valid_options.len() {
            aggregated = aggregate(aggregated, only_valid_options.pop().unwrap());
        }
        Ok(aggregated)
    }
}

impl<S, T, N, D> MapInplaceOps<Complex<T>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    fn map_inplace<'a, A, F>(&mut self, argument: A, map: F)
        where A: Sync + Copy + Send,
              F: Fn(Complex<T>, usize, A) -> Complex<T> + 'a + Sync
    {
        if !self.is_complex() {
            self.valid_len = 0;
            return;
        }

        let mut array = self.data.to_slice_mut();
        let length = array.len();
        Chunk::execute_with_range(Complexity::Small,
                                  &self.multicore_settings,
                                  &mut array[0..length],
                                  2,
                                  argument,
                                  move |array, range, argument| {
            let mut i = range.start / 2;
            let array = array_to_complex_mut(array);
            for num in array {
                *num = map(*num, i, argument);
                i += 1;
            }
        });
    }
}

impl<S, T, N, D, R> MapAggregateOps<Complex<T>, R> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain,
          R: Send
{
    type Output = ScalarResult<R>;

    fn map_aggregate<'a, A, FMap, FAggr>(&self,
                                         argument: A,
                                         map: FMap,
                                         aggregate: FAggr)
                                         -> ScalarResult<R>
        where A: Sync + Copy + Send,
              FMap: Fn(Complex<T>, usize, A) -> R + 'a + Sync,
              FAggr: Fn(R, R) -> R + 'a + Sync + Send
    {
        let aggregate = Arc::new(aggregate);
        let mut result = {
            if !self.is_complex() {
                return Err(ErrorReason::InputMustBeComplex);
            }

            let array = self.data.to_slice();
            let length = array.len();
            if length == 0 {
                return Err(ErrorReason::InputMustNotBeEmpty);
            }
            let aggregate = aggregate.clone();
            Chunk::map_on_array_chunks(Complexity::Small,
                                       &self.multicore_settings,
                                       &array[0..length],
                                       2,
                                       argument,
                                       move |array, range, argument| {
                let aggregate = aggregate.clone();
                let array = array_to_complex(array);
                let mut i = range.start / 2;
                let mut sum: Option<R> = None;
                for num in array {
                    let res = map(*num, i, argument);
                    sum = match sum {
                        None => Some(res),
                        Some(s) => Some(aggregate(s, res)),
                    };
                    i += 1;
                }
                sum
            })
        };
        let aggregate = aggregate.clone();
        // Would be nicer if we could use iter().fold(..) but we need
        // the value of R and not just a reference so we can't user an iter
        let mut only_valid_options = Vec::with_capacity(result.len());
        for _ in 0..result.len() {
            let elem = result.pop().unwrap();
            match elem {
                None => (),
                Some(e) => only_valid_options.push(e),
            };
        }

        if only_valid_options.len() == 0 {
            return Err(ErrorReason::InputMustNotBeEmpty);
        }
        let mut aggregated = only_valid_options.pop().unwrap();
        for _ in 0..only_valid_options.len() {
            aggregated = aggregate(aggregated, only_valid_options.pop().unwrap());
        }
        Ok(aggregated)
    }
}
