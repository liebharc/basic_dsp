use array_to_complex_mut;
use numbers::*;
use super::super::{Owner, ToFreqResult, TimeDomain, MetaData, DspVec,
                   ToSliceMut, NumberSpace, RededicateForceOps};
use window_functions::*;

/// Defines all operations which are valid on `DataVecs` containing time domain data.
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the vector isn't in time domain.
pub trait TimeDomainOperations<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Applies a window to the data vector.
    fn apply_window(&mut self, window: &WindowFunction<T>);

    /// Removes a window from the data vector.
    fn unapply_window(&mut self, window: &WindowFunction<T>);
}

impl<S, T, N, D> TimeDomainOperations<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToFreqResult,
          <DspVec<S, T, N, D> as ToFreqResult>::FreqResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: TimeDomain
{
    fn apply_window(&mut self, window: &WindowFunction<T>) {
        if self.is_complex() {
            self.multiply_window_priv(window.is_symmetric(),
                                      |array| array_to_complex_mut(array),
                                      window,
                                      |f, i, p| Complex::<T>::new(f.window(i, p), T::zero()));
        } else {
            self.multiply_window_priv(window.is_symmetric(),
                                      |array| array,
                                      window,
                                      |f, i, p| f.window(i, p));
        }
    }

    fn unapply_window(&mut self, window: &WindowFunction<T>) {
        if self.is_complex() {
            self.multiply_window_priv(window.is_symmetric(),
                                      |array| array_to_complex_mut(array),
                                      window,
                                      |f, i, p| {
                                          Complex::<T>::new(T::one() / f.window(i, p), T::zero())
                                      });
        } else {
            self.multiply_window_priv(window.is_symmetric(),
                                      |array| array,
                                      window,
                                      |f, i, p| T::one() / f.window(i, p));
        }
    }
}
