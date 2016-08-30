//! This module defines the basic vector trait and indexers.
use RealNumber;
use vector_types::{
    DataVecDomain};
use super::{
    GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec,
    ToSlice, Resize};

/// `DspVec` gives access to the basic properties of all data vectors.
///
/// A `DspVec` allocates memory if necessary. It will however never shrink/free memory unless it's
/// deleted and dropped.
pub trait DspVec<T>
    where T: RealNumber {
    /// The x-axis delta. If `domain` is time domain then `delta` is in `[s]`, in frequency domain `delta` is in `[Hz]`.
    fn delta(&self) -> T;

    /// Sets the x-axis delta. If `domain` is time domain then `delta` is in `[s]`, in frequency domain `delta` is in `[Hz]`.
    fn set_delta(&mut self, delta: T);

    /// The domain in which the data vector resides. Basically specifies the x-axis and the type of operations which
    /// are valid on this vector.
    ///
    /// The domain can be changed using the `RededicateOps` trait.
    fn domain(&self) -> DataVecDomain;

    /// Indicates whether the vector contains complex data. This also specifies the type of operations which are valid
    /// on this vector.
    ///
    /// The number space can be changed using the `RededicateOps` trait.
    fn is_complex(&self) -> bool;

    /// The number of valid elements in the vector. This can be changed
    /// with the `Resize` trait.
    fn len(&self) -> usize;

    /// The number of valid points. If the vector is complex then every valid point consists of two floating point numbers,
    /// while for real vectors every point only consists of one floating point number.
    fn points(&self) -> usize;
}

macro_rules! define_vector_conversions {
    ($($name:ident),*) => {
        $(
            impl<S, T> DspVec<T> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
              fn delta(&self) -> T {
                  self.delta
              }

              fn set_delta(&mut self, delta: T) {
                  self.delta = delta;
              }

              fn domain(&self) -> DataVecDomain {
                  self.domain
              }

              fn is_complex(&self) -> bool {
                  self.is_complex
              }

              fn len(&self) -> usize {
                  self.valid_len
              }

              fn points(&self) -> usize {
                  self.valid_len / if self.is_complex { 2 } else { 1 }
              }
            }

            impl<S, T> Resize for $name<S, T>
                where S: ToSlice<T> + Resize,
                      T: RealNumber {
                  fn resize(&mut self, len: usize) {
                      if len > self.alloc_len() {
                          self.data.resize(len);
                      }
                  }

                  fn alloc_len(&self) -> usize {
                      self.data.alloc_len()
                  }
            }
        )*
    }
}
define_vector_conversions!(
    GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec);
