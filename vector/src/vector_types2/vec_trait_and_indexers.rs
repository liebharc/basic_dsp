//! This module defines the basic vector trait and indexers.
use RealNumber;
use vector_types::{
    array_to_complex,
    array_to_complex_mut,
    ComplexIndex,
    ComplexIndexMut,
    DataVecDomain};
use super::{
    GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec,
    ToSlice, ToSliceMut, Resize};
use std::ops::*;
use num::complex::Complex;

/// Marker trait for types containing real data.
pub trait RealData { }

/// Marker trait for types containing complex data.
pub trait ComplexData { }

/// Marker trait for types containing time domain data.
pub trait TimeData { }

/// Marker trait for types containing frequency domain data.
pub trait FrequencyData { }

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

            impl<S, T> Index<usize> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = T;

                fn index(&self, index: usize) -> &T {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = &slice[0..len];
                    &slice[index]
                }
            }

            impl<S, T> IndexMut<usize> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn index_mut(&mut self, index: usize) -> &mut T {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = &mut slice[0..len];
                    &mut slice[index]
                }
            }

            impl<S, T> Index<RangeFull> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = [T];

                fn index(&self, _index: RangeFull) -> &[T] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = &slice[0..len];
                    slice
                }
            }

            impl<S, T> IndexMut<RangeFull> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = &mut slice[0..len];
                    slice
                }
            }

            impl<S, T> Index<RangeFrom<usize>> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = [T];

                fn index(&self, index: RangeFrom<usize>) -> &[T] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = &slice[0..len];
                    &slice[index]
                }
            }

            impl<S, T> IndexMut<RangeFrom<usize>> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = &mut slice[0..len];
                    &mut slice[index]
                }
            }

            impl<S, T> Index<RangeTo<usize>> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = [T];

                fn index(&self, index: RangeTo<usize>) -> &[T] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = &slice[0..len];
                    &slice[index]
                }
            }

            impl<S, T> IndexMut<RangeTo<usize>> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = &mut slice[0..len];
                    &mut slice[index]
                }
            }

            impl<S, T> Index<Range<usize>> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = [T];

                fn index(&self, index: Range<usize>) -> &[T] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = &slice[0..len];
                    &slice[index]
                }
            }

            impl<S, T> IndexMut<Range<usize>> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = &mut slice[0..len];
                    &mut slice[index]
                }
            }
        )*
    }
}

macro_rules! define_complex_vector_conversions {
    ($($name:ident),*) => {
        $(
            impl<S, T> ComplexIndex<usize> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = Complex<T>;

                fn complex(&self, index: usize) -> &Complex<T> {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = array_to_complex(&slice[0..len]);
                    &slice[index]
                }
            }

            impl<S, T> ComplexIndexMut<usize> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn complex_mut(&mut self, index: usize) -> &mut Complex<T> {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = array_to_complex_mut(&mut slice[0..len]);
                    &mut slice[index]
                }
            }

            impl<S, T> ComplexIndex<RangeFull> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = [Complex<T>];

                fn complex(&self, _index: RangeFull) -> &[Complex<T>] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = array_to_complex(&slice[0..len]);
                    slice
                }
            }

            impl<S, T> ComplexIndexMut<RangeFull> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn complex_mut(&mut self, _index: RangeFull) -> &mut [Complex<T>] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = array_to_complex_mut(&mut slice[0..len]);
                    slice
                }
            }

            impl<S, T> ComplexIndex<RangeFrom<usize>> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = [Complex<T>];

                fn complex(&self, index: RangeFrom<usize>) -> &[Complex<T>] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = array_to_complex(&slice[0..len]);
                    &slice[index]
                }
            }

            impl<S, T> ComplexIndexMut<RangeFrom<usize>> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn complex_mut(&mut self, index: RangeFrom<usize>) -> &mut [Complex<T>] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = array_to_complex_mut(&mut slice[0..len]);
                    &mut slice[index]
                }
            }

            impl<S, T> ComplexIndex<RangeTo<usize>> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = [Complex<T>];

                fn complex(&self, index: RangeTo<usize>) -> &[Complex<T>] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = array_to_complex(&slice[0..len]);
                    &slice[index]
                }
            }

            impl<S, T> ComplexIndexMut<RangeTo<usize>> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn complex_mut(&mut self, index: RangeTo<usize>) -> &mut [Complex<T>] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = array_to_complex_mut(&mut slice[0..len]);
                    &mut slice[index]
                }
            }

            impl<S, T> ComplexIndex<Range<usize>> for $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                type Output = [Complex<T>];

                fn complex(&self, index: Range<usize>) -> &[Complex<T>] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice();
                    let slice = array_to_complex(&slice[0..len]);
                    &slice[index]
                }
            }

            impl<S, T> ComplexIndexMut<Range<usize>> for $name<S, T>
                where S: ToSliceMut<T>,
                      T: RealNumber {
                fn complex_mut(&mut self, index: Range<usize>) -> &mut [Complex<T>] {
                    let len = self.valid_len;
                    let slice = self.data.to_slice_mut();
                    let slice = array_to_complex_mut(&mut slice[0..len]);
                    &mut slice[index]
                }
            }
        )*
    }
}

define_vector_conversions!(
    GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec);

define_complex_vector_conversions!(
    GenDspVec,
    ComplexTimeVec, ComplexFreqVec);

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn len_of_vec() {
        let vec: Vec<f32> = vec!(1.0, 2.0, 3.0);
        let dsp = vec.to_real_time();
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    fn len_of_slice() {
        let slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_real_freq();
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    #[allow(unused_mut)]
    fn len_of_slice_mut() {
        let mut slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_real_freq();
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    #[allow(unused_mut)]
    fn len_of_invalid_storage() {
        let mut slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_complex_freq();
        assert_eq!(dsp.len(), 0);
    }

    #[test]
    fn index_of_vec() {
        let vec = vec!(1.0, 2.0, 3.0);
        let dsp = vec.to_real_time();
        assert_eq!(dsp[..], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn index_of_slice() {
        let slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_real_time();
        assert_eq!(dsp[..], [1.0, 5.0, 4.0]);
    }
}
