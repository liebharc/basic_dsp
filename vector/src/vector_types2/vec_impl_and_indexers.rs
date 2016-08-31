//! This module defines the basic vector trait and indexers.
use RealNumber;
use vector_types::{
    array_to_complex,
    array_to_complex_mut,
    ComplexIndex,
    ComplexIndexMut};
use super::{
    DspVec,
    DataDomain,
    ToSlice,
    ToSliceMut,
    Resize,
    ComplexNumberSpace};
use multicore_support::MultiCoreSettings;
use std::ops::*;
use num::complex::Complex;

impl<S, T, N, D> DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber  {
    /// The x-axis delta. If `domain` is time domain then `delta` is in `[s]`, in frequency domain `delta` is in `[Hz]`.
    fn delta(&self) -> T {
        self.delta
    }

    /// Sets the x-axis delta. If `domain` is time domain then `delta` is in `[s]`, in frequency domain `delta` is in `[Hz]`.
    fn set_delta(&mut self, delta: T) {
        self.delta = delta;
    }

    /// The domain in which the data vector resides. Basically specifies the x-axis and the type of operations which
    /// are valid on this vector.
    ///
    /// The domain can be changed using the `RededicateOps` trait.
    fn domain(&self) -> DataDomain {
        self.domain
    }

    /// Indicates whether the vector contains complex data. This also specifies the type of operations which are valid
    /// on this vector.
    ///
    /// The number space can be changed using the `RededicateOps` trait.
    fn is_complex(&self) -> bool {
        self.is_complex
    }

    /// The number of valid elements in the vector. This can be changed
    /// with the `Resize` trait.
    fn len(&self) -> usize {
        self.valid_len
    }

    /// The number of valid points. If the vector is complex then every valid point consists of two floating point numbers,
    /// while for real vectors every point only consists of one floating point number.
    fn points(&self) -> usize {
        self.valid_len / if self.is_complex { 2 } else { 1 }
    }

    /// Gets the multi core settings which determine how the
    /// work is split between several cores if the amount of data
    /// gets larger.
    fn get_multicore_settings(&self) -> &MultiCoreSettings {
        &self.multicore_settings
    }

    /// Sets the multi core settings which determine how the
    /// work is split between several cores if the amount of data
    /// gets larger.
    fn set_multicore_settings(&mut self, settings: MultiCoreSettings) {
        self.multicore_settings = settings;
    }
}

impl<S, T, N, D> Resize for DspVec<S, T, N, D>
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

impl<S, T, N, D> Index<usize> for DspVec<S, T, N, D>
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

impl<S, T, N, D> IndexMut<usize> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        &mut slice[index]
    }
}

impl<S, T, N, D> Index<RangeFull> for DspVec<S, T, N, D>
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

impl<S, T, N, D> IndexMut<RangeFull> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber {
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        slice
    }
}

impl<S, T, N, D> Index<RangeFrom<usize>> for DspVec<S, T, N, D>
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

impl<S, T, N, D> IndexMut<RangeFrom<usize>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        &mut slice[index]
    }
}

impl<S, T, N, D> Index<RangeTo<usize>> for DspVec<S, T, N, D>
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

impl<S, T, N, D> IndexMut<RangeTo<usize>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber {
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        &mut slice[index]
    }
}

impl<S, T, N, D> Index<Range<usize>> for DspVec<S, T, N, D>
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

impl<S, T, N, D> IndexMut<Range<usize>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber {
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        &mut slice[index]
    }
}

impl<S, T, N, D> ComplexIndex<usize> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    type Output = Complex<T>;

    fn complex(&self, index: usize) -> &Complex<T> {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        &slice[index]
    }
}

impl<S, T, N, D> ComplexIndexMut<usize> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    fn complex_mut(&mut self, index: usize) -> &mut Complex<T> {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        &mut slice[index]
    }
}

impl<S, T, N, D> ComplexIndex<RangeFull> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    type Output = [Complex<T>];

    fn complex(&self, _index: RangeFull) -> &[Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        slice
    }
}

impl<S, T, N, D> ComplexIndexMut<RangeFull> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    fn complex_mut(&mut self, _index: RangeFull) -> &mut [Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        slice
    }
}

impl<S, T, N, D> ComplexIndex<RangeFrom<usize>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    type Output = [Complex<T>];

    fn complex(&self, index: RangeFrom<usize>) -> &[Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        &slice[index]
    }
}

impl<S, T, N, D> ComplexIndexMut<RangeFrom<usize>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    fn complex_mut(&mut self, index: RangeFrom<usize>) -> &mut [Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        &mut slice[index]
    }
}

impl<S, T, N, D> ComplexIndex<RangeTo<usize>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    type Output = [Complex<T>];

    fn complex(&self, index: RangeTo<usize>) -> &[Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        &slice[index]
    }
}

impl<S, T, N, D> ComplexIndexMut<RangeTo<usize>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    fn complex_mut(&mut self, index: RangeTo<usize>) -> &mut [Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        &mut slice[index]
    }
}

impl<S, T, N, D> ComplexIndex<Range<usize>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    type Output = [Complex<T>];

    fn complex(&self, index: Range<usize>) -> &[Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        &slice[index]
    }
}

impl<S, T, N, D> ComplexIndexMut<Range<usize>> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace {
    fn complex_mut(&mut self, index: Range<usize>) -> &mut [Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        &mut slice[index]
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn len_of_vec() {
        let vec: Vec<f32> = vec!(1.0, 2.0, 3.0);
        let dsp = vec.to_real_time_vec();
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    fn len_of_slice() {
        let slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_real_freq_vec();
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    #[allow(unused_mut)]
    fn len_of_slice_mut() {
        let mut slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_real_freq_vec();
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    #[allow(unused_mut)]
    fn len_of_invalid_storage() {
        let mut slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_complex_freq_vec();
        assert_eq!(dsp.len(), 0);
    }

    #[test]
    fn index_of_vec() {
        let vec = vec!(1.0, 2.0, 3.0);
        let dsp = vec.to_real_time_vec();
        assert_eq!(dsp[..], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn index_of_slice() {
        let slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_real_time_vec();
        assert_eq!(dsp[..], [1.0, 5.0, 4.0]);
    }
}
