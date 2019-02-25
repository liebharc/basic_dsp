//! This module defines the basic vector trait and indexers.
use super::{
    Buffer, BufferBorrow, ComplexNumberSpace, DataDomain, Domain, DspVec, ErrorReason, NumberSpace,
    ToSlice, ToSliceMut, TypeMetaData, VoidResult
};
use crate::multicore_support::MultiCoreSettings;
use crate::numbers::*;
use std::ops::*;
use crate::{array_to_complex, array_to_complex_mut};
use super::meta::Real as RealSpace;
use super::meta::Complex as ComplexSpace;

/// Like [`std::ops::Index`](https://doc.rust-lang.org/std/ops/trait.Index.html)
/// but with a different method name so that it can be used to implement an additional range
/// accessor for float data.
pub trait FloatIndex<Idx>
where
    Idx: Sized,
{
    type Output: ?Sized;
    /// The method for float indexing. If the vector is real valued then this function just returns the values
    // in the vector. If the vector is compled valued then will returned real and imaginary values interleaved (all
    // values at even indexes are the real parts, while all values at odd indexes are imag parts).
    fn data(&self, index: Idx) -> &Self::Output;
}

/// Like [`std::ops::IndexMut`](https://doc.rust-lang.org/std/ops/trait.IndexMut.html)
/// but with a different method name so that it can be used to implement a additional range
/// accessor for float data.
pub trait FloatIndexMut<Idx>: FloatIndex<Idx>
where
    Idx: Sized,
{
    /// The method for real indexing
    fn data_mut(&mut self, index: Idx) -> &mut Self::Output;
}

/// Like [`std::ops::Index`](https://doc.rust-lang.org/std/ops/trait.Index.html)
/// but with a different method name so that it can be used to implement an additional range
/// accessor for complex data.
///
/// Note if indexers will return an empty array in case the vector isn't complex.
pub trait ComplexIndex<Idx>
where
    Idx: Sized,
{
    type Output: ?Sized;
    /// The method for complex indexing
    fn datac(&self, index: Idx) -> &Self::Output;
}

/// Like [`std::ops::IndexMut`](https://doc.rust-lang.org/std/ops/trait.IndexMut.html)
/// but with a different method name so that it can be used to implement a additional range
/// accessor for complex data.
///
/// Note if indexers will return an empty array in case the vector isn't complex.
pub trait ComplexIndexMut<Idx>: ComplexIndex<Idx>
where
    Idx: Sized,
{
    /// The method for complex indexing
    fn datac_mut(&mut self, index: Idx) -> &mut Self::Output;
}

/// A trait which provides information about number space and domain.
pub trait MetaData {
    /// The domain in which the data vector resides. Basically specifies the x-axis and the
    /// type of operations which are valid on this vector.
    ///
    /// The domain can be changed using the `RededicateOps` trait.
    fn domain(&self) -> DataDomain;

    /// Indicates whether the vector contains complex data. This also specifies the type
    /// of operations which are valid on this vector.
    ///
    /// The number space can be changed using the `RededicateOps` trait.
    fn is_complex(&self) -> bool;
}

/// Operations to resize a data type.
pub trait ResizeOps {
    /// Changes `self.len()`.
    /// If `self.is_complex()` is true then `len` must be an even number.
    /// `len > self.alloc_len()` is only possible if the underlying storage supports resizing.
    fn resize(&mut self, len: usize) -> VoidResult;
}

/// Operations to resize a data type.
pub trait ResizeBufferedOps<S: ToSliceMut<T>, T: RealNumber> {
    /// Changes `self.len()`.
    /// If `self.is_complex()` is true then `len` must be an even number.
    /// `len > self.alloc_len()` is only possible if the underlying storage or the buffer
    /// supports resizing.
    fn resize_b<B>(&mut self, buffer: &mut B, len: usize) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>;
}

/// A trait for vector types.
pub trait Vector<T>: MetaData + ResizeOps
    + FloatIndex<RangeFull, Output=[T]> + FloatIndex<Range<usize>>  + FloatIndex<RangeFrom<usize>, Output=[T]>
    + FloatIndex<RangeTo<usize>> + FloatIndex<usize>
where
    T: RealNumber,
{
    /// The x-axis delta. If `domain` is time domain then `delta` is in `[s]`,
    /// in frequency domain `delta` is in `[Hz]`.
    fn delta(&self) -> T;

    /// Sets the x-axis delta. If `domain` is time domain then `delta` is in `[s]`,
    /// in frequency domain `delta` is in `[Hz]`.
    fn set_delta(&mut self, delta: T);

    /// The number of valid elements in the vector. This can be changed
    /// with the `Resize` trait.
    fn len(&self) -> usize;

    /// Indicates whether or not the vector is empty.
    fn is_empty(&self) -> bool;

    /// The number of valid points. If the vector is complex then every valid
    /// point consists of two floating point numbers,
    /// while for real vectors every point only consists of one floating point number.
    fn points(&self) -> usize;

    /// Gets the multi core settings which determine how the
    /// work is split between several cores if the amount of data
    /// gets larger.
    fn get_multicore_settings(&self) -> &MultiCoreSettings;

    /// Sets the multi core settings which determine how the
    /// work is split between several cores if the amount of data
    /// gets larger.
    fn set_multicore_settings(&mut self, settings: MultiCoreSettings);

    /// Gets the number of allocated elements in the underlying vector.
    /// The allocated length may be larger than the length of valid points.
    /// In most cases you likely want to have `len`or `points` instead.
    fn alloc_len(&self) -> usize;
}

/// Gets the meta data of a type. This can be used to create a new type with the same
/// meta data.
/// # Example
///
/// ```
/// use basic_dsp_vector::*;
/// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_real_time_vec();
/// let meta_data = vector.get_meta_data();
/// let slice = &vector[0..2];
/// let slice = slice.to_dsp_vec(&meta_data);
/// assert_eq!(false, slice.is_complex());
/// ```
pub trait GetMetaData<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    /// Gets a copy of the vector meta data. This can be used to create
    /// new types with the same meta data.
    fn get_meta_data(&self) -> TypeMetaData<T, N, D>;
}

impl<S, T, N, D> MetaData for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn domain(&self) -> DataDomain {
        self.domain.domain()
    }

    fn is_complex(&self) -> bool {
        self.number_space.is_complex()
    }
}

impl<T, N, D> MetaData for TypeMetaData<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn domain(&self) -> DataDomain {
        self.domain.domain()
    }

    fn is_complex(&self) -> bool {
        self.number_space.is_complex()
    }
}

impl<S, T, N, D> ResizeOps for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn resize(&mut self, len: usize) -> VoidResult {
        if self.is_complex() && len % 2 != 0 {
            return Err(ErrorReason::InputMustHaveAnEvenLength);
        }

        if len > self.alloc_len() {
            self.data.try_resize(len)?;
        }

        self.valid_len = len;

        Ok(())
    }
}

impl<S, T, N, D> ResizeBufferedOps<S, T> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn resize_b<B>(&mut self, buffer: &mut B, len: usize) -> VoidResult
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        if self.is_complex() && len % 2 != 0 {
            return Err(ErrorReason::InputMustHaveAnEvenLength);
        }
        let res = self.resize(len);
        match res {
            Ok(_) => Ok(()),
            Err(_) => {
                let orig_len = self.len();
                let mut temp = buffer.borrow(len);
                temp[0..orig_len].clone_from_slice(self.data(..));
                temp.trade(&mut self.data);
                self.valid_len = len;
                Ok(())
            }
        }
    }
}

impl<S, T, N, D> Vector<T> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn delta(&self) -> T {
        self.delta
    }

    fn set_delta(&mut self, delta: T) {
        self.delta = delta;
    }

    fn len(&self) -> usize {
        self.valid_len
    }

    fn is_empty(&self) -> bool {
        self.valid_len == 0
    }

    fn points(&self) -> usize {
        self.valid_len / if self.is_complex() { 2 } else { 1 }
    }

    fn get_multicore_settings(&self) -> &MultiCoreSettings {
        &self.multicore_settings
    }

    fn set_multicore_settings(&mut self, settings: MultiCoreSettings) {
        self.multicore_settings = settings;
    }

    fn alloc_len(&self) -> usize {
        self.data.len()
    }
}

impl<S, T, N, D> GetMetaData<T, N, D> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn get_meta_data(&self) -> TypeMetaData<T, N, D> {
        TypeMetaData {
            number_space: self.number_space.clone(),
            domain: self.domain.clone(),
            delta: self.delta,
            multicore_settings: self.multicore_settings,
        }
    }
}

impl<S, T, N, D> FloatIndex<usize> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = T;

    fn data(&self, index: usize) -> &T {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = &slice[0..len];
        &slice[index]
    }
}

impl<S, T, N, D> FloatIndexMut<usize> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn data_mut(&mut self, index: usize) -> &mut T {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        &mut slice[index]
    }
}

impl<S, T, N, D> FloatIndex<RangeFull> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = [T];

    fn data(&self, _index: RangeFull) -> &[T] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        &slice[0..len]
    }
}

impl<S, T, N, D> FloatIndexMut<RangeFull> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn data_mut(&mut self, _index: RangeFull) -> &mut [T] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        &mut slice[0..len]
    }
}

impl<S, T, N, D> FloatIndex<RangeFrom<usize>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = [T];

    fn data(&self, index: RangeFrom<usize>) -> &[T] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = &slice[0..len];
        &slice[index]
    }
}

impl<S, T, N, D> FloatIndexMut<RangeFrom<usize>> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn data_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        &mut slice[index]
    }
}

impl<S, T, N, D> FloatIndex<RangeTo<usize>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = [T];

    fn data(&self, index: RangeTo<usize>) -> &[T] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = &slice[0..len];
        &slice[index]
    }
}

impl<S, T, N, D> FloatIndexMut<RangeTo<usize>> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn data_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        &mut slice[index]
    }
}

impl<S, T, N, D> FloatIndex<Range<usize>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = [T];

    fn data(&self, index: Range<usize>) -> &[T] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = &slice[0..len];
        &slice[index]
    }
}

impl<S, T, N, D> FloatIndexMut<Range<usize>> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn data_mut(&mut self, index: Range<usize>) -> &mut [T] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = &mut slice[0..len];
        &mut slice[index]
    }
}

impl<S, T, N, D> ComplexIndex<usize> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    type Output = Complex<T>;

    fn datac(&self, index: usize) -> &Complex<T> {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        &slice[index]
    }
}

impl<S, T, N, D> ComplexIndexMut<usize> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn datac_mut(&mut self, index: usize) -> &mut Complex<T> {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        &mut slice[index]
    }
}

impl<S, T, N, D> ComplexIndex<RangeFull> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    type Output = [Complex<T>];

    fn datac(&self, _index: RangeFull) -> &[Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        array_to_complex(&slice[0..len])
    }
}

impl<S, T, N, D> ComplexIndexMut<RangeFull> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn datac_mut(&mut self, _index: RangeFull) -> &mut [Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        array_to_complex_mut(&mut slice[0..len])
    }
}

impl<S, T, N, D> ComplexIndex<RangeFrom<usize>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    type Output = [Complex<T>];

    fn datac(&self, index: RangeFrom<usize>) -> &[Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        &slice[index]
    }
}

impl<S, T, N, D> ComplexIndexMut<RangeFrom<usize>> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn datac_mut(&mut self, index: RangeFrom<usize>) -> &mut [Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        &mut slice[index]
    }
}

impl<S, T, N, D> ComplexIndex<RangeTo<usize>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    type Output = [Complex<T>];

    fn datac(&self, index: RangeTo<usize>) -> &[Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        &slice[index]
    }
}

impl<S, T, N, D> ComplexIndexMut<RangeTo<usize>> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn datac_mut(&mut self, index: RangeTo<usize>) -> &mut [Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        &mut slice[index]
    }
}

impl<S, T, N, D> ComplexIndex<Range<usize>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    type Output = [Complex<T>];

    fn datac(&self, index: Range<usize>) -> &[Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice();
        let slice = array_to_complex(&slice[0..len]);
        &slice[index]
    }
}

impl<S, T, N, D> ComplexIndexMut<Range<usize>> for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn datac_mut(&mut self, index: Range<usize>) -> &mut [Complex<T>] {
        let len = self.valid_len;
        let slice = self.data.to_slice_mut();
        let slice = array_to_complex_mut(&mut slice[0..len]);
        &mut slice[index]
    }
}

impl<S, T, D> Index<usize> for DspVec<S, T, RealSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.data(index)
    }
}

impl<S, T, D> IndexMut<usize> for DspVec<S, T, RealSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.data_mut(index)
    }
}

impl<S, T, D> Index<RangeFull> for DspVec<S, T, RealSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = [T];

    fn index(&self, index: RangeFull) -> &[T] {
        self.data(index)
    }
}

impl<S, T, D> IndexMut<RangeFull> for DspVec<S, T, RealSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: RangeFull) -> &mut [T] {
        self.data_mut(index)
    }
}

impl<S, T, D> Index<RangeFrom<usize>> for DspVec<S, T, RealSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = [T];

    fn index(&self, index: RangeFrom<usize>) -> &[T] {
        self.data(index)
    }
}

impl<S, T, D> IndexMut<RangeFrom<usize>> for DspVec<S, T, RealSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
        self.data_mut(index)
    }
}

impl<S, T, D> Index<RangeTo<usize>> for DspVec<S, T, RealSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = [T];

    fn index(&self, index: RangeTo<usize>) -> &[T] {
        self.data(index)
    }
}

impl<S, T, D> IndexMut<RangeTo<usize>> for DspVec<S, T, RealSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
        self.data_mut(index)
    }
}

impl<S, T, D> Index<Range<usize>> for DspVec<S, T, RealSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = [T];

    fn index(&self, index: Range<usize>) -> &[T] {
        self.data(index)
    }
}

impl<S, T, D> IndexMut<Range<usize>> for DspVec<S, T, RealSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        self.data_mut(index)
    }
}

impl<S, T, D> Index<usize> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = Complex<T>;

    fn index(&self, index: usize) -> &Complex<T> {
        self.datac(index)
    }
}

impl<S, T, D> IndexMut<usize> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: usize) -> &mut Complex<T> {
        self.datac_mut(index)
    }
}

impl<S, T, D> Index<RangeFull> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = [Complex<T>];

    fn index(&self, index: RangeFull) -> &[Complex<T>] {
        self.datac(index)
    }
}

impl<S, T, D> IndexMut<RangeFull> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: RangeFull) -> &mut [Complex<T>] {
        self.datac_mut(index)
    }
}

impl<S, T, D> Index<RangeFrom<usize>> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = [Complex<T>];

    fn index(&self, index: RangeFrom<usize>) -> &[Complex<T>] {
        self.datac(index)
    }
}

impl<S, T, D> IndexMut<RangeFrom<usize>> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [Complex<T>] {
        self.datac_mut(index)
    }
}

impl<S, T, D> Index<RangeTo<usize>> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = [Complex<T>];

    fn index(&self, index: RangeTo<usize>) -> &[Complex<T>] {
        self.datac(index)
    }
}

impl<S, T, D> IndexMut<RangeTo<usize>> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [Complex<T>] {
        self.datac_mut(index)
    }
}

impl<S, T, D> Index<Range<usize>> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    D: Domain,
{
    type Output = [Complex<T>];

    fn index(&self, index: Range<usize>) -> &[Complex<T>] {
        self.datac(index)
    }
}

impl<S, T, D> IndexMut<Range<usize>> for DspVec<S, T, ComplexSpace, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    D: Domain,
{
    fn index_mut(&mut self, index: Range<usize>) -> &mut [Complex<T>] {
        self.datac_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn len_of_vec() {
        let vec: Vec<f32> = vec![1.0, 2.0, 3.0];
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
        let vec = vec![1.0, 2.0, 3.0];
        let dsp = vec.to_real_time_vec();
        assert_eq!(dsp.data(..), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn index_of_slice() {
        let slice = [1.0, 5.0, 4.0];
        let dsp = slice.to_real_time_vec();
        assert_eq!(dsp.data(..), [1.0, 5.0, 4.0]);
    }
}
