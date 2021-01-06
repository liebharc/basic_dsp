use super::super::meta;
use super::{
    array_to_complex, array_to_complex_mut, complex_to_array, complex_to_array_mut, Buffer,
    BufferBorrow, DataDomain, Domain, DspVec, ErrorReason, FromVector, MetaData, NumberSpace,
    ToComplexVector, ToDspVector, ToRealVector, ToSlice, ToSliceMut, TypeMetaData, VoidResult,
};
use crate::inline_vector::InlineVector;
use crate::multicore_support::MultiCoreSettings;
/// ! Support for types in the Rust core
use crate::numbers::*;
use arrayvec;
use arrayvec::{Array, ArrayVec};
use std;

/// Buffer borrow type for `SingleBuffer`.
pub struct FixedLenBufferBurrow<'a, T: RealNumber + 'a> {
    data: &'a mut [T],
}

impl<'a, T: RealNumber + 'a> ToSlice<T> for FixedLenBufferBurrow<'a, T> {
    fn to_slice(&self) -> &[T] {
        self.data.to_slice()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn alloc_len(&self) -> usize {
        self.data.alloc_len()
    }

    fn try_resize(&mut self, len: usize) -> VoidResult {
        self.data.try_resize(len)
    }
}

impl<'a, T: RealNumber + 'a> ToSliceMut<T> for FixedLenBufferBurrow<'a, T>  {
    fn to_slice_mut(&mut self) -> &mut [T] {
        self.data.to_slice_mut()
    }
}

impl<'a, S: ToSliceMut<T>, T: RealNumber> BufferBorrow<S, T> for FixedLenBufferBurrow<'a, T> {
    fn trade(self, storage: &mut S) {
        let len = std::cmp::min(storage.len(), self.data.len());
        let storage = storage.to_slice_mut();
        storage[0..len]
            .to_slice_mut()
            .copy_from_slice(&self.data[0..len]);
    }
}

/// A buffer which gets initalized with a data storage type and then always keeps that.
pub struct FixedLenBuffer<S, T>
where
    S: ToSliceMut<T>,
    T: RealNumber,
{
    data: S,
    data_type: std::marker::PhantomData<T>,
}

impl<S, T> FixedLenBuffer<S, T>
where
    S: ToSliceMut<T>,
    T: RealNumber,
{
    /// Creates a new buffer from a storage type. The buffer will internally hold
    /// its storage for it's complete life time.
    pub fn new(storage: S) -> FixedLenBuffer<S, T> {
        FixedLenBuffer {
            data: storage,
            data_type: std::marker::PhantomData,
        }
    }
}

impl<'a, S, T> Buffer<'a, S, T> for FixedLenBuffer<S, T>
where
    S: ToSliceMut<T>,
    T: RealNumber + 'a,
{
    type Borrow = FixedLenBufferBurrow<'a, T>;

    fn borrow(&'a mut self, len: usize) -> Self::Borrow {
        if self.data.len() < len {
            panic!("FixedLenBuffer: Out of memory");
        }

        FixedLenBufferBurrow {
            data: &mut self.data.to_slice_mut()[0..len],
        }
    }

    fn alloc_len(&self) -> usize {
        self.data.len()
    }
}

/// A vector with real numbers in time domain.
pub type RealTimeVec<S, T> = DspVec<S, T, meta::Real, meta::Time>;
/// A vector with real numbers in frequency domain.
pub type RealFreqVec<S, T> = DspVec<S, T, meta::Real, meta::Freq>;
/// A vector with complex numbers in time domain.
pub type ComplexTimeVec<S, T> = DspVec<S, T, meta::Complex, meta::Time>;
/// A vector with complex numbers in frequency domain.
pub type ComplexFreqVec<S, T> = DspVec<S, T, meta::Complex, meta::Freq>;
/// A vector with no information about number space or domain at compile time.
pub type GenDspVec<S, T> = DspVec<S, T, meta::RealOrComplex, meta::TimeOrFreq>;

/// A vector with real numbers in time domain.
pub type RealTimeVecSlice32<'a> = DspVec<&'a [f32], f32, meta::Real, meta::Time>;
/// A vector with real numbers in frequency domain.
pub type RealFreqVecSlice32<'a> = DspVec<&'a [f32], f32, meta::Real, meta::Freq>;
/// A vector with complex numbers in time domain.
pub type ComplexTimeVecSlice32<'a> = DspVec<&'a [f32], f32, meta::Complex, meta::Time>;
/// A vector with complex numbers in frequency domain.
pub type ComplexFreqVecSlice32<'a> = DspVec<&'a [f32], f32, meta::Complex, meta::Freq>;
/// A vector with no information about number space or domain at compile time.
pub type GenDspVecSlice32<'a> = DspVec<&'a [f32], f32, meta::RealOrComplex, meta::TimeOrFreq>;

/// A vector with real numbers in time domain.
pub type RealTimeVecSlice64<'a> = DspVec<&'a [f64], f64, meta::Real, meta::Time>;
/// A vector with real numbers in frequency domain.
pub type RealFreqVecSlice64<'a> = DspVec<&'a [f64], f64, meta::Real, meta::Freq>;
/// A vector with complex numbers in time domain.
pub type ComplexTimeVecSlice64<'a> = DspVec<&'a [f64], f64, meta::Complex, meta::Time>;
/// A vector with complex numbers in frequency domain.
pub type ComplexFreqVecSlice64<'a> = DspVec<&'a [f64], f64, meta::Complex, meta::Freq>;
/// A vector with no information about number space or domain at compile time.
pub type GenDspVecSlice64<'a> = DspVec<&'a [f64], f64, meta::RealOrComplex, meta::TimeOrFreq>;

impl<'a, T> ToSlice<T> for &'a [T] {
    fn to_slice(&self) -> &[T] {
        self
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn is_empty(&self) -> bool {
        (*self).is_empty()
    }

    fn alloc_len(&self) -> usize {
        self.len()
    }

    fn try_resize(&mut self, len: usize) -> VoidResult {
        if len > self.len() {
            Err(ErrorReason::TypeCanNotResize)
        } else {
            Ok(())
        }
    }
}

impl<T> ToSlice<T> for [T] {
    fn to_slice(&self) -> &[T] {
        self
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn alloc_len(&self) -> usize {
        self.len()
    }

    fn try_resize(&mut self, len: usize) -> VoidResult {
        if len > self.len() {
            Err(ErrorReason::TypeCanNotResize)
        } else {
            Ok(())
        }
    }
}

impl<T> ToSliceMut<T> for [T] {
    fn to_slice_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<'a, T> ToSlice<T> for &'a mut [T] {
    fn to_slice(&self) -> &[T] {
        self
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }

    fn alloc_len(&self) -> usize {
        self.len()
    }

    fn try_resize(&mut self, len: usize) -> VoidResult {
        if len > self.len() {
            Err(ErrorReason::TypeCanNotResize)
        } else {
            Ok(())
        }
    }
}

impl<'a, T> ToSliceMut<T> for &'a mut [T] {
    fn to_slice_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<A: arrayvec::Array> ToSlice<A::Item> for arrayvec::ArrayVec<A>
where
    A::Item: RealNumber,
{
    fn to_slice(&self) -> &[A::Item] {
        self.as_slice()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn alloc_len(&self) -> usize {
        self.capacity()
    }

    fn try_resize(&mut self, len: usize) -> VoidResult {
        if len > self.capacity() {
            return Err(ErrorReason::TypeCanNotResize);
        }

        if len > self.len() {
            while len > self.len() {
                self.push(A::Item::zero());
            }
        } else {
            while len < self.len() {
                self.pop();
            }
        }
        Ok(())
    }
}

impl<A: arrayvec::Array> ToSliceMut<A::Item> for arrayvec::ArrayVec<A>
where
    A::Item: RealNumber,
{
    fn to_slice_mut(&mut self) -> &mut [A::Item] {
        self.as_mut_slice()
    }
}

impl<T> ToSlice<T> for InlineVector<T>
where
    T: RealNumber,
{
    fn to_slice(&self) -> &[T] {
        &self[..]
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn alloc_len(&self) -> usize {
        self.capacity()
    }

    fn try_resize(&mut self, len: usize) -> VoidResult {
        self.try_resize(len)
    }
}

impl<T> ToSliceMut<T> for InlineVector<T>
where
    T: RealNumber,
{
    fn to_slice_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

impl<A: Array> ToDspVector<A::Item> for ArrayVec<A>
where
    A::Item: RealNumber,
{
    fn to_gen_dsp_vec(self, is_complex: bool, domain: DataDomain) -> GenDspVec<Self, A::Item> {
        let mut len = self.len();
        if len % 2 != 0 && is_complex {
            len = 0;
        }
        GenDspVec {
            data: self,
            delta: A::Item::one(),
            domain: meta::TimeOrFreq {
                domain_current: domain,
            },
            number_space: meta::RealOrComplex {
                is_complex_current: is_complex,
            },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_dsp_vec<N, D>(
        self,
        meta_data: &TypeMetaData<A::Item, N, D>,
    ) -> DspVec<Self, A::Item, N, D>
    where
        N: NumberSpace,
        D: Domain,
    {
        let mut len = self.len();
        if len % 2 != 0 && meta_data.is_complex() {
            len = 0;
        }
        DspVec {
            data: self,
            delta: meta_data.delta,
            domain: meta_data.domain.clone(),
            number_space: meta_data.number_space.clone(),
            valid_len: len,
            multicore_settings: meta_data.multicore_settings,
        }
    }
}

impl<A: Array> ToRealVector<A::Item> for ArrayVec<A>
where
    A::Item: RealNumber,
{
    fn to_real_time_vec(self) -> RealTimeVec<Self, A::Item> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: A::Item::one(),
            domain: meta::Time,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, A::Item> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: A::Item::one(),
            domain: meta::Freq,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<A: Array> ToComplexVector<ArrayVec<A>, A::Item> for ArrayVec<A>
where
    A::Item: RealNumber,
{
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, A::Item> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: A::Item::one(),
            domain: meta::Time,
            number_space: meta::Complex,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, A::Item> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: A::Item::one(),
            domain: meta::Freq,
            number_space: meta::Complex,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<T> ToDspVector<T> for InlineVector<T>
where
    T: RealNumber,
{
    fn to_gen_dsp_vec(self, is_complex: bool, domain: DataDomain) -> GenDspVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 && is_complex {
            len = 0;
        }
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: meta::TimeOrFreq {
                domain_current: domain,
            },
            number_space: meta::RealOrComplex {
                is_complex_current: is_complex,
            },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_dsp_vec<N, D>(self, meta_data: &TypeMetaData<T, N, D>) -> DspVec<Self, T, N, D>
    where
        N: NumberSpace,
        D: Domain,
    {
        let mut len = self.len();
        if len % 2 != 0 && meta_data.is_complex() {
            len = 0;
        }
        DspVec {
            data: self,
            delta: meta_data.delta,
            domain: meta_data.domain.clone(),
            number_space: meta_data.number_space.clone(),
            valid_len: len,
            multicore_settings: meta_data.multicore_settings,
        }
    }
}

impl<T> ToRealVector<T> for InlineVector<T>
where
    T: RealNumber,
{
    fn to_real_time_vec(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<T> ToComplexVector<InlineVector<T>, T> for InlineVector<T>
where
    T: RealNumber,
{
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Complex,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Complex,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<'a, T> ToDspVector<T> for &'a [T]
where
    T: RealNumber,
{
    fn to_gen_dsp_vec(self, is_complex: bool, domain: DataDomain) -> GenDspVec<Self, T> {
        let len = self.len();
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: meta::TimeOrFreq {
                domain_current: domain,
            },
            number_space: meta::RealOrComplex {
                is_complex_current: is_complex,
            },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_dsp_vec<N, D>(self, meta_data: &TypeMetaData<T, N, D>) -> DspVec<Self, T, N, D>
    where
        N: NumberSpace,
        D: Domain,
    {
        let mut len = self.len();
        if len % 2 != 0 && meta_data.is_complex() {
            len = 0;
        }
        DspVec {
            data: self,
            delta: meta_data.delta,
            domain: meta_data.domain.clone(),
            number_space: meta_data.number_space.clone(),
            valid_len: len,
            multicore_settings: meta_data.multicore_settings,
        }
    }
}

impl<'a, T> ToRealVector<T> for &'a [T]
where
    T: RealNumber,
{
    fn to_real_time_vec(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<'a, T> ToComplexVector<&'a [T], T> for &'a [T]
where
    T: RealNumber,
{
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Complex,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Complex,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<'a, T> ToComplexVector<&'a [T], T> for &'a [Complex<T>]
where
    T: RealNumber,
{
    fn to_complex_time_vec(self) -> ComplexTimeVec<&'a [T], T> {
        let array = complex_to_array(self);
        let len = array.len();
        ComplexTimeVec {
            data: array,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<&'a [T], T> {
        let array = complex_to_array(self);
        let len = array.len();
        ComplexFreqVec {
            data: array,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<'a, T> ToDspVector<T> for &'a mut [T]
where
    T: RealNumber,
{
    fn to_gen_dsp_vec(self, is_complex: bool, domain: DataDomain) -> GenDspVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 && is_complex {
            len = 0;
        }
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: meta::TimeOrFreq {
                domain_current: domain,
            },
            number_space: meta::RealOrComplex {
                is_complex_current: is_complex,
            },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_dsp_vec<N, D>(self, meta_data: &TypeMetaData<T, N, D>) -> DspVec<Self, T, N, D>
    where
        N: NumberSpace,
        D: Domain,
    {
        let mut len = self.len();
        if len % 2 != 0 && meta_data.is_complex() {
            len = 0;
        }
        DspVec {
            data: self,
            delta: meta_data.delta,
            domain: meta_data.domain.clone(),
            number_space: meta_data.number_space.clone(),
            valid_len: len,
            multicore_settings: meta_data.multicore_settings,
        }
    }
}

impl<'a, T> ToRealVector<T> for &'a mut [T]
where
    T: RealNumber,
{
    fn to_real_time_vec(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Real,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<'a, T> ToComplexVector<&'a mut [T], T> for &'a mut [T]
where
    T: RealNumber,
{
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Complex,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Complex,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<'a, T> ToComplexVector<&'a mut [T], T> for &'a mut [Complex<T>]
where
    T: RealNumber,
{
    fn to_complex_time_vec(self) -> ComplexTimeVec<&'a mut [T], T> {
        let array = complex_to_array_mut(self);
        let len = array.len();
        ComplexTimeVec {
            data: array,
            delta: T::one(),
            domain: meta::Time,
            number_space: meta::Complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<&'a mut [T], T> {
        let array = complex_to_array_mut(self);
        let len = array.len();
        ComplexFreqVec {
            data: array,
            delta: T::one(),
            domain: meta::Freq,
            number_space: meta::Complex,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<'a, T, D> FromVector<T> for DspVec<&'a [T], T, meta::Complex, D>
where
    T: RealNumber,
    D: Domain,
{
    type Output = &'a [Complex<T>];

    fn get(self) -> (Self::Output, usize) {
        let len = self.valid_len / 2;
        (array_to_complex(self.data), len)
    }
}

impl<'a, T, D> FromVector<T> for DspVec<&'a mut [T], T, meta::Complex, D>
where
    T: RealNumber,
    D: Domain,
{
    type Output = &'a mut [Complex<T>];

    fn get(self) -> (Self::Output, usize) {
        let len = self.valid_len / 2;
        (array_to_complex_mut(self.data), len)
    }
}
