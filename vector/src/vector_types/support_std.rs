///! Support for types in Rust std
use numbers::*;
use std::mem;
use std::ops::*;
use std::result;
use super::{round_len, DataDomain, NumberSpace, Domain, ErrorReason, DspVec, GenDspVec,
            RealTimeVec, RealFreqVec, ComplexTimeVec, ComplexFreqVec, RealData, ComplexData,
            RealOrComplexData, TimeData, FrequencyData, TimeOrFrequencyData, ToSlice,
            TypeMetaData, MetaData, BufferBorrow, BufferNew};
use super::{ToComplexVector, ToRealVector, ToDspVector, ToSliceMut,
            VoidResult, Resize, Buffer};
use multicore_support::MultiCoreSettings;

/// Conversion from two instances of a generic data type into a dsp vector with complex data.
pub trait InterleaveToVector<T>: ToSlice<T>
    where T: RealNumber
{
    /// Create a new vector in real number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn interleave_to_complex_time_vec(&self,
                                      other: &Self)
                                      -> result::Result<ComplexTimeVec<Vec<T>, T>, ErrorReason>;

    /// Create a new vector in real number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn interleave_to_complex_freq_vec(&self,
                                      other: &Self)
                                      -> result::Result<ComplexFreqVec<Vec<T>, T>, ErrorReason>;
}

/// Buffer borrow type for `SingleBuffer`.
pub struct SingleBufferBurrow<'a, T: RealNumber + 'a> {
    owner: &'a mut SingleBuffer<T>,
    len: usize
}

impl<'a, T: RealNumber> Deref for SingleBufferBurrow<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.owner.temp[0..self.len]
    }
}

impl<'a, T: RealNumber> DerefMut for SingleBufferBurrow<'a, T> {
    fn deref_mut(&mut self) -> &mut[T] {
        &mut self.owner.temp[0..self.len]
    }
}

impl<'a, T: RealNumber> BufferBorrow<Vec<T>, T> for SingleBufferBurrow<'a, T> {  
    fn swap(self, storage: &mut Vec<T>) {
        mem::swap(&mut self.owner.temp, storage);
    }
}

/// A buffer which stores a single vector and never shrinks.
pub struct SingleBuffer<T>
    where T: RealNumber
{
    temp: Vec<T>,
}

impl<T> SingleBuffer<T>
    where T: RealNumber
{
    /// Creates a new buffer which is ready to be passed around.
    pub fn new() -> SingleBuffer<T> {
        SingleBuffer { temp: Vec::new() }
    }

    /// Creates a new buffer which is ready to be passed around.
    pub fn with_capacity(len: usize) -> SingleBuffer<T> {
        SingleBuffer { temp: vec![T::zero(); len] }
    }
}

impl<T> Buffer<Vec<T>, T> for SingleBuffer<T>
    where T: RealNumber
{
    fn get(&mut self, len: usize) -> Vec<T> {
        if len <= self.temp.len() {
            let mut result = Vec::new();
            mem::swap(&mut result, &mut self.temp);
            return result;
        }

        vec![T::zero(); len]
    }

    fn construct_new(&mut self, len: usize) -> Vec<T> {
        vec![T::zero(); len]
    }

    fn free(&mut self, storage: Vec<T>) {
        if storage.len() > self.temp.len() {
            self.temp = storage;
        }
    }

    fn alloc_len(&self) -> usize {
        self.temp.capacity()
    }
}

impl<'a, T> BufferNew<'a, Vec<T>, T> for SingleBuffer<T>
    where T: RealNumber + 'a
{
    type Borrow = SingleBufferBurrow<'a, T>;

    fn get(&'a mut self, len: usize) -> Self::Borrow {
        if self.temp.len() < len {
            self.temp = vec![T::zero(); len];
        }
        
        SingleBufferBurrow { 
            owner: self,
            len: len
        }
    }

    fn construct_new(&mut self, len: usize) -> Vec<T> {
        vec![T::zero(); len]
    }

    fn alloc_len(&self) -> usize {
        self.temp.capacity()
    }
}

/// This type can be used everytime the API asks for a buffer to disable any buffering.
pub struct NoBuffer;

/// Buffer borrow type for `NoBuffer`.
pub struct NoBufferBurrow<T: RealNumber> {
    data: Vec<T>
}

impl<T: RealNumber> Deref for NoBufferBurrow<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.data
    }
}

impl<T: RealNumber> DerefMut for NoBufferBurrow<T> {
    fn deref_mut(&mut self) -> &mut[T] {
        &mut self.data
    }
}

impl<T: RealNumber> BufferBorrow<Vec<T>, T> for NoBufferBurrow<T> {  
    fn swap(mut self, storage: &mut Vec<T>) {
        mem::swap(&mut self.data, storage);
    }
}

impl<'a, T> BufferNew<'a, Vec<T>, T> for NoBuffer
    where T: RealNumber + 'a
{
    type Borrow = NoBufferBurrow<T>;

    fn get(&'a mut self, len: usize) -> Self::Borrow {
        NoBufferBurrow { 
            data: vec![T::zero(); len]
        }
    }

    fn construct_new(&mut self, len: usize) -> Vec<T> {
        vec![T::zero(); len]
    }

    fn alloc_len(&self) -> usize {
        0
    }
}

impl<T> Buffer<Vec<T>, T> for NoBuffer
    where T: RealNumber
{
    fn get(&mut self, len: usize) -> Vec<T> {
        vec![T::zero(); len]
    }

    fn construct_new(&mut self, len: usize) -> Vec<T> {
        vec![T::zero(); len]
    }

    fn free(&mut self, _: Vec<T>) { }

    fn alloc_len(&self) -> usize {
        0
    }
}

/// A vector with real numbers in time domain.
pub type RealTimeVec32 = DspVec<Vec<f32>, f32, RealData, TimeData>;
/// A vector with real numbers in frequency domain.
pub type RealFreqVec32 = DspVec<Vec<f32>, f32, RealData, FrequencyData>;
/// A vector with complex numbers in time domain.
pub type ComplexTimeVec32 = DspVec<Vec<f32>, f32, ComplexData, TimeData>;
/// A vector with complex numbers in frequency domain.
pub type ComplexFreqVec32 = DspVec<Vec<f32>, f32, ComplexData, FrequencyData>;
/// A vector with no information about number space or domain at compile time.
pub type GenDspVec32 = DspVec<Vec<f32>, f32, RealOrComplexData, TimeOrFrequencyData>;

/// A vector with real numbers in time domain.
pub type RealTimeVec64 = DspVec<Vec<f64>, f64, RealData, TimeData>;
/// A vector with real numbers in frequency domain.
pub type RealFreqVec64 = DspVec<Vec<f64>, f64, RealData, FrequencyData>;
/// A vector with complex numbers in time domain.
pub type ComplexTimeVec64 = DspVec<Vec<f64>, f64, ComplexData, TimeData>;
/// A vector with complex numbers in frequency domain.
pub type ComplexFreqVec64 = DspVec<Vec<f64>, f64, ComplexData, FrequencyData>;
/// A vector with no information about number space or domain at compile time.
pub type GenDspVec64 = DspVec<Vec<f64>, f64, RealOrComplexData, TimeOrFrequencyData>;

impl<T> ToSlice<T> for Vec<T>
    where T: RealNumber
{
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
        self.capacity()
    }

    fn try_resize(&mut self, len: usize) -> VoidResult {
        self.resize(len, T::zero());
        Ok(())
    }
}

impl<T> ToSliceMut<T> for Vec<T>
    where T: RealNumber
{
    fn to_slice_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

impl<T> Resize for Vec<T>
    where T: RealNumber
{
    fn resize(&mut self, len: usize) {
        self.resize(len, T::zero());
    }
}

impl<T> ToSlice<T> for Box<[T]> {
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

impl<T> ToSliceMut<T> for Box<[T]> {
    fn to_slice_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T> ToDspVector<T> for Vec<T>
    where T: RealNumber
{
    fn to_gen_dsp_vec(self, is_complex: bool, domain: DataDomain) -> GenDspVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 && is_complex {
            len = 0;
        }
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: TimeOrFrequencyData { domain_current: domain },
            number_space: RealOrComplexData { is_complex_current: is_complex },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_dsp_vec<N, D>(self, meta_data: &TypeMetaData<T, N, D>) -> DspVec<Self, T, N, D>
        where N: NumberSpace, D: Domain {
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
            multicore_settings: meta_data.multicore_settings
        }
    }
}

impl<T> ToRealVector<T> for Vec<T>
    where T: RealNumber
{
    fn to_real_time_vec(mut self) -> RealTimeVec<Self, T> {
        let len = self.len();
        expand_to_full_capacity(&mut self);
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_real_freq_vec(mut self) -> RealFreqVec<Self, T> {
        let len = self.len();
        expand_to_full_capacity(&mut self);
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<T> ToComplexVector<Vec<T>, T> for Vec<T>
    where T: RealNumber
{
    fn to_complex_time_vec(mut self) -> ComplexTimeVec<Self, T> {
        let len = self.len();
        expand_to_full_capacity(&mut self);
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(mut self) -> ComplexFreqVec<Self, T> {
        let len = self.len();
        expand_to_full_capacity(&mut self);
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: if len % 2 == 0 { len } else { 0 },
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<T> ToComplexVector<Vec<T>, T> for Vec<Complex<T>>
    where T: RealNumber
{
    fn to_complex_time_vec(self) -> ComplexTimeVec<Vec<T>, T> {
        let len = self.len();
        let vec = complex_vec_to_interleaved_vec(self);
        ComplexTimeVec {
            data: vec,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: 2 * len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Vec<T>, T> {
        let len = self.len();
        let vec = complex_vec_to_interleaved_vec(self);
        ComplexFreqVec {
            data: vec,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: 2 * len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<T> ToDspVector<T> for Box<[T]>
    where T: RealNumber
{
    fn to_gen_dsp_vec(self, is_complex: bool, domain: DataDomain) -> GenDspVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 && is_complex {
            len = 0;
        }
        GenDspVec {
            data: self,
            delta: T::one(),
            domain: TimeOrFrequencyData { domain_current: domain },
            number_space: RealOrComplexData { is_complex_current: is_complex },
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_dsp_vec<N, D>(self, meta_data: &TypeMetaData<T, N, D>) -> DspVec<Self, T, N, D>
        where N: NumberSpace, D: Domain {
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
            multicore_settings: meta_data.multicore_settings
        }
    }
}

impl<T> ToRealVector<T> for Box<[T]>
    where T: RealNumber
{
    fn to_real_time_vec(self) -> RealTimeVec<Self, T> {
        let len = self.len();
        RealTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_real_freq_vec(self) -> RealFreqVec<Self, T> {
        let len = self.len();
        RealFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: RealData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<T> ToComplexVector<Box<[T]>, T> for Box<[T]>
    where T: RealNumber
{
    fn to_complex_time_vec(self) -> ComplexTimeVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexTimeVec {
            data: self,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }

    fn to_complex_freq_vec(self) -> ComplexFreqVec<Self, T> {
        let mut len = self.len();
        if len % 2 != 0 {
            len = 0;
        }
        ComplexFreqVec {
            data: self,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: len,
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<Type, T> InterleaveToVector<T> for Type
    where Type: ToSlice<T>,
          T: RealNumber
{
    fn interleave_to_complex_time_vec(&self,
                                      other: &Self)
                                      -> result::Result<ComplexTimeVec<Vec<T>, T>, ErrorReason> {
        if self.len() != other.len() {
            return Err(ErrorReason::InputMustHaveTheSameSize);
        }

        let rounded_len = round_len(self.len() + other.len());
        let mut data = Vec::with_capacity(rounded_len);

        let len = self.len();
        let real = self.to_slice();
        let imag = other.to_slice();
        for i in 0..len {
            data.push(real[i]);
            data.push(imag[i]);
        }

        let data_length = data.len();

        Ok(ComplexTimeVec {
            data: data,
            delta: T::one(),
            domain: TimeData,
            number_space: ComplexData,
            valid_len: data_length,
            multicore_settings: MultiCoreSettings::default(),
        })
    }

    fn interleave_to_complex_freq_vec(&self,
                                      other: &Self)
                                      -> result::Result<ComplexFreqVec<Vec<T>, T>, ErrorReason> {
        if self.len() != other.len() {
            return Err(ErrorReason::InputMustHaveTheSameSize);
        }

        let rounded_len = round_len(self.len() + other.len());
        let mut data = Vec::with_capacity(rounded_len);

        let len = self.len();
        let real = self.to_slice();
        let imag = other.to_slice();
        for i in 0..len {
            data.push(real[i]);
            data.push(imag[i]);
        }

        let data_length = data.len();

        Ok(ComplexFreqVec {
            data: data,
            delta: T::one(),
            domain: FrequencyData,
            number_space: ComplexData,
            valid_len: data_length,
            multicore_settings: MultiCoreSettings::default(),
        })
    }
}

fn expand_to_full_capacity<T>(vec: &mut Vec<T>)
    where T: Zero
{
    while vec.len() < vec.capacity() {
        vec.push(T::zero());
    }
}

fn complex_vec_to_interleaved_vec<T>(mut vec: Vec<Complex<T>>) -> Vec<T>
    where T: RealNumber
{
    use std::mem;

    expand_to_full_capacity(&mut vec);
    let boxed = vec.into_boxed_slice();
    let len = boxed.len();
    unsafe {
        let mut trans: Box<[T]> = mem::transmute(boxed);
        let vec = Vec::<T>::from_raw_parts(&mut trans[0] as *mut T, len * 2, len * 2);
        mem::forget(trans); // TODO memory leak?
        vec
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;
    use super::complex_vec_to_interleaved_vec;

    #[test]
    fn complex_vec_to_interleaved_vec_test() {
        let complex = vec!(Complex32::new(0.0, 0.0); 5);
        let real = complex_vec_to_interleaved_vec(complex);
        assert_eq!(real.len(), 10);
        assert_eq!(&real[..],
                   &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

}
