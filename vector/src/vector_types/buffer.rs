//! Defines the buffers. Finding a good general purpose buffering scheme is hard.
//! So the best option seems to be to create an abstraction so that the buffering can be adjusted
//! to what an application needs.

use RealNumber;
use super::ToSliceMut;
use std::mem;
use arrayvec;

/// A buffer which can be used by other types. Types will call buffers to create new arrays.
/// A buffer may can implement any buffering strategy.
pub trait Buffer<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Asks the buffer for new storage.
    /// `len` is the minimum size the storage needs to provide.
    /// S doesn't need to have be initialized with any default value.
    fn get(&mut self, len: usize) -> S;

    /// Asks the buffer for newly created storage which isn't buffered.
    ///
    /// The purpose if this method is to abstract the creation of a
    /// certain storage type.
    fn construct_new(&mut self, len: usize) -> S;

    /// Returns a storage to the buffer. The buffer isn't required to free it and might just
    /// reuse the storage.
    fn free(&mut self, storage: S);

    /// Returns the allocated length of all storage within this buffer.
    fn alloc_len(&self) -> usize;
}

/// A buffer which stores a single vector and never shrinks.
pub struct SingleBuffer<T>
    where T: RealNumber
{
    temp: Vec<T>,
}

#[cfg(feature="std")]
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

#[cfg(feature="std")]
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

/// A buffer which stores a single vector and never shrinks.
pub struct SingleBufferArrayVec<A: arrayvec::Array>
     where A::Item: RealNumber
{
    temp: Option<arrayvec::ArrayVec<A>>,
}

#[cfg(feature="std")]
impl<A: arrayvec::Array> SingleBufferArrayVec<A>
     where A::Item: RealNumber
{
    /// Creates a new buffer which is ready to be passed around.
    pub fn new() -> SingleBufferArrayVec<A> {
        SingleBufferArrayVec { temp: None }
    }
}

#[cfg(feature="std")]
impl<A: arrayvec::Array> Buffer<arrayvec::ArrayVec<A>, A::Item> for SingleBufferArrayVec<A>
    where A::Item: RealNumber
{
    fn get(&mut self, len: usize) -> arrayvec::ArrayVec<A> {
        let result =
            if self.temp.is_some() {
                let mut result = None;
                mem::swap(&mut result, &mut self.temp);
                result.unwrap()
            } else {
                 arrayvec::ArrayVec::new()
            };
        assert!(len <= result.len());
        result
    }

    fn construct_new(&mut self, len: usize) -> arrayvec::ArrayVec<A> {
        let result = arrayvec::ArrayVec::new();
        assert!(len <= result.len());
        result
    }

    fn free(&mut self, storage: arrayvec::ArrayVec<A>) {
        if self.temp.is_none() {
            self.temp = Some(storage);
        }
    }

    fn alloc_len(&self) -> usize {
        match self.temp {
            Some(ref a) => a.capacity(),
            None => 0
        }
    }
}
