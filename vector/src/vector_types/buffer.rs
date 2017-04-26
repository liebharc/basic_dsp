//! Defines the buffers. Finding a good general purpose buffering scheme is hard.
//! So the best option seems to be to create an abstraction so that the buffering can be adjusted
//! to what an application needs.
use numbers::*;
use super::ToSliceMut;

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
