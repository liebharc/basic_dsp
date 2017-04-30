//! Defines the buffers. Finding a good general purpose buffering scheme is hard.
//! So the best option seems to be to create an abstraction so that the buffering can be adjusted
//! to what an application needs.
use numbers::*;
use super::ToSliceMut;
use std::ops::*;

/// A "slice-like" type which also allows to 
pub trait BufferBorrow<S: ToSliceMut<T>, T: RealNumber> : DerefMut<Target=[T]> {
    /// Moves the content of this slice into `storage`.
    /// This operation might just copy all contents into `storage` or 
    fn swap(self, storage: &mut S);
}

/// A buffer which can be used by other types. Types will call buffers to create new arrays.
/// A buffer may can implement any buffering strategy.
pub trait BufferNew<'a, S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// The type of the burrow which is returned.
    type Borrow: BufferBorrow<S, T>;

    /// Asks the buffer for new storage of exactly size `len`.
    /// S doesn't need to have be initialized with any default value.
    fn get(&'a mut self, len: usize) -> Self::Borrow;

    /// Asks the buffer for newly created storage which isn't buffered.
    ///
    /// The purpose if this method is to abstract the creation of a
    /// certain storage type.
    fn construct_new(&mut self, len: usize) -> S;

    /// Returns the allocated length of all storage within this buffer.
    fn alloc_len(&self) -> usize;
}
