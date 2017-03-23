//! Requirements which a type needs to fulfill
//! so that it can serve as a vector storage
use RealNumber;
use super::{VoidResult, ErrorReason};
use arrayvec;
use num::Zero;

/// A trait to convert a type into a slice.
pub trait ToSlice<T> {
    /// Convert to a slice.
    fn to_slice(&self) -> &[T];

    /// Length of a slice.
    fn len(&self) -> usize;

    /// Indicates whether or not this storage type is empty.
    fn is_empty(&self) -> bool;

    /// Gets the allocated length of a storage.
    /// It's expected that `self.alloc_len() >= self.len()`
    /// in all cases.
    fn alloc_len(&self) -> usize;

    /// Resizes the storage to support at least `len` elements or
    /// returns an error if resizing isn't supported.
    fn try_resize(&mut self, len: usize) -> VoidResult;
}

/// A trait to convert a type into a mutable slice.
pub trait ToSliceMut<T>: ToSlice<T> {
    /// Convert to a mutable slice.
    fn to_slice_mut(&mut self) -> &mut [T];
}

/// A trait for storage types which are known to have the capability to increase their capacity.
pub trait Resize {
    /// Resize a storage type. Must work for any value of `len`,
    /// however it's okay if after this method `self.alloc_len() > len`
    /// or in words: It's okay if the method allocates more memory than
    /// specified in the parameter.
    fn resize(&mut self, len: usize);
}

/// A marker trait which states the the type owns its storage.
///
/// Some operations would be possible on data even if doesn't
/// own the storage (usually a slice). However the result
/// would be likely not as desired. E.g. if suddenly half
/// the storage is in real number space and the other half in
/// complex.
pub trait Owner {}

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

impl<T> Owner for [T] {}

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

impl<T> Owner for Box<[T]> {}

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

impl<T> Owner for Vec<T> {}


impl<A: arrayvec::Array> ToSlice<A::Item> for arrayvec::ArrayVec<A>
    where A::Item: RealNumber
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
        }
        else {
            while len < self.len() {
                self.pop();
            }
        }
        Ok(())
    }
}

impl<A: arrayvec::Array> ToSliceMut<A::Item> for arrayvec::ArrayVec<A>
    where A::Item: RealNumber
{
    fn to_slice_mut(&mut self) -> &mut [A::Item] {
        self.as_mut_slice()
    }
}

impl<A: arrayvec::Array> Owner for arrayvec::ArrayVec<A> {}
