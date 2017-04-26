//! Requirements which a type needs to fulfill
//! so that it can serve as a vector storage
use super::VoidResult;

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
