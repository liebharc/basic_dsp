//! Requirements which a type needs to fulfill
//! so that it can serve as a vector storage
use RealNumber;
use super::DspVec;

/// A trait to convert a type into a slice.
pub trait ToSlice<T> {
    /// Convert to a slice.
    fn to_slice(&self) -> &[T];

    /// Length of a slice.
    fn len(&self) -> usize;
}

/// A trait to convert a type into a mutable slice.
pub trait ToSliceMut<T> : ToSlice<T> {
    /// Convert to a mutable slice.
    fn to_slice_mut(&mut self) -> &mut [T];
}

/// A trait to resize a storage type.
pub trait Resize {
    /// Resize a storage type. Must work for any value of `len`,
    /// however it's okay if after this method `self.alloc_len() > len`
    /// or in words: It's okay if the method allocates more memory than
    /// specified in the parameter.
    fn resize(&mut self, len: usize);

    /// Gets the allocated length of a storage.
    /// It's expected that `self.alloc_len() >= self.len()`
    /// in all cases.
    fn alloc_len(&self) -> usize;
}

impl<'a, T> ToSlice<T> for &'a [T] {
    fn to_slice(&self) -> &[T] {
        self
    }

    fn len(&self) -> usize {
        (*self).len()
    }
}

impl<T> ToSlice<T> for [T] {
    fn to_slice(&self) -> &[T] {
        self
    }

    fn len(&self) -> usize {
        self.len()
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
}

impl<'a, T> ToSliceMut<T> for &'a mut [T] {
    fn to_slice_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T> ToSlice<T> for Vec<T> {
    fn to_slice(&self) -> &[T] {
        &self[..]
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> ToSliceMut<T> for Vec<T> {
    fn to_slice_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

impl<T> Resize for Vec<T>
    where T: RealNumber {
    fn resize(&mut self, len: usize) {
        self.resize(len, T::zero());
    }

    fn alloc_len(&self) -> usize {
        self.capacity()
    }
}

impl<S, T> ToSlice<T> for DspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber {
    fn to_slice(&self) -> &[T] {
        self.data.to_slice()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<S, T> ToSliceMut<T> for DspVec<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber {
    fn to_slice_mut(&mut self) -> &mut [T] {
        self.data.to_slice_mut()
    }
}
