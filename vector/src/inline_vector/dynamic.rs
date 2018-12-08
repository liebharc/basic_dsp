use arrayvec::*;
use crate::numbers::*;
use std::iter::FromIterator;
use std::ops::*;
use std::slice::{Iter, IterMut};
use std::usize;
use crate::{ErrorReason, VoidResult};

/// A type which internally switches between stack and heap allocation.
/// This is supposed to perform faster but the main reason is that this
/// way we automatically have a limited stack allocation available on systems
/// without heap, and on systems with heap allocation we don't have to worry
/// about introducing artifical limits.
pub enum InlineVector<T> {
    Inline(ArrayVec<[T; 64]>),
    Dynamic(Vec<T>),
}

impl<T> InlineVector<T>
where
    T: Copy,
{
    /// Create a new vector of a given size filled with a given default value.
    pub fn of_size(default: T, n: usize) -> InlineVector<T> {
        let mut result = Self::with_capacity(n);
        for _ in 0..n {
            result.push(default);
        }

        result
    }
}

/// Most of the operations behave as defined for `std::Vec`.
impl<T> InlineVector<T> {
    /// Returns the maximimum size the vector can possibly have.
    pub fn max_capacity() -> usize {
        usize::MAX
    }

    /// Returns a vector with a given capacity.
    pub fn with_capacity(n: usize) -> InlineVector<T> {
        if n <= 64 {
            InlineVector::Inline(ArrayVec::<[T; 64]>::new())
        } else {
            InlineVector::Dynamic(Vec::with_capacity(n))
        }
    }

    /// Returns a vector with a default capacity. The default
    /// capacity will be relativly small and should only be used
    /// to store small lookup tables or results.
    ///
    /// As of today the default capacity is `64`.
    pub fn with_default_capcacity() -> InlineVector<T> {
        Self::with_capacity(64)
    }

    /// Returns a vector holding a single element.
    pub fn with_elem(elem: T) -> InlineVector<T> {
        let mut vector = Self::with_capacity(1);
        vector.push(elem);
        vector
    }

    /// Returns an empty vector.
    pub fn empty() -> InlineVector<T> {
        Self::with_capacity(0)
    }

    pub fn push(&mut self, elem: T) {
        match *self {
            InlineVector::Inline(ref mut v) => {
                v.push(elem);
            }
            InlineVector::Dynamic(ref mut v) => v.push(elem),
        };
    }

    pub fn pop(&mut self) -> Option<T> {
        match *self {
            InlineVector::Inline(ref mut v) => v.pop(),
            InlineVector::Dynamic(ref mut v) => v.pop(),
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        match *self {
            InlineVector::Inline(ref mut v) => v.remove(index),
            InlineVector::Dynamic(ref mut v) => v.remove(index),
        }
    }

    pub fn len(&self) -> usize {
        match *self {
            InlineVector::Inline(ref v) => v.len(),
            InlineVector::Dynamic(ref v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        match *self {
            InlineVector::Inline(ref v) => v.capacity(),
            InlineVector::Dynamic(ref v) => v.capacity(),
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<T> {
        match *self {
            InlineVector::Inline(ref v) => v.iter(),
            InlineVector::Dynamic(ref v) => v.iter(),
        }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        match *self {
            InlineVector::Inline(ref mut v) => v.iter_mut(),
            InlineVector::Dynamic(ref mut v) => v.iter_mut(),
        }
    }

    pub fn append(&mut self, other: &mut Self) {
        while !other.is_empty() {
            self.push(other.remove(0));
        }
    }

    pub fn insert(&mut self, index: usize, element: T) {
        match *self {
            InlineVector::Inline(ref mut v) => {
                v.insert(index, element);
            }
            InlineVector::Dynamic(ref mut v) => v.insert(index, element),
        }
    }
}

impl<T: Zero + Clone> InlineVector<T> {
    pub fn try_resize(&mut self, len: usize) -> VoidResult {
        match *self {
            InlineVector::Inline(ref v) => {
                if v.capacity() >= len {
                    Ok(())
                } else {
                    Err(ErrorReason::TypeCanNotResize)
                }
            }
            InlineVector::Dynamic(ref mut v) => {
                if v.capacity() >= len {
                    v.resize(len, T::zero());
                    Ok(())
                } else {
                    // We could increase the vector capacity, but then
                    // Inline and Dynamic would behave very different and we want
                    // to avoid that
                    Err(ErrorReason::TypeCanNotResize)
                }
            }
        }
    }
}

impl<T> Index<usize> for InlineVector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match *self {
            InlineVector::Inline(ref v) => &v[index],
            InlineVector::Dynamic(ref v) => &v[index],
        }
    }
}

impl<T> IndexMut<usize> for InlineVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match *self {
            InlineVector::Inline(ref mut v) => &mut v[index],
            InlineVector::Dynamic(ref mut v) => &mut v[index],
        }
    }
}

impl<T> Index<RangeFull> for InlineVector<T> {
    type Output = [T];

    fn index(&self, _index: RangeFull) -> &[T] {
        match *self {
            InlineVector::Inline(ref v) => &v[..],
            InlineVector::Dynamic(ref v) => &v[..],
        }
    }
}

impl<T> IndexMut<RangeFull> for InlineVector<T> {
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        match *self {
            InlineVector::Inline(ref mut v) => &mut v[..],
            InlineVector::Dynamic(ref mut v) => &mut v[..],
        }
    }
}

impl<T> Index<RangeFrom<usize>> for InlineVector<T> {
    type Output = [T];

    fn index(&self, index: RangeFrom<usize>) -> &[T] {
        match *self {
            InlineVector::Inline(ref v) => &v[index],
            InlineVector::Dynamic(ref v) => &v[index],
        }
    }
}

impl<T> IndexMut<RangeFrom<usize>> for InlineVector<T> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
        match *self {
            InlineVector::Inline(ref mut v) => &mut v[index],
            InlineVector::Dynamic(ref mut v) => &mut v[index],
        }
    }
}

impl<T> Index<RangeTo<usize>> for InlineVector<T> {
    type Output = [T];

    fn index(&self, index: RangeTo<usize>) -> &[T] {
        match *self {
            InlineVector::Inline(ref v) => &v[index],
            InlineVector::Dynamic(ref v) => &v[index],
        }
    }
}

impl<T> IndexMut<RangeTo<usize>> for InlineVector<T> {
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
        match *self {
            InlineVector::Inline(ref mut v) => &mut v[index],
            InlineVector::Dynamic(ref mut v) => &mut v[index],
        }
    }
}

impl<T: Clone> Clone for InlineVector<T> {
    fn clone(&self) -> Self {
        match *self {
            InlineVector::Inline(ref v) => InlineVector::Inline(v.clone()),
            InlineVector::Dynamic(ref v) => InlineVector::Dynamic(v.clone()),
        }
    }
}

impl<T> FromIterator<T> for InlineVector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut c = InlineVector::with_capacity(64);

        for i in iter {
            c.push(i);
        }

        c
    }
}
