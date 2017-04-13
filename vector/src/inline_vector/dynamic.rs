use std::ops::*;
use arrayvec::*;
use {VoidResult, ErrorReason, Buffer};
use numbers::*;
use std::mem;
use std::iter::FromIterator;
use std::slice::Iter;

/// A type which internally switches between stack and heap allocation.
/// This is supposed to perform faster but the main reason is that this
/// way we automatically have a limited stack allocation available on systems
/// without heap, and on systems with heap allocation we don't have to worry
/// about introducing artifical limits.
///
/// Thanks to: http://stackoverflow.com/questions/27859822/alloca-variable-length-arrays-in-rust
pub enum InlineVector<T> {
    Inline(ArrayVec<[T; 64]>),
    Dynamic(Vec<T>),
}

impl<T> InlineVector<T>
    where T: Copy {
    pub fn of_size(default: T, n: usize) -> InlineVector<T> {
        let mut result = Self::with_capacity(n);
        for _ in 0..n {
            result.push(default);
        }

        result
    }
}

impl<T> InlineVector<T> {
    pub fn with_capacity(n: usize) -> InlineVector<T> {
        if n <= 64 {
            InlineVector::Inline(ArrayVec::<[T; 64]>::new())
        } else {
            InlineVector::Dynamic(
                Vec::with_capacity(n)
            )
        }
    }

    pub fn with_elem(elem: T) -> InlineVector<T> {
        let mut vector = Self::with_capacity(1);
        vector.push(elem);
        vector
    }

    pub fn empty() -> InlineVector<T> {
        Self::with_capacity(0)
    }

    pub fn push(&mut self, elem: T) {
        match self {
            &mut InlineVector::Inline(ref mut v) => {
                let res = v.push(elem);
                if res.is_some() {
                    panic!("InlineVector capacity exceeded, please open a defect against `basic_dsp`");
                }
            },
            &mut InlineVector::Dynamic(ref mut v) => v.push(elem)
        };
    }

    pub fn pop(&mut self) -> Option<T> {
        match self {
            &mut InlineVector::Inline(ref mut v) => {
                v.pop()
            },
            &mut InlineVector::Dynamic(ref mut v) => v.pop()
        }
    }

    pub fn len(&self) -> usize {
        match self {
            &InlineVector::Inline(ref v) => v.len(),
            &InlineVector::Dynamic(ref v) => v.len()
        }
    }

    pub fn capacity(&self) -> usize {
        match self {
            &InlineVector::Inline(ref v) => v.capacity(),
            &InlineVector::Dynamic(ref v) => v.capacity()
        }
    }

	pub fn iter(&self) -> Iter<T> {
		match self {
            &InlineVector::Inline(ref v) => v.iter(),
            &InlineVector::Dynamic(ref v) => v.iter()
        }
	}
}

impl<T: Zero + Clone> InlineVector<T> {
    pub fn try_resize(&mut self, len: usize) -> VoidResult {
        match self {
            &mut InlineVector::Inline(ref v) => {
                if v.capacity() >= len {
                    Ok(())
                } else {
                    Err(ErrorReason::TypeCanNotResize)
                }
            },
            &mut InlineVector::Dynamic(ref mut v) => {
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

/// A buffer which stores a single inline vector and never shrinks.
pub struct InternalBuffer<T>
    where T: RealNumber
{
    temp: InlineVector<T>,
}

impl<T> Index<usize> for InlineVector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match self {
            &InlineVector::Inline(ref v) => &v[index],
            &InlineVector::Dynamic(ref v) => &v[index]
        }
    }
}

impl<T> IndexMut<usize> for InlineVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match self {
            &mut InlineVector::Inline(ref mut v) => &mut v[index],
            &mut InlineVector::Dynamic(ref mut v) => &mut v[index]
        }
    }
}

impl<T> Index<RangeFull> for InlineVector<T> {
    type Output = [T];

    fn index(&self, _index: RangeFull) -> &[T] {
        match self {
            &InlineVector::Inline(ref v) => &v[..],
            &InlineVector::Dynamic(ref v) => &v[..]
        }
    }
}

impl<T> IndexMut<RangeFull> for InlineVector<T> {
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        match self {
            &mut InlineVector::Inline(ref mut v) => &mut v[..],
            &mut InlineVector::Dynamic(ref mut v) => &mut v[..]
        }
    }
}

impl<T> Index<RangeFrom<usize>> for InlineVector<T> {
    type Output = [T];

    fn index(&self, index: RangeFrom<usize>) -> &[T] {
        match self {
            &InlineVector::Inline(ref v) => &v[index],
            &InlineVector::Dynamic(ref v) => &v[index]
        }
    }
}

impl<T> IndexMut<RangeFrom<usize>> for InlineVector<T> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
        match self {
            &mut InlineVector::Inline(ref mut v) => &mut v[index],
            &mut InlineVector::Dynamic(ref mut v) => &mut v[index]
        }
    }
}

impl<T> Index<RangeTo<usize>> for InlineVector<T> {
    type Output = [T];

    fn index(&self, index: RangeTo<usize>) -> &[T] {
        match self {
            &InlineVector::Inline(ref v) => &v[index],
            &InlineVector::Dynamic(ref v) => &v[index]
        }
    }
}

impl<T> IndexMut<RangeTo<usize>> for InlineVector<T> {
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
        match self {
            &mut InlineVector::Inline(ref mut v) => &mut v[index],
            &mut InlineVector::Dynamic(ref mut v) => &mut v[index]
        }
    }
}

impl<T: Clone> Clone for InlineVector<T> {
    fn clone(&self) -> Self {
         match self {
            &InlineVector::Inline(ref v) => InlineVector::Inline(v.clone()),
            &InlineVector::Dynamic(ref v) => InlineVector::Dynamic(v.clone())
        }
    }
}

impl<T> FromIterator<T> for InlineVector<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let mut c = InlineVector::with_capacity(64);

        for i in iter {
            c.push(i);
        }

        c
    }
}

impl<T> InternalBuffer<T>
    where T: RealNumber
{
    /// Creates a new buffer which is ready to be passed around.
    pub fn new() -> InternalBuffer<T> {
        InternalBuffer { temp: InlineVector::with_capacity(0) }
    }
}

impl<T> Buffer<InlineVector<T>, T> for InternalBuffer<T>
    where T: RealNumber
{
    fn get(&mut self, len: usize) -> InlineVector<T> {
        if len <= self.temp.len() {
            let mut result = InlineVector::with_capacity(0);
            mem::swap(&mut result, &mut self.temp);
            return result;
        }

        InlineVector::of_size(T::zero(), len)
    }

    fn construct_new(&mut self, len: usize) -> InlineVector<T> {
        InlineVector::of_size(T::zero(), len)
    }

    fn free(&mut self, storage: InlineVector<T>) {
        if storage.len() > self.temp.len() {
            self.temp = storage;
        }
    }

    fn alloc_len(&self) -> usize {
        self.temp.capacity()
    }
}
