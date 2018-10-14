use num_complex::Complex;
use std::marker::PhantomData;

/// Holds information about how to partition the access to a slice to get aligned
/// reads/writes to the largest portion of the slice.
pub struct SimdPartition<T> {
    /// Left part of the slice which must not be accessed with SIMD operations
    pub left: usize,
    /// Right  part of the slice which must not be accessed with SIMD operations
    right: usize,
    len: usize,
    data_type: PhantomData<T>,
}

/// Iterator around the left and right side of a vector.
pub struct EdgeIteratorMut<'a, T: 'a> {
    pos: *mut T,
    left: *mut T,
    right: *mut T,
    end: *mut T,
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T> EdgeIteratorMut<'a, T> {
    pub fn new(slice: &mut [T], left: usize, right: usize) -> impl Iterator<Item = &mut T> {
        let start = slice.as_mut_ptr();
        let len = slice.len() as isize;
        let left = left as isize;
        let right = right as isize;
        unsafe {
            EdgeIteratorMut {
                pos: start,
                left: start.offset(left - 1),
                right: start.offset(len - right),
                end: start.offset(len - 1),
                _marker: std::marker::PhantomData,
            }
        }
    }
}

impl<'a, T> Iterator for EdgeIteratorMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        unsafe {
            // Jump from end of left to right.
            if self.pos > self.left && self.pos < self.right {
                self.pos = self.right;
            }

            if self.pos > self.end {
                return None;
            } else {
                let value = &mut *self.pos;
                self.pos = self.pos.offset(1);
                Some(value)
            }
        }
    }
}

/// Iterator with index around the left and right side of a vector.
pub struct IndexedEdgeIteratorMut<'a, T: 'a> {
    pos: *mut T,
    idx: isize,
    left: *mut T,
    right_idx: isize,
    right: *mut T,
    end: *mut T,
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T> IndexedEdgeIteratorMut<'a, T> {
    pub fn new(
        slice: &mut [T],
        left: usize,
        right: usize,
    ) -> impl Iterator<Item = (isize, &mut T)> {
        let start = slice.as_mut_ptr();
        let len = slice.len() as isize;
        let left = left as isize;
        let right = right as isize;
        unsafe {
            IndexedEdgeIteratorMut {
                pos: start,
                idx: 0,
                left: start.offset(left - 1),
                right_idx: len - right,
                right: start.offset(len - right),
                end: start.offset(len - 1),
                _marker: std::marker::PhantomData,
            }
        }
    }
}

impl<'a, T> Iterator for IndexedEdgeIteratorMut<'a, T> {
    type Item = (isize, &'a mut T);

    fn next(&mut self) -> Option<(isize, &'a mut T)> {
        unsafe {
            // Jump from end of left to right.
            if self.pos > self.left && self.pos < self.right {
                self.pos = self.right;
                self.idx = self.right_idx;
            }

            if self.pos > self.end {
                return None;
            } else {
                let value = &mut *self.pos;
                let idx = self.idx;
                self.pos = self.pos.offset(1);
                self.idx += 1;
                Some((idx, value))
            }
        }
    }
}

/// Creates an iterator for the first `left` and last `right` elements in a slice
fn create_edge_iter_mut<T>(
    slice: &mut [T],
    right: usize,
    left: usize,
    len: usize,
) -> impl Iterator<Item = &mut T> {
    let (left_values, right_values) = slice.split_at_mut(left);
    let right = right - left;
    let right_len = len - left;

    left_values
        .iter_mut()
        .chain(right_values[right..right_len].iter_mut())
}

impl<T> SimdPartition<T> {
    pub fn new_all_scalar(len: usize) -> Self {
        Self {
            left: len,
            right: 0,
            len,
            data_type: PhantomData,
        }
    }

    pub fn new_simd(left: usize, right: usize, len: usize) -> Self {
        Self {
            left,
            right,
            len,
            data_type: PhantomData,
        }
    }

    /// Iterator over the left and right side of the slice.
    pub fn edge_iter<'a>(&self, slice: &'a [T]) -> impl Iterator<Item = &'a T> {
        slice[0..self.left]
            .iter()
            .chain(slice[self.len - self.right..self.len].iter())
    }

    /// Iterator over the left and right side of the slice. Expects complex data.
    pub fn cedge_iter<'a>(&self, slice: &'a [Complex<T>]) -> impl Iterator<Item = &'a Complex<T>> {
        slice[0..self.left / 2]
            .iter()
            .chain(slice[self.len / 2 - self.right / 2..self.len / 2].iter())
    }

    /// Iterator over the left and right side of the slice. Expects the real part of complex data.
    pub fn edge_iter_mut<'a>(&self, slice: &'a mut [T]) -> impl Iterator<Item = &'a mut T> {
        EdgeIteratorMut::new(&mut slice[0..self.len], self.left, self.right)
    }

    /// Iterator over the left and right side of the slice. Expects complex data.
    pub fn cedge_iter_mut<'a>(
        &self,
        slice: &'a mut [Complex<T>],
    ) -> impl Iterator<Item = &'a mut Complex<T>> {
        EdgeIteratorMut::new(&mut slice[0..self.len / 2], self.left / 2, self.right / 2)
    }

    /// Iterator over the left and right side of the slice. Expects the real part of complex data.
    pub fn redge_iter_mut<'a>(&self, slice: &'a mut [T]) -> impl Iterator<Item = &'a mut T> {
        EdgeIteratorMut::new(&mut slice[0..self.len / 2], self.left / 2, self.right / 2)
    }

    /// Gets the center of a slice.
    pub fn center<'a>(&self, slice: &'a [T]) -> &'a [T] {
        if self.left == self.len {
            &[]
        } else {
            &slice[self.left..self.len - self.right]
        }
    }

    // Gets the center of a slice.
    pub fn center_mut<'a>(&self, slice: &'a mut [T]) -> &'a mut [T] {
        if self.left == self.len {
            &mut []
        } else {
            &mut slice[self.left..self.len - self.right]
        }
    }

    // Gets the center of a slice, expects the real part of complex data.
    pub fn rcenter_mut<'a>(&self, slice: &'a mut [T]) -> &'a mut [T] {
        if self.left == self.len {
            &mut []
        } else {
            &mut slice[self.left / 2..self.len / 2 - self.right / 2]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EdgeIteratorMut, IndexedEdgeIteratorMut};

    #[test]
    fn edge_iter_test() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        for n in EdgeIteratorMut::new(&mut data, 2, 3) {
            *n = -*n;
        }

        let expected = vec![-1, -2, 3, 4, 5, 6, -7, -8, -9];
        assert_eq!(&data, &expected);
    }

    #[test]
    fn indexed_edge_iter_test() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        for (idx, n) in IndexedEdgeIteratorMut::new(&mut data, 2, 3) {
            *n = -*n - 10 * (idx + 1);
        }

        let expected = vec![-11, -22, 3, 4, 5, 6, -77, -88, -99];
        assert_eq!(&data, &expected);
    }
}
