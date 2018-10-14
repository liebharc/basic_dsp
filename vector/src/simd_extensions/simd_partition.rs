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
    data_type: PhantomData<T>
}

fn create_edge_iter_mut<'a, T>(slice: &'a mut [T], right: usize, left: usize, len: usize) -> impl Iterator<Item = &'a mut T> {
    let (left_values, right_values) = slice.split_at_mut(left);
    let right = right - left;
    let right_len = len - left;

    left_values.iter_mut().chain(right_values[right..right_len].iter_mut())
}

impl<T> SimdPartition<T> {
    pub fn new_all_scalar(len: usize) -> Self {
        Self {
            left: len,
            right: len,
            len: len,
            data_type: PhantomData
        }
    }

    pub fn new_simd(left: usize, right: usize, len: usize) -> Self {
        Self {
            left: left,
            right: right,
            len: len,
            data_type: PhantomData
        }
    }

    /// Iterator over the left and right side of the slice.
    pub fn edge_iter<'a>(&self, slice: &'a [T]) -> impl Iterator<Item = &'a T> {
        slice[0..self.left].iter().chain(slice[self.right..].iter())
    }

    /// Iterator over the left and right side of the slice. Expects complex data.
    pub fn cedge_iter<'a>(&self, slice: &'a [Complex<T>]) -> impl Iterator<Item = &'a Complex<T>> {
        slice[0..self.left / 2].iter().chain(slice[self.right / 2..].iter())
    }

    /// Iterator over the left and right side of the slice. Expects the real part of complex data.
    pub fn edge_iter_mut<'a>(&self, slice: &'a mut [T]) -> impl Iterator<Item = &'a mut T> {
        create_edge_iter_mut(slice, self.right, self.left, self.len)
    }

    /// Iterator over the left and right side of the slice. Expects complex data.
    pub fn cedge_iter_mut<'a>(&self, slice: &'a mut [Complex<T>]) -> impl Iterator<Item = &'a mut Complex<T>> {
        create_edge_iter_mut(slice, self.right / 2, self.left / 2, self.len / 2)
    }

    /// Iterator over the left and right side of the slice. Expects the real part of complex data.
    pub fn redge_iter_mut<'a>(&self, slice: &'a mut [T]) -> impl Iterator<Item = &'a mut T> {
        create_edge_iter_mut(slice, self.right / 2, self.left / 2, self.len / 2)
    }

    /// Gets the center of a slice.
    pub fn center<'a>(&self, slice: &'a [T]) -> &'a [T] {
        if self.left == self.len { &[] } else { &slice[self.left.. self.right] }
    }

    // Gets the center of a slice.
    pub fn center_mut<'a>(&self, slice: &'a mut [T]) -> &'a mut [T] {
        if self.left == self.len { &mut [] } else { &mut slice[self.left.. self.right] }
    }

    // Gets the center of a slice, expects the real part of complex data.
    pub fn rcenter_mut<'a>(&self, slice: &'a mut [T]) -> &'a mut [T] {

        if self.left == self.len { &mut [] } else { &mut slice[self.left / 2 .. self.right / 2] }
    }
}