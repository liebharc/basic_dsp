use super::super::{Domain, DspVec, MetaData, NumberSpace, ToSliceMut, Vector};
use crate::array_to_complex_mut;
use crate::numbers::*;

/// A trait to calculate the diff (1st derivative in a discrete number space) or cumulative sum
/// (integral  in a discrete number space).
pub trait DiffSumOps {
    /// Calculates the delta of each elements to its previous element. This will decrease
    /// the vector length by one point.
    ///
    /// # Example
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(2.0, 3.0, 2.0, 6.0).to_real_time_vec();
    /// vector.diff();
    /// assert_eq!([1.0, -1.0, 4.0], vector[..]);
    /// let mut vector = vec!(2.0, 2.0, 3.0, 3.0, 5.0, 5.0).to_complex_time_vec();
    /// vector.diff();
    /// assert_eq!([1.0, 1.0, 2.0, 2.0], vector[..]);
    /// ```
    fn diff(&mut self);

    /// Calculates the delta of each elements to its previous element. The first element
    /// will remain unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(2.0, 3.0, 2.0, 6.0).to_real_time_vec();
    /// vector.diff_with_start();
    /// assert_eq!([2.0, 1.0, -1.0, 4.0], vector[..]);
    /// let mut vector = vec!(2.0, 2.0, 3.0, 3.0, 5.0, 5.0).to_complex_time_vec();
    /// vector.diff_with_start();
    /// assert_eq!([2.0, 2.0, 1.0, 1.0, 2.0, 2.0], vector[..]);
    /// ```
    fn diff_with_start(&mut self);

    /// Calculates the cumulative sum of all elements. This operation undoes the
    /// `diff_with_start`operation.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(2.0, 1.0, -1.0, 4.0).to_real_time_vec();
    /// vector.cum_sum();
    /// assert_eq!([2.0, 3.0, 2.0, 6.0], vector[..]);
    /// ```
    fn cum_sum(&mut self);
}

impl<S, T, N, D> DiffSumOps for DspVec<S, T, N, D>
where
    S: ToSliceMut<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn diff(&mut self) {
        let is_complex = self.is_complex();
        let step = if is_complex { 2 } else { 1 };
        let len = self.len();
        self.valid_len -= step;
        let data = self.data.to_slice_mut();

        if is_complex {
            let data = array_to_complex_mut(data);
            for j in 0..len / 2 - 1 {
                data[j] = data[j + 1] - data[j];
            }
        } else {
            for j in 0..len - 1 {
                data[j] = data[j + 1] - data[j];
            }
        }
    }

    fn diff_with_start(&mut self) {
        let is_complex = self.is_complex();
        let len = self.len();
        if len == 0 {
            return;
        }

        let data = self.data.to_slice_mut();

        if is_complex {
            let data = array_to_complex_mut(data);
            let mut temp = data[0];
            for n in &mut data[1..len / 2] {
                let diff = *n - temp;
                temp = *n;
                *n = diff;
            }
        } else {
            let mut temp = data[0];
            for n in &mut data[1..len] {
                let diff = *n - temp;
                temp = *n;
                *n = diff;
            }
        }
    }

    fn cum_sum(&mut self) {
        let data_length = self.len();
        let mut i = 0;
        let mut j = if self.is_complex() { 2 } else { 1 };

        let data = self.data.to_slice_mut();
        while j < data_length {
            data[j] = data[j] + data[i];
            i += 1;
            j += 1;
        }
    }
}
