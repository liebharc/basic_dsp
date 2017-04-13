use std::mem;
use std::ptr;
use {array_to_complex_mut};
use numbers::*;
use multicore_support::*;
use super::super::{VoidResult, Buffer, Owner, ErrorReason, NumberSpace,
                   Domain, ResizeOps, DspVec, Vector, ToSliceMut, MetaData};

/// This trait allows to reorganize the data by changing positions of the individual elements.
pub trait ReorganizeDataOps<T>
    where T: RealNumber
{
    /// Reverses the data inside the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).to_real_time_vec();
    /// vector.reverse();
    /// assert_eq!([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], vector[..]);
    /// ```
    fn reverse(&mut self);

    /// This function swaps both halves of the vector. This operation is also called FFT shift
    /// Use it after a `plain_fft` to get a spectrum which is centered at `0 Hz`.
    ///
    /// `swap_halvesb` requires a buffer but performs faster.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).to_real_time_vec();
    /// vector.swap_halves();
    /// assert_eq!([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], vector[..]);
    /// ```
    fn swap_halves(&mut self);
}

/// An option which defines how a vector should be padded
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum PaddingOption {
    /// Appends zeros to the end of the vector.
    End,
    /// Surrounds the vector with zeros at the beginning and at the end.
    Surround,

    /// Inserts zeros in the center of the vector
    Center,
}

/// A trait to insert zeros into the data at some specified positions.
pub trait InsertZerosOps<T>
    where T: RealNumber
{
    /// Appends zeros add the end of the vector until the vector has the size given
    /// in the points argument.
    /// If `points` smaller than the `self.len()` then this operation won't do anything, however
    /// in future it will raise an error.
    ///
    /// Note: Each point is two floating point numbers if the vector is complex.
    /// Note2: Adding zeros to the signal changes its power. If this function is used to
    /// zero pad to a power
    /// of 2 in order to speed up FFT calculation then it might be necessary to multiply it
    /// with `len_after/len_before`\
    /// so that the spectrum shows the expected power. Of course this is depending
    /// on the application.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0).to_real_time_vec();
    /// vector.zero_pad(4, PaddingOption::End).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 0.0, 0.0], vector[..]);
    /// let mut vector = vec!(1.0, 2.0).to_complex_time_vec();
    /// vector.zero_pad(2, PaddingOption::End).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 0.0, 0.0], vector[..]);
    /// ```
    fn zero_pad(&mut self, points: usize, option: PaddingOption) -> VoidResult;

    /// Interleaves zeros `factor - 1`times after every vector element, so that the resulting
    /// vector will have a length of `self.len() * factor`.
    ///
    /// Note: Remember that each complex number consists of two floating points and interleaving
    /// will take that into account.
    ///
    /// If factor is 0 (zero) then `self` will be returned.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0).to_real_time_vec();
    /// vector.zero_interleave(2);
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], vector[..]);
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// vector.zero_interleave(2).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], vector[..]);
    /// ```
    fn zero_interleave(&mut self, factor: u32) -> VoidResult;
}

/// A trait to insert zeros into the data at some specified positions. A buffer is used
/// for types which can't be resized and/or to speed up the calculation.
pub trait InsertZerosOpsBuffered<S, T>
    where T: RealNumber,
          S: ToSliceMut<T>
{
    /// Appends zeros add the end of the vector until the vector has the size given in the
    /// points argument.
    /// If `points` smaller than the `self.len()` then this operation will return an error.
    ///
    /// Note: Each point is two floating point numbers if the vector is complex.
    /// Note2: Adding zeros to the signal changes its power. If this function is used to
    /// zero pad to a power
    /// of 2 in order to speed up FFT calculation then it might be necessary to multiply it
    /// with `len_after/len_before`\
    /// so that the spectrum shows the expected power. Of course this is depending on the
    /// application.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0).to_real_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// vector.zero_pad_b(&mut buffer, 4, PaddingOption::End).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 0.0, 0.0], vector[..]);
    /// let mut vector = vec!(1.0, 2.0).to_complex_time_vec();
    /// vector.zero_pad_b(&mut buffer, 2, PaddingOption::End).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 0.0, 0.0], vector[..]);
    /// ```
    fn zero_pad_b<B>(&mut self, buffer: &mut B, points: usize, option: PaddingOption) -> VoidResult
        where B: Buffer<S, T>;

    /// Interleaves zeros `factor - 1`times after every vector element, so that the resulting
    /// vector will have a length of `self.len() * factor`.
    ///
    /// Note: Remember that each complex number consists of two floating points and interleaving
    /// will take that into account.
    ///
    /// If factor is 0 (zero) then `self` will be returned.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0).to_real_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// vector.zero_interleave_b(&mut buffer, 2);
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], vector[..]);
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// vector.zero_interleave_b(&mut buffer, 2);
    /// assert_eq!([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], vector[..]);
    /// ```
    fn zero_interleave_b<B>(&mut self, buffer: &mut B, factor: u32) where B: Buffer<S, T>;
}

/// Splits the data into several smaller pieces of equal size.
pub trait SplitOps {
    /// Splits the vector into several smaller vectors. `self.len()` must be dividable by
    /// `targets.len()` without a remainder and this condition must be true too
    /// `targets.len() > 0`.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `InvalidArgumentLength`: `self.points()` isn't dividable by `targets.len()`
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector =
    ///     vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).to_real_time_vec();
    /// let mut split = &mut
    ///     [&mut Vec::new().to_real_time_vec(),
    ///      &mut Vec::new().to_real_time_vec()];
    /// vector.split_into(split).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 3.0, 5.0, 7.0, 9.0], split[0][..]);
    /// ```
    fn split_into(&self, targets: &mut [&mut Self]) -> VoidResult;
}

/// Merges several pieces of equal size into one data chunk.
pub trait MergeOps {
    /// Merges several vectors into `self`. All vectors must have the same size and
    /// at least one vector must be provided.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `InvalidArgumentLength`: if `sources.len() == 0`
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut parts = &mut
    ///     [&vec!(1.0, 2.0).to_real_time_vec(),
    ///      &vec!(1.0, 2.0).to_real_time_vec()];
    /// let mut vector = Vec::new().to_real_time_vec();
    /// vector.merge(parts).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 1.0, 2.0, 2.0], vector[..]);
    /// ```
    fn merge(&mut self, sources: &[&Self]) -> VoidResult;
}

fn reverse_array<T>(data: &mut [T])
    where T: Copy {
    let len = data.len();
    // +1 makes sure that for odd numbers the first `lo` gets the extra item
    // (which is later on ignored)
    let (lo, up) = data.split_at_mut((len + 1) / 2);
    for (lo, up) in lo.iter_mut().zip(up.iter_mut().rev()) {
        mem::swap(lo, up);
    }
}

impl<S, T, N, D> ReorganizeDataOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn reverse(&mut self) {
        let len = self.len();
        if self.is_complex() {
            let mut data = self.data.to_slice_mut();
            let mut data = array_to_complex_mut(&mut data[0..len]);
            reverse_array(data);
        } else {
            let mut data = self.data.to_slice_mut();
            reverse_array(&mut data[0..len]);
        }
    }

    fn swap_halves(&mut self) {
        self.swap_halves_priv(true);
    }
}

macro_rules! zero_interleave {
    ($self_: ident, $buffer: ident, $step: ident, $tuple: expr) => {
        {
            if $step <= 1 {
                return;
            }

            let step = $step as usize;
            let old_len = $self_.len();
            let new_len = step * old_len;
            $self_.valid_len = new_len;
            let mut target = $buffer.get(new_len);
			{
				let mut target = target.to_slice_mut();
				let source = &$self_.data.to_slice();
                Chunk::from_src_to_dest(
                    Complexity::Small, &$self_.multicore_settings,
                    &source[0..old_len], $tuple,
                    &mut target[0..new_len], $tuple * step, (),
                    move|original, range, target, _arg| {
                         // Zero target
                        let ptr = &mut target[0] as *mut T;
                        unsafe {
                            ptr::write_bytes(ptr, 0, new_len);
                        }
                        let skip = step * $tuple;
                        let mut i = 0;
                        let mut j = range.start;
                        while i < target.len() {
                            let original_ptr = unsafe { original.get_unchecked(j) };
                            let target_ptr = unsafe { target.get_unchecked_mut(i) };
                            unsafe {
                                ptr::copy(original_ptr, target_ptr, $tuple);
                            }

                            j += $tuple;
                            i += skip;
                        }
            	});
			}

			mem::swap(&mut $self_.data, &mut target);
			$buffer.free(target);
        }
    }
}

impl<S, T, N, D> InsertZerosOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn zero_pad(&mut self, points: usize, option: PaddingOption) -> VoidResult {
        let len_before = self.len();
        let is_complex = self.is_complex();
        let step = if is_complex { 2 } else { 1 };
        let len = points * step;
        if len <= len_before {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        try!(self.resize(len));
        let data = self.data.to_slice_mut();
        match option {
            PaddingOption::End => {
                // Zero target
                let dest = &mut data[len_before] as *mut T;
                unsafe {
                    ptr::write_bytes(dest, 0, len - len_before);
                }
                Ok(())
            }
            PaddingOption::Surround => {
                let diff = (len - len_before) / step;
                let mut right = (diff - 1) / 2;
                let mut left = diff - right ;
                if is_complex {
                    right *= 2;
                    left *= 2;
                }

                unsafe {
                    let src = &data[0] as *const T;
                    let dest = &mut data[left] as *mut T;
                    ptr::copy(src, dest, len_before);
                    if right > 0 {
                        let dest = &mut data[len - right] as *mut T;
                        ptr::write_bytes(dest, 0, right);
                    }
                    let dest = &mut data[0] as *mut T;
                    ptr::write_bytes(dest, 0, left);
                }
                Ok(())
            }
            PaddingOption::Center => {
                let points_before = len_before / step;
                let mut right = (points_before - 1) / 2;
                let mut left = points_before - right ;
                if is_complex {
                    right *= 2;
                    left *= 2;
                }

                if right > 0 {
                    unsafe {
                        let src = &data[left] as *const T;
                        let dest = &mut data[len - right] as *mut T;
                        // src and dest may overload with copy, a unit test covers this case
                        ptr::copy(src, dest, right);
                    }
                }

                unsafe {
                    let dest = &mut data[left] as *mut T;
                    ptr::write_bytes(dest, 0, len - left - right);

                }
                Ok(())
            }
        }
    }

    fn zero_interleave(&mut self, factor: u32) -> VoidResult {
        let len_before = self.len();
        let is_complex = self.is_complex();
        let factor = factor as usize;
        let len = len_before * factor;
        if len < len_before {
            return Ok(());
        }

        try!(self.resize(len));

        if is_complex {
            let data = self.data.to_slice_mut();
            let data = array_to_complex_mut(data);
            for j in 0..len / 2 {
                let i = len / 2 - 1 - j;
                if i % factor == 0 {
                    data[i] = data[i / factor];
                } else {
                    data[i] = Complex::<T>::new(T::zero(), T::zero());
                }
            }
        } else {
            let data = self.data.to_slice_mut();
            for j in 0..len {
                let i = len - 1 - j;
                if i % factor == 0 {
                    data[i] = data[i / factor];
                } else {
                    data[i] = T::zero();
                }
            }
        }

        Ok(())
    }
}


impl<S, T, N, D> InsertZerosOpsBuffered<S, T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn zero_pad_b<B>(&mut self, buffer: &mut B, points: usize, option: PaddingOption) -> VoidResult
        where B: Buffer<S, T>
    {
        let len_before = self.len();
        let is_complex = self.is_complex();
        let len = if is_complex { 2 * points } else { points };
        if len <= len_before {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        let mut target = buffer.get(len);
        {
            let data = self.data.to_slice();
            let target = target.to_slice_mut();
            self.valid_len = len;
            match option {
                PaddingOption::End => {
                    // Zero target
                    let src = &data[0] as *const T;
                    let dest_start = &mut target[0] as *mut T;
                    let dest_end = &mut target[len_before] as *mut T;
                    unsafe {
                        ptr::copy(src, dest_start, len_before);
                        ptr::write_bytes(dest_end, 0, len - len_before);
                    }
                }
                PaddingOption::Surround => {
                    let diff = (len - len_before) / if is_complex { 2 } else { 1 };
                    let mut right = (diff) / 2;
                    let mut left = diff - right;
                    if is_complex {
                        right *= 2;
                        left *= 2;
                    }

                    unsafe {
                        let src = &data[0] as *const T;
                        let dest = &mut target[left] as *mut T;
                        ptr::copy(src, dest, len_before);
                        if right > 0 {
                            let dest = &mut target[len - right] as *mut T;
                            ptr::write_bytes(dest, 0, right);
                        }
                        let dest = &mut target[0] as *mut T;
                        ptr::write_bytes(dest, 0, left);
                    }
                }
                PaddingOption::Center => {
                    let step = if is_complex { 2 } else { 1 };
                    let points_before = len_before / step;
                    let mut right = points_before / 2;
                    let mut left = points_before - right;
                    if is_complex {
                        right *= 2;
                        left *= 2;
                    }

                    unsafe {
                        let src = &data[len_before - right] as *const T;
                        let dest = &mut target[len - right] as *mut T;
                        ptr::copy(src, dest, right);
                        let src = &data[0] as *const T;
                        let dest = &mut target[0] as *mut T;
                        ptr::copy(src, dest, left);
                        let dest = &mut target[left] as *mut T;
                        ptr::write_bytes(dest, 0, len - len_before);
                    }
                }
            }
        }

        mem::swap(&mut self.data, &mut target);
        buffer.free(target);
        Ok(())
    }

    fn zero_interleave_b<B>(&mut self, buffer: &mut B, factor: u32)
        where B: Buffer<S, T>
    {
        if self.is_complex() {
            zero_interleave!(self, buffer, factor, 2)
        } else {
            zero_interleave!(self, buffer, factor, 1)
        }
    }
}

impl<S, T, N, D> SplitOps for DspVec<S, T, N, D>
    where S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn split_into(&self, targets: &mut [&mut Self]) -> VoidResult {
        let num_targets = targets.len();
        let data_length = self.len();
        if num_targets == 0 || data_length % num_targets != 0 {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        for t in targets.iter_mut() {
            try!(t.resize(data_length / num_targets));
        }

        let data = &self.data.to_slice();
        if self.is_complex() {
            for i in 0..(data_length / 2) {
                let target = targets[i % num_targets].data.to_slice_mut();
                let pos = i / num_targets;
                target[2 * pos] = data[2 * i];
                target[2 * pos + 1] = data[2 * i + 1];
            }
        } else {
            for (i, d) in data.iter().enumerate() {
                let target = targets[i % num_targets].data.to_slice_mut();
                let pos = i / num_targets;
                target[pos] = *d;
            }
        }

        Ok(())
    }
}

impl<S, T, N, D> MergeOps for DspVec<S, T, N, D>
    where S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn merge(&mut self, sources: &[&Self]) -> VoidResult {
        {
            let num_sources = sources.len();
            if num_sources == 0 {
                return Err(ErrorReason::InvalidArgumentLength);
            }

            for src in sources.iter().take(num_sources).skip(1) {
                if sources[0].len() != src.len() {
                    return Err(ErrorReason::InvalidArgumentLength);
                }
            }

            try!(self.resize(sources[0].len() * num_sources));

            let data_length = self.len();
            let is_complex = self.is_complex();
            let data = self.data.to_slice_mut();
            if is_complex {
                for i in 0..(data_length / 2) {
                    let source = sources[i % num_sources].data.to_slice();
                    let pos = i / num_sources;
                    data[2 * i] = source[2 * pos];
                    data[2 * i + 1] = source[2 * pos + 1];
                }
            } else {
                for (i, d) in data.iter_mut().enumerate() {
                    let source = sources[i % num_sources].data.to_slice();
                    let pos = i / num_sources;
                    *d = source[pos];
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::*;

    #[test]
    fn swap_halves_real_even_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0].to_real_time_vec();
        v.swap_halves();
        assert_eq!(&v[..], &[3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn swap_halves_real_odd_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
            .to_real_time_vec();
        v.swap_halves();
        assert_eq!(&v[..],
                   &[7.0, 8.0, 9.0, 10.0, 11.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn swap_halves_complex_even_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_complex_time_vec();
        v.swap_halves();
        assert_eq!(&v[..], &[5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn swap_halves_complex_odd_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        v.swap_halves();
        assert_eq!(&v[..], &[7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn zero_pad_end_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        v.zero_pad(9, PaddingOption::End).unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_surround_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        v.zero_pad(10, PaddingOption::Surround).unwrap();
        let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                        10.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_center_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        v.zero_pad(10, PaddingOption::Center).unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(&v[..], &expected);
    }


    #[test]
    fn zero_pad_b_center_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        v.zero_pad_b(&mut buffer, 10, PaddingOption::Center).unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_surround_odd_signal_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].to_real_time_vec();
        v.zero_pad(20, PaddingOption::Surround).unwrap();
        // The expected result is required so that the convolution theorem holds true
        // (mul in freq is the same as conv)
        let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                        10.0, 11.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_b_end_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        v.zero_pad_b(&mut buffer, 9, PaddingOption::End).unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_b_surround_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        v.zero_pad_b(&mut buffer, 10, PaddingOption::Surround).unwrap();
        let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                        10.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_b_surround_odd_signal_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0].to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        v.zero_pad_b(&mut buffer, 10, PaddingOption::Surround).unwrap();
        let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                        10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_on_slice_fail_test() {
        let a: Box<[f64]> = Box::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let mut v = a.to_complex_time_vec();
        assert_eq!(v.zero_pad(9, PaddingOption::End),
                   Err(ErrorReason::TypeCanNotResize));
    }

    #[test]
    fn zero_pad_on_slice_shrinked_test() {
        let a: Box<[f64]> = Box::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                                      12.0, 13.0, 14.0, 11.0, 12.0, 13.0, 14.0]);
        let mut v = a.to_complex_time_vec();
        v.resize(10).unwrap();
        v.zero_pad(9, PaddingOption::End).unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_surround_overlap_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        v.zero_pad(8, PaddingOption::Surround).unwrap();
        let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0];
        assert_eq!(&v[..], &expected);
    }

    #[test]
    fn zero_pad_center_overlap_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].to_complex_time_vec();
        v.zero_pad(8, PaddingOption::Center).unwrap();
        let expected = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(&v[..], &expected);
    }


    #[test]
    fn zero_interleave_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0].to_real_time_vec();
        v.zero_interleave(2).unwrap();
        assert_eq!(&v[..], &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn zero_interleave_even_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0].to_real_time_vec();
        v.zero_interleave(2).unwrap();
        assert_eq!(&v[..], &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn zero_interleave_b_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        v.zero_interleave_b(&mut buffer, 2);
        assert_eq!(&v[..], &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn zero_interleave_complex_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0].to_complex_time_vec();
        v.zero_interleave(2).unwrap();
        assert_eq!(&v[..], &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn zero_interleave_b_complex_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0].to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        v.zero_interleave_b(&mut buffer, 2);
        assert_eq!(&v[..], &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }
}
