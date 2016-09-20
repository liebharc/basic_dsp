use std::mem;
use std::ptr;
use RealNumber;
use multicore_support::*;
use super::super::{
	array_to_complex_mut,
	VoidResult, Buffer, Owner, ErrorReason,
	NumberSpace, Domain,
	DspVec, Vector, ToSliceMut, Resize
};

pub trait ReorganizeDataOps<S, T>
 	where T: RealNumber,
	      S: ToSliceMut<T> {
	/// Reverses the data inside the vector.
	///
	/// # Example
	///
	/// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).to_real_time_vec();
    /// vector.reverse();
    /// assert_eq!([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], vector[..]);
    /// ```
	fn reverse(&mut self);

	/// This function swaps both halves of the vector. This operation is also called FFT shift
	/// Use it after a `plain_fft` to get a spectrum which is centered at `0 Hz`.
	///
	/// # Example
	///
	/// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).to_real_time_vec();
	/// let mut buffer = SingleBuffer::new();
    /// vector.swap_halves(&mut buffer);
    /// assert_eq!([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], vector[..]);
    /// ```
	fn swap_halves<B>(&mut self, buffer: &mut B)
		where B: Buffer<S, T>;
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

pub trait InsertZerosOps<S, T>
 	where T: RealNumber,
	      S: ToSliceMut<T> {
	/// Appends zeros add the end of the vector until the vector has the size given in the points argument.
	/// If `points` smaller than the `self.len()` then this operation won't do anything.
	///
	/// Note: Each point is two floating point numbers if the vector is complex.
	/// Note2: Adding zeros to the signal changes its power. If this function is used to zero pad to a power
	/// of 2 in order to speed up FFT calculation then it might be necessary to multiply it with `len_after/len_before`\
	/// so that the spectrum shows the expected power. Of course this is depending on the application.
	/// # Example
	///
	/// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let mut vector = vec!(1.0, 2.0).to_real_time_vec();
	/// let mut buffer = SingleBuffer::new();
    /// vector.zero_pad(&mut buffer, 4, PaddingOption::End);
    /// assert_eq!([1.0, 2.0, 0.0, 0.0], vector[..]);
	/// let mut vector = vec!(1.0, 2.0).to_complex_time_vec();
	/// let mut buffer = SingleBuffer::new();
    /// vector.zero_pad(&mut buffer, 2, PaddingOption::End);
    /// assert_eq!([1.0, 2.0, 0.0, 0.0], vector[..]);
    /// ```
	fn zero_pad<B>(&mut self, buffer: &mut B, points: usize, option: PaddingOption)
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
    /// use basic_dsp_vector::vector_types2::*;
    /// let mut vector = vec!(1.0, 2.0).to_real_time_vec();
	/// let mut buffer = SingleBuffer::new();
    /// vector.zero_interleave(&mut buffer, 2);
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], vector[..]);
	/// let mut vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
	/// let mut buffer = SingleBuffer::new();
    /// vector.zero_interleave(&mut buffer, 2);
    /// assert_eq!([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], vector[..]);
    /// ```
	fn zero_interleave<B>(&mut self, buffer: &mut B, factor: u32)
		where B: Buffer<S, T>;
}

pub trait SplitOps {
	/// Splits the vector into several smaller vectors. `self.len()` must be dividable by
    /// `targets.len()` without a remainder and this condition must be true too `targets.len() > 0`.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `InvalidArgumentLength`: `self.points()` isn't dividable by `targets.len()`
    ///
    /// # Example
    ///
    /// ```
	/// use basic_dsp_vector::vector_types2::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).to_real_time_vec();
    /// let mut split = &mut
    ///     [Box::new(Vec::new().to_real_time_vec()),
    ///     Box::new(Vec::new().to_real_time_vec())];
    /// vector.split_into(split).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 3.0, 5.0, 7.0, 9.0], split[0][..]);
    /// ```
    fn split_into(&self, targets: &mut [Box<Self>]) -> VoidResult;
}

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
	/// use basic_dsp_vector::vector_types2::*;
    /// let mut parts = &mut
    ///     [Box::new(vec!(1.0, 2.0).to_real_time_vec()),
    ///     Box::new(vec!(1.0, 2.0).to_real_time_vec())];
	/// let mut vector = Vec::new().to_real_time_vec();
    /// vector.merge(parts).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 1.0, 2.0, 2.0], vector[..]);
    /// ```
    fn merge(&mut self, sources: &[Box<Self>]) -> VoidResult;
}

impl<S, T, N, D> ReorganizeDataOps<S, T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T> + Owner,
          T: RealNumber,
          N: NumberSpace,
          D: Domain {
	fn reverse(&mut self) {
		if self.is_complex() {
			let len = self.points();
			let mut data = self.data.to_slice_mut();
			let mut data = array_to_complex_mut(&mut data[..]);
			for i in 0..len / 2 {
				let temp = data[i];
				data[i] = data[len - 1 - i];
				data[len - 1 - i] = temp;
			}
		} else {
			let len = self.len();
			let mut data = self.data.to_slice_mut();
			for i in 0..len / 2 {
				let temp = data[i];
				data[i] = data[len - 1 - i];
				data[len - 1 - i] = temp;
			}
		}

	}

	fn swap_halves<B>(&mut self, buffer: &mut B)
		where B: Buffer<S, T> {
		self.swap_halves_priv(buffer, true);
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

impl<S, T, N, D> InsertZerosOps<S, T> for DspVec<S, T, N, D>
	where S: ToSliceMut<T> + Owner,
	  T: RealNumber,
	  N: NumberSpace,
	  D: Domain {
	fn zero_pad<B>(&mut self, buffer: &mut B, points: usize, option: PaddingOption)
		where B: Buffer<S, T> {
		let len_before = self.len();
		let is_complex = self.is_complex();
		let len = if is_complex { 2 * points } else { points };
		if len < len_before {
			return;
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
						ptr::write_bytes(dest_end, 0, target.len() - len_before);
					}
				}
				PaddingOption::Surround => {
					let diff = (len - len_before) / if is_complex { 2 } else { 1 };
					let mut right = diff / 2;
					let mut left = diff - right;
					if is_complex {
						right *= 2;
						left *= 2;
					}

					unsafe {
						let src = &data[0] as *const T;
						let dest = &mut target[left] as *mut T;
						ptr::copy(src, dest, len_before);
						let dest = &mut target[len - right] as *mut T;
						ptr::write_bytes(dest, 0, right);
						let dest = &mut target[0] as *mut T;
						ptr::write_bytes(dest, 0, left);
					}
				}
				PaddingOption::Center => {
					let mut diff = (len - len_before) / if is_complex { 2 } else { 1 };
					let mut right = diff / 2;
					let mut left = diff - right;
					if is_complex {
						right *= 2;
						left *= 2;
						diff *= 2;
					}

					unsafe {
						let src = &data[left] as *const T;
						let dest = &mut target[len-right] as *mut T;
						ptr::copy(src, dest, right);
						let dest = &mut target[left] as *mut T;
						ptr::write_bytes(dest, 0, len - diff);
					}
				}
			}
		}

		mem::swap(&mut self.data, &mut target);
		buffer.free(target);
	}


	fn zero_interleave<B>(&mut self, buffer: &mut B, factor: u32)
		where B: Buffer<S, T> {
		if self.is_complex() {
			zero_interleave!(self, buffer, factor, 2)
		} else {
			zero_interleave!(self, buffer, factor, 1)
		}
	}
}

impl<S, T, N, D> SplitOps for DspVec<S, T, N, D>
	where S: ToSliceMut<T> + Owner + Resize ,
	  T: RealNumber,
	  N: NumberSpace,
	  D: Domain {
    fn split_into(&self, targets: &mut [Box<Self>]) -> VoidResult {
		let num_targets = targets.len();
		let data_length = self.len();
		if num_targets == 0 || data_length % num_targets != 0 {
			return Err(ErrorReason::InvalidArgumentLength);
		}

		for i in 0..num_targets {
			targets[i].data.resize(data_length / num_targets);
			targets[i].shrink(data_length / num_targets).expect("We just resized to this size so shrink should never fail");
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
			for i in 0..data_length {
				let target = targets[i % num_targets].data.to_slice_mut();
				let pos = i / num_targets;
				target[pos] = data[i];
			}
		}

		Ok(())
	}
}

impl<S, T, N, D> MergeOps for DspVec<S, T, N, D>
	where S: ToSliceMut<T> + Owner + Resize,
	  T: RealNumber,
	  N: NumberSpace,
	  D: Domain {
    fn merge(&mut self, sources: &[Box<Self>]) -> VoidResult {
		{
			let num_sources = sources.len();
			if num_sources == 0 {
				return Err(ErrorReason::InvalidArgumentLength);
			}

			for i in 1..num_sources {
				if sources[0].len() != sources[i].len() {
					return Err(ErrorReason::InvalidArgumentLength);
				}
			}

			self.data.resize(sources[0].len() * num_sources);
			self.shrink(sources[0].len() * num_sources).expect("We just resized to this size so shrink should never fail");

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
			   for i in 0..data_length {
					let source = sources[i % num_sources].data.to_slice();
					let pos = i / num_sources;
					data[i] = source[pos];
				}
			}
		}

		Ok(())
	}
}
