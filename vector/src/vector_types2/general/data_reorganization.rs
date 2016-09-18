use RealNumber;
use super::super::{
	array_to_complex_mut,
	VoidResult, Buffer, Owner,
	NumberSpace, Domain,
	DspVec, Vector, ToSliceMut,
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

pub trait InsertZerosOps {
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
	/// use basic_dsp_vector::{PaddingOption, RealTimeVector32, ComplexTimeVector32, GenericVectorOps, DataVec, RealIndex, InterleavedIndex};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.zero_pad(4, PaddingOption::End).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 0.0, 0.0], result.real(0..));
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0]);
	/// let result = vector.zero_pad(2, PaddingOption::End).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 0.0, 0.0], result.interleaved(0..));
	/// ```
	fn zero_pad(&mut self, points: usize, option: PaddingOption);

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
	/// use basic_dsp_vector::{RealTimeVector32, ComplexTimeVector32, GenericVectorOps, DataVec, RealIndex, InterleavedIndex};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.zero_interleave(2).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 0.0, 2.0, 0.0], result.real(0..));
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
	/// let result = vector.zero_interleave(2).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], result.interleaved(0..));
	/// ```
	fn zero_interleave(&mut self, factor: u32);
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
    /// use basic_dsp_vector::{RealTimeVector32, GenericVectorOps, DataVec, RealIndex};
    /// let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    /// let merge = RealTimeVector32::from_array(&a);
    /// let mut split = &mut
    ///     [Box::new(RealTimeVector32::empty()),
    ///     Box::new(RealTimeVector32::empty())];
    /// merge.split_into(split).unwrap();
    /// assert_eq!([1.0, 3.0, 5.0, 7.0, 9.0], split[0].real(0..));
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
    /// use basic_dsp_vector::{RealTimeVector32, GenericVectorOps, DataVec, RealIndex};
    /// let vector = RealTimeVector32::empty();
    /// let parts = &[
    ///     Box::new(RealTimeVector32::from_array(&[1.0, 2.0])),
    ///     Box::new(RealTimeVector32::from_array(&[1.0, 2.0]))];
    /// let merged = vector.merge(parts).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 1.0, 2.0, 2.0], merged.real(0..));
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
