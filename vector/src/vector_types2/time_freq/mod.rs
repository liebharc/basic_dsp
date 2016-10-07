mod freq;
pub use self::freq::*;
mod time;
pub use self::time::*;
mod time_to_freq;
pub use self::time_to_freq::*;
mod freq_to_time;
pub use self::freq_to_time::*;
mod correlation;
pub use self::correlation::*;
mod convolution;
pub use self::convolution::*;
mod interpolation;
pub use self::interpolation::*;

use std::mem;
use rustfft::FFT;
use RealNumber;
use num::Zero;
use std::ops::*;
use simd_extensions::*;
use multicore_support::*;
use super::{
	array_to_complex, array_to_complex_mut,
    Buffer, Vector,
    DspVec, ToSliceMut, NumberSpace, Domain
};

fn fft<S, T, N, D, B>(vec: &mut DspVec<S, T, N, D>, buffer: &mut B, reverse: bool)
	where S: ToSliceMut<T>,
    	  T: RealNumber,
		  N: NumberSpace,
		  D: Domain,
		  B: Buffer<S, T> {
	let len = vec.len();
	let mut temp = buffer.get(len);
	{
		let temp = temp.to_slice_mut();
		let points = len / 2; // By two since vector is always complex
		let rbw = (T::from(points).unwrap()) * vec.delta;
		vec.delta = rbw;
		let mut fft = FFT::new(points, reverse);
		let signal = vec.data.to_slice();
		let spectrum = &mut temp[0..len];
		let signal = array_to_complex(&signal[0..len]);
		let spectrum = array_to_complex_mut(spectrum);
		fft.process(signal, spectrum);
	}

	mem::swap(&mut vec.data, &mut temp);
	buffer.free(temp);
}

impl<S, T, N, D> DspVec<S, T, N, D>
	where S: ToSliceMut<T>,
		  T: RealNumber,
		  N: NumberSpace,
		  D: Domain
{
	fn convolve_function_priv<B, TT,C,CMut,F>(
		&mut self,
		buffer: &mut B,
		ratio: T,
		conv_len: usize,
		convert: C,
		convert_mut: CMut,
		fun: F)
			where
				B: Buffer<S, T>,
				C: Fn(&[T]) -> &[TT],
				CMut: Fn(&mut [T]) -> &mut [TT],
				F: Fn(T)->TT,
				TT: Zero + Mul<Output=TT> + Copy
	{
		let len = self.len();
		let mut temp = buffer.get(len);
		{
			let data = self.data.to_slice();
			let temp = temp.to_slice_mut();
			let complex = convert(&data[0..len]);
			let dest = convert_mut(&mut temp[0..len]);
			let len = complex.len();
			let mut i = 0;
			let conv_len =
				if conv_len > len {
					len
				} else {
					conv_len
				};
			let sconv_len = conv_len as isize;
			for num in dest {
				let iter = WrappingIterator::new(complex, i - sconv_len - 1, 2 * conv_len + 1);
				let mut sum = TT::zero();
				let mut j = -(T::from(conv_len).unwrap());
				for c in iter {
					sum = sum + c * fun(-j * ratio);
					j = j + T::one();
				}
				(*num) = sum;
				i += 1;
			}
		}

		mem::swap(&mut temp, &mut self.data);
		buffer.free(temp);
	}

	fn convolve_vector_scalar<B>(&mut self, buffer: &mut B, vector: &Self)
		where B: Buffer<S, T> {
		let points = self.points();
		let other_points = vector.points();
		let (other_start, other_end, full_conv_len, conv_len) =
				if other_points > points {
					let center = other_points / 2;
					let conv_len = points / 2;
					(center - conv_len, center + conv_len, points, conv_len)
				} else {
					(0, other_points, other_points, other_points - other_points / 2)
				};
		if self.is_complex() {
			let len = self.len();
			let mut temp = buffer.get(len);
			{
				let other = vector.data.to_slice();
				let data = self.data.to_slice();
				let temp = temp.to_slice_mut();
				let other = array_to_complex(&other[0..vector.len()]);
				let complex = array_to_complex(&data[0..len]);
				let dest = array_to_complex_mut(&mut temp[0..len]);
				let other_iter = &other[other_start .. other_end];
				let conv_len = conv_len as isize;
				let mut i = 0;
				for num in dest {
					*num = Self::convolve_iteration(complex, other_iter, i, conv_len, full_conv_len);
					i += 1;
				}
			}
			mem::swap(&mut temp, &mut self.data);
			buffer.free(temp);
		} else {
			let len = self.len();
			let mut temp = buffer.get(len);
			{
				let other = vector.data.to_slice();
				let data = self.data.to_slice();
				let temp = temp.to_slice_mut();
				let other = &other[0..vector.len()];
				let data = &data[0..len];
				let dest = &mut temp[0..len];
				let other_iter = &other[other_start .. other_end];
				let conv_len = conv_len as isize;
				let mut i = 0;
				for num in dest {
					*num = Self::convolve_iteration(data, other_iter, i, conv_len, full_conv_len);
					i += 1;
				}
			}
			mem::swap(&mut temp, &mut self.data);
			buffer.free(temp);
		}
	}

	#[inline]
	fn convolve_iteration<TT>(data: &[TT], other_iter: &[TT], i: isize, conv_len: isize, full_conv_len: usize) -> TT
		where TT: Zero + Clone + Copy + Add<Output=TT> + Mul<Output=TT> {
		let data_iter = ReverseWrappingIterator::new(data, i + conv_len, full_conv_len);
		let mut sum = TT::zero();
		let iteration =
			data_iter
			.zip(other_iter);
		for (this, other) in iteration {
			sum = sum + this * (*other);
		}
		sum
	}

	fn convolve_vector_simd<B>(&mut self, buffer: &mut B, vector: &Self)
		where B: Buffer<S, T> {
		if self.is_complex() {
			self.convolve_vector_simd_impl(
				buffer,
				vector,
				|x| array_to_complex(x),
				|x| array_to_complex_mut(x),
				|x,y| x.mul_complex(y),
				|x| x.sum_complex())
		} else {
			self.convolve_vector_simd_impl(
				buffer,
				vector,
				|x| x,
				|x| x,
				|x,y| x * y,
				|x| x.sum_real())
		}
	}

	/// Creates shifted and reversed copies of the given data vector.
	/// This function is especially designed for convolutions.
	fn create_shifted_copies(&self) -> Vec<Vec<T::Reg>>{
		let step = if self.is_complex() { 2 } else { 1 };
		let number_of_shifts = T::Reg::len() / step;
		let mut shifted_copies = Vec::with_capacity(number_of_shifts);
		let mut i = 0;
		while i < number_of_shifts {
			let mut data = self.data.to_slice().iter().rev();

			// In general (number_of_shifts - i) indicates which prepared vector we need to use
			// if we later calculate end % number_of_shifts. Some examples:
			// number_of_shifts: 4, end: 13 -> mod: 1. The code will round end to the next SIMD register
			// which ends at 16. In order to get back to 13 we therefore have to ignore 3 numbers.
			// Ignoring is done by shifting and inserting zeros. So in this example the correct shift is 3
			// which equals number_of_shifts(4) - mod(1).
			// Now mod: 0 is a special case. This is because if we round up to the next SIMD register then
			// we still don't need to add any offset and so for the case 0, 0 is the right shift.
			let shift = match i {
				0 => 0,
				x => (number_of_shifts - x) * step
			};
			let min_len = self.len() + shift;
			let len =  (min_len + T::Reg::len() - 1) / T::Reg::len();
			let mut copy: Vec<T::Reg> = Vec::with_capacity(len);

			let mut j = len  * T::Reg::len();;
			let mut k = 0;
			let mut current = vec!(T::zero(); T::Reg::len());
			while j > 0 {
				j -= step;
				if j < shift || j >= min_len {
					current[k] = T::zero();
					k += 1;
					if k >= current.len() {
						copy.push(T::Reg::load_unchecked(&current, 0));
						k = 0;
					}
					if step > 1 {
						current[k] = T::zero();
						k += 1;
						if k >= current.len() {
							copy.push(T::Reg::load_unchecked(&current, 0));
							k = 0;
						}
					}
				} else {
					if step > 1 {
						let im = *data.next().unwrap();
						let re = *data.next().unwrap();
						current[k] = re;
						k += 1;
						if k >= current.len() {
							copy.push(T::Reg::load_unchecked(&current, 0));
							k = 0;
						}
						current[k] = im;
						k += 1;
						if k >= current.len() {
							copy.push(T::Reg::load_unchecked(&current, 0));
							k = 0;
						}
					}
					else {
						current[k] = *data.next().unwrap();
						k += 1;
						if k >= current.len() {
							copy.push(T::Reg::load_unchecked(&current, 0));
							k = 0;
						}
					}
				}
			}

			assert_eq!(k, 0);
			assert_eq!(copy.len(), len);
			shifted_copies.push(copy);
			i += 1;
		}
		shifted_copies
	}

	fn convolve_vector_simd_impl<B, TT, C, CMut, RMul, RSum>(
		&mut self,
		buffer: &mut B,
		vector: &Self,
		convert: C,
		convert_mut: CMut,
		simd_mul: RMul,
		simd_sum: RSum)
			where
				B: Buffer<S, T>,
				TT: Zero + Clone + Copy + Add<Output=TT> + Mul<Output=TT>,
				C: Fn(&[T]) -> &[TT],
				CMut: Fn(&mut [T]) -> &mut [TT],
				RMul: Fn(T::Reg, T::Reg) -> T::Reg,
				RSum: Fn(T::Reg) -> TT {
		let points = self.points();
		let other_points = vector.points();
		assert!(other_points < points);
		let (other_start, other_end, full_conv_len, conv_len) =
					(0, other_points, other_points, other_points - other_points / 2);
		let len = self.len();
		let mut temp = buffer.get(len);
		{
			let other = vector.data.to_slice();
			let data = self.data.to_slice();
			let temp = temp.to_slice_mut();
			let points = self.points();
			let other = convert(&other[0..vector.len()]);
			let complex = convert(&data[0..len]);
			let dest = convert_mut(&mut temp[0..len]);
			let other_iter = &other[other_start .. other_end];

			let shifts = Self::create_shifted_copies(vector);

			let scalar_len = conv_len + T::Reg::len(); // + $reg::len() due to rounding of odd numbers
			let conv_len = conv_len as isize;
			let mut i = 0;
			for num in &mut dest[0..scalar_len] {
				*num = Self::convolve_iteration(complex, other_iter, i, conv_len, full_conv_len);
				i += 1;
			}

			let (scalar_left, _, vectorization_length) = T::Reg::calc_data_alignment_reqs(&data[0..len]);
			let simd = T::Reg::array_to_regs(&data[scalar_left..vectorization_length]);
			for num in &mut dest[scalar_len .. points - scalar_len] {
				let end = (i + conv_len) as usize;
				let shift = end % shifts.len();
				let end = (end + shifts.len() - 1) / shifts.len();
				let mut sum = T::Reg::splat(T::zero());
				let shifted = &shifts[shift];
				let simd_iter = simd[end - shifted.len() .. end].iter();
				let iteration =
					simd_iter
					.zip(shifted);
				for (this, other) in iteration {
					sum = sum + simd_mul(*this, *other);
				}
				(*num) = simd_sum(sum);
				i += 1;
			}

			for num in &mut dest[points - scalar_len .. points] {
				*num = Self::convolve_iteration(complex, other_iter, i, conv_len, full_conv_len);
				i += 1;
			}
		}
		mem::swap(&mut temp, &mut self.data);
		buffer.free(temp);
	}

	fn multiply_function_priv<TT, CMut, FA, F>(
		&mut self,
		is_symmetric: bool,
		ratio: T,
		convert_mut: CMut,
		function_arg: FA,
		fun: F)
			where
				CMut: Fn(&mut [T]) -> &mut [TT],
				FA: Copy + Sync + Send,
				F: Fn(FA, T)->TT + 'static + Sync,
				TT: Zero + Mul<Output=TT> + Copy + Send + Sync + From<T>
	{
		if !is_symmetric {
			let len = self.len();
			let points = self.points();
			let mut data = self.data.to_slice_mut();
			let converted = convert_mut(&mut data[0..len]);
			Chunk::execute_with_range(
				Complexity::Medium, &self.multicore_settings,
				converted, 1, (ratio, function_arg),
				move |array, range, (ratio, arg)| {
					let two = T::from(2.0).unwrap();
					let scale = TT::from(ratio);
					let offset = if points % 2 != 0 { 1 } else { 0 };
					let max = T::from(points - offset).unwrap() / two;
					let mut j = -(T::from(points - offset).unwrap()) / two + T::from(range.start).unwrap();
					for num in array {
						*num = (*num) * scale * fun(arg, j / max * ratio);
						j = j + T::one();
					}
				});
		} else {
			let len = self.len();
			let points = self.points();
			let mut data = self.data.to_slice_mut();
			let converted = convert_mut(&mut data[0..len]);
			Chunk::execute_sym_pairs_with_range(
				Complexity::Medium, &self.multicore_settings,
				converted, 1, (ratio, function_arg),
				move |array1, range1, array2, range2, (ratio, arg)| {
					let two = T::from(2.0).unwrap();
					assert!(array1.len() >= array2.len());
					assert!(range1.end <= range2.start);let scale = TT::from(ratio);
					let len1 = array1.len();
					let len2 = array2.len();
					let offset = if points % 2 != 0 { 1 } else { 0 };
					let max = T::from(points - offset).unwrap() / two;
					let mut j1 = -(T::from(points - offset).unwrap()) / two + T::from(range1.start).unwrap();
					let mut j2 = (T::from(points - offset).unwrap()) / two - T::from(range2.end - 1).unwrap();
					let mut i1 = 0;
					let mut i2 = 0;
					{
						let mut iter1 = array1.iter_mut();
						let mut iter2 = array2.iter_mut().rev();
						while j1 < j2 {
							let num = iter1.next().unwrap();
							(*num) = (*num) * scale * fun(arg, j1 / max * ratio);
							j1 = j1 + T::one();
							i1 += 1;
						}
						while j2 < j1 {
							let num = iter2.next().unwrap();
							(*num) = (*num) * scale * fun(arg, j2 / max * ratio);
							j2 = j2 + T::one();
							i2 += 1;
						}
						// At this point we can be sure that `j1 == j2`
						for (num1, num2) in iter1.zip(iter2) {
							let arg = scale * fun(arg, j1 / max * ratio);
							*num1 = (*num1) * arg;
							*num2 = (*num2) * arg;
							j1 = j1 + T::one();
						}
						j2 = j1;
					}

					// Now we have to deal with differences in length
					// `common_length` is the number of iterations we spent
					// in the previous loop.
					let pos1 = len1 - i1;
					let pos2 = len2 - i2;
					let common_length = if pos1 < pos2 { pos1 } else { pos2 };
					for num in &mut array1[i1 + common_length..len1] {
						(*num) = (*num) * scale * fun(arg, j1 / max * ratio);
						j1 = j1 + T::one();
					}
					for num in &mut array2[0..len2-common_length-i2] {
						(*num) = (*num) * scale * fun(arg, j2 / max * ratio);
						j2 = j2 + T::one();
					}
				});
		}
	}
}

struct WrappingIterator<T>
    where T: Clone {
    start: *const T,
    end: *const T,
    pos: *const T,
    count: usize
}

impl<T> Iterator for WrappingIterator<T>
    where T: Clone {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.count == 0 {
                return None;
            }

            let mut n = self.pos;
            if n < self.end {
                n = n.offset(1);
            } else {
                n = self.start;
            }

            self.pos = n;
            self.count -= 1;
            Some((*n).clone())
        }
    }
}

impl<T> WrappingIterator<T>
    where T: Clone {
    pub fn new(slice: &[T], pos: isize, iter_len: usize) -> Self {
        use std::isize;

        assert!(slice.len() <= isize::MAX as usize);
        let len = slice.len() as isize;
        let mut pos = pos % len;
        while pos < 0 {
            pos += len;
        }

        let start = slice.as_ptr();
        unsafe {
            WrappingIterator {
                start: start,
                end: start.offset(len - 1),
                pos: start.offset(pos),
                count: iter_len
            }
        }
    }
}

struct ReverseWrappingIterator<T>
    where T: Clone {
    start: *const T,
    end: *const T,
    pos: *const T,
    count: usize
}

impl<T> Iterator for ReverseWrappingIterator<T>
    where T: Clone {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.count == 0 {
                return None;
            }

            let mut n = self.pos;
            if n > self.start {
                n = n.offset(-1);
            } else {
                n = self.end;
            }

            self.pos = n;
            self.count -= 1;
            Some((*n).clone())
        }
    }
}

impl<T> ReverseWrappingIterator<T>
    where T: Clone {
    pub fn new(slice: &[T], pos: isize, iter_len: usize) -> Self {
        use std::isize;

        assert!(slice.len() <= isize::MAX as usize);
        let len = slice.len() as isize;
        let mut pos = pos % len;
        while pos < 0 {
            pos += len;
        }

        let start = slice.as_ptr();
        unsafe {
            ReverseWrappingIterator {
                start: start,
                end: start.offset(len - 1),
                pos: start.offset(pos),
                count: iter_len
            }
        }
    }
}
