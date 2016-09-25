use RealNumber;
use conv_types::*;
use num::{Complex, Zero};
use std::mem;
use std::ops::*;
use super::super::{
	array_to_complex, array_to_complex_mut,
	VoidResult, ToSliceMut,
	DspVec, NumberSpace, TimeDomain,
	DataDomain, Vector,
	Buffer
};

/// Provides a convolution operations.
pub trait Convolution<S, T, C>
    where S: ToSliceMut<T>,
		  T: RealNumber {
    /// Convolves `self` with the convolution function `impulse_response`. For performance consider to
    /// to use `FrequencyMultiplication` instead of this operation depending on `len`.
    ///
    /// An optimized convolution algorithm is used if  `1.0 / ratio` is an integer (inside a `1e-6` tolerance)
    /// and `len` is smaller than a threshold (`202` right now).
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeComplex`: if `self` is in real number space but `impulse_response` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    fn convolve<B>(&mut self, buffer: &mut B, impulse_response: C, ratio: T, len: usize)
		where B: Buffer<S, T>;
}

/// Provides a convolution operation for types which at some point are slice based.
pub trait ConvolutionOps {
    /// Convolves `self` with the convolution function `impulse_response`. For performance it's recommended
    /// to use multiply both vectors in frequency domain instead of this operation.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    /// 2. `VectorMetaDataMustAgree`: in case `self` and `impulse_response` are not in the same number space and same domain.
    /// 3. `InvalidArgumentLength`: if `self.points() < impulse_response.points()`.
    fn convolve_vector(&mut self, impulse_response: &Self) -> VoidResult;
}

/// Provides a frequency response multiplication operations.
pub trait FrequencyMultiplication<T, C>
    where T: RealNumber {
    /// Multiplies `self` with the frequency response function `frequency_response`.
    ///
    /// In order to multiply a vector with another vector in frequency response use `mul`.
    /// # Assumptions
    /// The operation assumes that the vector contains a full spectrum centered at 0 Hz. If half a spectrum
    /// or a FFT shifted spectrum is provided the operation will come back with invalid results.
    /// # Failures
    /// TransRes may report the following `ErrorReason` members:
    ///
    /// 1. `VectorMustBeComplex`: if `self` is in real number space but `frequency_response` is in complex number space.
    /// 2. `VectorMustBeInFreqDomain`: if `self` is in time domain.
    fn multiply_frequency_response(&mut self, frequency_response: C, ratio: T) -> VoidResult;
}

macro_rules! assert_time {
    ($self_: ident) => {
        if $self_.domain() != DataDomain::Time {
            $self_.valid_len = 0;
			return;
        }
    }
}

impl<S, T, N, D> DspVec<S, T, N, D>
	where S: ToSliceMut<T>,
		  T: RealNumber,
		  N: NumberSpace,
		  D: TimeDomain
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
}

impl<'a, S, T, N, D> Convolution<S, T, &'a RealImpulseResponse<T>> for DspVec<S, T, N, D>
	where S: ToSliceMut<T>,
		  T: RealNumber,
		  N: NumberSpace,
		  D: TimeDomain {
	fn convolve<B>(&mut self, buffer: &mut B, function: &RealImpulseResponse<T>, ratio: T, len: usize)
	 	where B: Buffer<S, T> {
		assert_time!(self);
		if !self.is_complex() {
			/*let ratio_inv = T::one() / ratio;
			if len <= 202 && self.len() > 2000 && (ratio_inv.round() - ratio_inv).abs() < T::from(1e-6).unwrap() && ratio > T::from(0.5).unwrap() {
				let mut imp_resp = ComplexTimeVector::<T>::from_constant_with_delta(
					Complex::<T>::zero(),
					(2 * len + 1) * ratio as usize,
					self.delta());
				let mut i = 0;
				let mut j = -(T::from(len).unwrap());
				while i < imp_resp.len() {
					let value = function.calc(j * ratio_inv);
					imp_resp[i] = value;
					i += 1;
					j += 1.0;
				}

				return self.convolve_vector(&imp_resp.to_gen_borrow());
			}*/

			self.convolve_function_priv(
					buffer,
					ratio,
					len,
					|data|data,
					|temp|temp,
					|x|function.calc(x)
				);
		} else {
			/*let ratio_inv = 1.0 / ratio;
			if len <= 202 && self.len() > 2000 && (ratio_inv.round() - ratio_inv).abs() < T::from(1e-6).unwrap() && ratio > T::from(0.5).unwrap() {
				let mut imp_resp = ComplexTimeVector::<T>::from_constant_with_delta(Complex::<T>::zero(), (2 * len + 1) * ratio as usize, self.delta());
				let mut i = 0;
				let mut j = -(T::from(len).unwrap());
				while i < imp_resp.len() {
					let value = function.calc(j * ratio_inv);
					imp_resp[i] = value;
					i += 2;
					j += 1.0;
				}

				return self.convolve_vector(&imp_resp.to_gen_borrow());
			}*/

			self.convolve_function_priv(
				buffer,
				ratio,
				len,
				|data|array_to_complex(data),
				|temp|array_to_complex_mut(temp),
				|x|Complex::<T>::new(function.calc(x), T::zero())
			);
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

#[cfg(test)]
mod tests {
    use super::{WrappingIterator, ReverseWrappingIterator};
    use super::super::super::*;
    use conv_types::*;
    use RealNumber;
    use std::fmt::Debug;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
        where T: RealNumber + Debug {
        assert_eq!(left.len(), right.len());
        for i in 0..left.len() {
            if (left[i] - right[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?}", left, right);
            }
        }
    }
	/*
    #[test]
    fn convolve_complex_freq_and_freq32() {
        let vector = vec!(1.0; 10).to_complex_freq_vec();
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
		let mut buffer = SingleBuffer::new();
        let result = vector.multiply_frequency_response(&mut buffer, &rc as &RealFrequencyResponse<f32>, 2.0).unwrap();
        let expected =
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq_tol(result.interleaved(0..), &expected, 1e-4);
    }

    #[test]
    fn convolve_complex_freq_and_freq_even32() {
        let vector = ComplexFreqVector32::from_constant(Complex32::new(1.0, 1.0), 6);
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(1.0);
        let result = vector.multiply_frequency_response(&rc as &RealFrequencyResponse<f32>, 2.0).unwrap();
        let expected =
            [0.0, 0.0, 0.5, 0.5, 1.5, 1.5, 2.0, 2.0, 1.5, 1.5, 0.5, 0.5];
        assert_eq_tol(result.interleaved(0..), &expected, 1e-4);
    }*/

    #[test]
    fn convolve_real_time_and_time32() {
        let mut vector = vec!(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0).to_real_time_vec();
        let rc: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
		let mut buffer = SingleBuffer::new();
        vector.convolve(&mut buffer, &rc as &RealImpulseResponse<f32>, 0.2, 5);
        let expected =
            [0.0, 0.2171850639713355, 0.4840621929215732, 0.7430526238101408, 0.9312114164253432,
             1.0, 0.9312114164253432, 0.7430526238101408, 0.4840621929215732, 0.2171850639713355];
        assert_eq_tol(&vector[..], &expected, 1e-4);
    }

    #[test]
    fn convolve_complex_time_and_time32() {
        let len = 11;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        time[len] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
		let mut buffer = SingleBuffer::new();
        time.convolve(&mut buffer, &sinc as &RealImpulseResponse<f32>, 0.5, len / 2);
        let res = time.magnitude();
        let expected =
            [0.12732396, 0.000000027827534, 0.21220659, 0.000000027827534, 0.63661975,
             1.0, 0.63661975, 0.000000027827534, 0.21220659, 0.000000027827534, 0.12732396];
        assert_eq_tol(&res[..], &expected, 1e-4);
    }
/*
    #[test]
    fn compare_conv_freq_mul() {
        let len = 11;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), len);
        time[len] = 1.0;
        let freq = time.clone().fft().unwrap();
        let sinc: SincFunction<f32> = SincFunction::new();
        let ratio = 0.5;
        let freq_res = freq.multiply_frequency_response(&sinc as &RealFrequencyResponse<f32>, 1.0 / ratio).unwrap();
        let time_res = time.convolve(&sinc as &RealImpulseResponse<f32>, 0.5, len).unwrap();
        let ifreq_res = freq_res.ifft().unwrap();
        let time_res = time_res.magnitude().unwrap();
        let ifreq_res = ifreq_res.magnitude().unwrap();
        assert_eq!(ifreq_res.is_complex(), time_res.is_complex());
        assert_eq!(ifreq_res.domain(), time_res.domain());
        assert_eq_tol(time_res.real(0..), ifreq_res.real(0..), 0.2);
    }*/

    #[test]
    fn invalid_length_parameter() {
        let len = 20;
        let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
        let sinc: SincFunction<f32> = SincFunction::new();
		let mut buffer = SingleBuffer::new();
        time.convolve(&mut buffer, &sinc as &RealImpulseResponse<f32>, 0.5, 10 * len);
        // As long as we don't panic we are happy with the error handling here
    }
/*
    #[test]
    fn convolve_complex_vectors32() {
        const LEN: usize = 11;
        let mut time = ComplexTimeVector32::from_constant(Complex32::new(0.0, 0.0), LEN);
        time[LEN] = 1.0;
        let sinc: SincFunction<f32> = SincFunction::new();
        let mut argument_data = [0.0; LEN];
        {
            let mut v = -5.0;
            for a in &mut argument_data {
                *a = (&sinc as &RealImpulseResponse<f32>).calc(v * 0.5);
                v += 1.0;
            }
        }
        let argument = ComplexTimeVector32::from_real_imag(&argument_data, &[0.0; LEN]);
        assert_eq!(time.points(), argument.points());
        let result = time.convolve_vector(&argument).unwrap();
        assert_eq!(result.points(), LEN);
        let result = result.magnitude().unwrap();
        assert_eq!(result.points(), LEN);
        let expected =
            [0.12732396, 0.000000027827534, 0.21220659, 0.000000027827534, 0.63661975,
             1.0, 0.63661975, 0.000000027827534, 0.21220659, 0.000000027827534, 0.12732396];
        assert_eq_tol(result.real(0..), &expected, 1e-4);
    }*/

    #[test]
    fn wrapping_iterator() {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut iter = WrappingIterator::new(&array, -3, 8);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 3.0);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 1.0);
    }

    #[test]
    fn wrapping_rev_iterator() {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut iter = ReverseWrappingIterator::new(&array, 2, 5);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 5.0);
        assert_eq!(iter.next().unwrap(), 4.0);
        assert_eq!(iter.next().unwrap(), 3.0);
    }
/*
    #[test]
    fn vector_conv_vs_freq_multiplication() {
        let a = ComplexTimeVector32::from_interleaved(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = ComplexTimeVector32::from_interleaved(&[15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0]);
        let conv = a.clone().convolve_vector(&b).unwrap();
        let a = a.fft().unwrap();
        let b = b.fft().unwrap();
        let mul = a.mul(&b).unwrap();
        let mul = mul.ifft().unwrap();
        let mul = mul.reverse().unwrap();
        let mul = mul.swap_halves().unwrap();
        assert_eq_tol(mul.interleaved(0..), conv.interleaved(0..), 1e-4);
    }

    #[test]
    fn shift_left_by_1_as_conv() {
        let a = RealTimeVector32::from_array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = RealTimeVector32::from_array(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let conv = a.convolve_vector(&b).unwrap();
        let conv = conv.magnitude().unwrap();
        let exp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq_tol(conv.real(0..), &exp, 1e-4);
    }

    #[test]
    fn shift_left_by_1_as_conv_shorter() {
        let a = RealTimeVector32::from_array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = RealTimeVector32::from_array(&[0.0, 0.0, 1.0]);
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let conv = a.convolve_vector(&b).unwrap();
        let conv = conv.magnitude().unwrap();
        let exp = [9.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq_tol(conv.real(0..), &exp, 1e-4);
    }

    #[test]
    fn vector_conv_vs_freq_multiplication_pure_real_data() {
        let a = RealTimeVector32::from_array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = RealTimeVector32::from_array(&[15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0]);
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let conv = a.clone().convolve_vector(&b).unwrap();
        let a = a.fft().unwrap();
        let b = b.fft().unwrap();
        let mul = a.mul(&b).unwrap();
        let mul = mul.ifft().unwrap();
        let mul = mul.magnitude().unwrap();
        let mul = mul.reverse().unwrap();
        let mul = mul.swap_halves().unwrap();
        let conv = conv.magnitude().unwrap();
        assert_eq_tol(mul.real(0..), conv.real(0..), 1e-4);
    }

    #[test]
    fn vector_conv_vs_freq_multiplication_pure_real_data_odd() {
        let a = RealTimeVector32::from_array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = RealTimeVector32::from_array(&[15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0]);
        let a = a.to_complex().unwrap();
        let b = b.to_complex().unwrap();
        let conv = a.clone().convolve_vector(&b).unwrap();
        let a = a.fft().unwrap();
        let b = b.fft().unwrap();
        let mul = a.mul(&b).unwrap();
        let mul = mul.ifft().unwrap();
        let mul = mul.magnitude().unwrap();
        let mul = mul.reverse().unwrap();
        let mul = mul.swap_halves().unwrap();
        let conv = conv.magnitude().unwrap();
        assert_eq_tol(mul.real(0..), conv.real(0..), 1e-4);
    }*/
}
