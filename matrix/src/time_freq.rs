
use basic_dsp_vector::*;
use basic_dsp_vector::window_functions::*;
use basic_dsp_vector::conv_types::*;
use super::*;
use TransformContent;
use std::marker;
use std::mem;

macro_rules! try_transform {
    ($op: expr, $matrix: ident) => {
        {
			match $op {
				Ok(rows) => Ok($matrix {
					rows: rows,
					storage_type: marker::PhantomData,
					number_type: marker::PhantomData
				}),
				Err((r, rows)) => Err((
					r,
					$matrix {
						rows: rows,
						storage_type: marker::PhantomData,
						number_type: marker::PhantomData
				})),
			}
        }
    }
}

macro_rules! add_mat_impl {
    ($($matrix:ident);*) => {
        $(
			impl <V: Vector<T> + ToTimeResult, S: ToSlice<T>, T: RealNumber>
				ToTimeResult for $matrix<V, S, T>
				where <V as ToTimeResult>::TimeResult: Vector<T> {
					type TimeResult = $matrix<V::TimeResult, S, T>;
			}

			impl <V: Vector<T> + ToFreqResult, S: ToSlice<T>, T: RealNumber>
				ToFreqResult for $matrix<V, S, T>
				where <V as ToFreqResult>::FreqResult: Vector<T> {
					type FreqResult = $matrix<V::FreqResult, S, T>;
			}

			impl <V: Vector<T> + ToRealTimeResult, S: ToSlice<T>, T: RealNumber>
				ToRealTimeResult for $matrix<V, S, T>
				where <V as ToRealTimeResult>::RealTimeResult: Vector<T> {
					type RealTimeResult = $matrix<V::RealTimeResult, S, T>;
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
					TimeToFrequencyDomainOperations<S, T> for $matrix<V, S, T>
					where <V as ToFreqResult>::FreqResult: Vector<T>,
                          V: TimeToFrequencyDomainOperations<S, T> {
				fn plain_fft<B>(self, buffer: &mut B) -> Self::FreqResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.plain_fft(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn fft<B>(self, buffer: &mut B) -> Self::FreqResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.fft(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn windowed_fft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> Self::FreqResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.windowed_fft(buffer, window));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
					SymmetricTimeToFrequencyDomainOperations<S, T> for $matrix<V, S, T>
					where <V as ToFreqResult>::FreqResult: Vector<T>,
                          V: SymmetricTimeToFrequencyDomainOperations<S, T> {
				fn plain_sfft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
                    where B: Buffer<S, T> {
					let rows = self.rows.transform_res(|v|v.plain_sfft(buffer));
					try_transform!(rows, $matrix)
				}

				fn sfft<B>(self, buffer: &mut B) -> TransRes<Self::FreqResult>
                    where B: Buffer<S, T> {
					let rows = self.rows.transform_res(|v|v.sfft(buffer));
					try_transform!(rows, $matrix)
				}

				fn windowed_sfft<B>(
						self,
						buffer: &mut B,
						window: &WindowFunction<T>) -> TransRes<Self::FreqResult>
                    where B: Buffer<S, T> {
					let rows = self.rows.transform_res(|v|v.windowed_sfft(buffer, window));
					try_transform!(rows, $matrix)
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
					FrequencyToTimeDomainOperations<S, T> for $matrix<V, S, T>
					where <V as ToTimeResult>::TimeResult: Vector<T>,
                          V: FrequencyToTimeDomainOperations<S, T> {
				fn plain_ifft<B>(self, buffer: &mut B) -> Self::TimeResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.plain_ifft(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn ifft<B>(self, buffer: &mut B) -> Self::TimeResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.ifft(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn windowed_ifft<B>(self, buffer: &mut B, window: &WindowFunction<T>) -> Self::TimeResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.windowed_ifft(buffer, window));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
					SymmetricFrequencyToTimeDomainOperations<S, T> for $matrix<V, S, T>
					where <V as ToRealTimeResult>::RealTimeResult: Vector<T>,
                          V: SymmetricFrequencyToTimeDomainOperations<S, T> {
				fn plain_sifft<B>(self, buffer: &mut B) -> TransRes<Self::RealTimeResult>
                    where B: Buffer<S, T> {
					let rows = self.rows.transform_res(|v|v.plain_sifft(buffer));
					try_transform!(rows, $matrix)
				}

				fn sifft<B>(self, buffer: &mut B) -> TransRes<Self::RealTimeResult>
                    where B: Buffer<S, T> {
					let rows = self.rows.transform_res(|v|v.sifft(buffer));
					try_transform!(rows, $matrix)
				}

				fn windowed_sifft<B>(
						self,
						buffer: &mut B,
						window: &WindowFunction<T>) -> TransRes<Self::RealTimeResult>
                    where B: Buffer<S, T> {
					let rows = self.rows.transform_res(|v|v.windowed_sifft(buffer, window));
					try_transform!(rows, $matrix)
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber> FrequencyDomainOperations<S, T>
                    for $matrix<V, S, T>
                    where V: FrequencyDomainOperations<S, T> {
				fn mirror<B>(&mut self, buffer: &mut B) where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.mirror(buffer);
                    }
				}

				fn fft_shift(&mut self) {
                    for v in self.rows_mut() {
                        v.fft_shift();
                    }
				}

				fn ifft_shift(&mut self) {
                    for v in self.rows_mut() {
                        v.ifft_shift();
                    }
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber> TimeDomainOperations<S, T>
                    for $matrix<V, S, T>
                    where V: TimeDomainOperations<S, T> {
				fn apply_window(&mut self, window: &WindowFunction<T>) {
                    for v in self.rows_mut() {
                        v.apply_window(window);
                    }
				}

				fn unapply_window(&mut self, window: &WindowFunction<T>) {
                    for v in self.rows_mut() {
                        v.unapply_window(window);
                    }
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
					CrossCorrelationArgumentOps<S, T> for $matrix<V, S, T>
					where <V as ToFreqResult>::FreqResult: Vector<T>,
                          V: CrossCorrelationArgumentOps<S, T> {
				fn prepare_argument<B>(self, buffer: &mut B) -> Self::FreqResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.prepare_argument(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn prepare_argument_padded<B>(self, buffer: &mut B) -> Self::FreqResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.prepare_argument_padded(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}
			}

            // We haven't implemented this for an argument of type vector
            // since there is no guarantee that FreqResult of a matrix wouldn't
            // be a vector. Thus Rust fails since there could be a case where two
            // implementations of CrossCorrelationOps exists for a matrix. For
            // now this is what it is, until a) we figure out a solution for this
            // or b) Rust adds a new feature so that we can specify that FreqResult is
            // either a matrix or a vecor but never both (Negative bounds could be helpful).
			impl<S: ToSliceMut<T>, T: RealNumber, N: NumberSpace, D: Domain>
					CrossCorrelationOps<
						S,
						T,
						$matrix<<DspVec<S, T, N, D> as ToFreqResult>::FreqResult, S, T>>
					for $matrix<DspVec<S, T, N, D>, S, T>
					where DspVec<S, T, N, D>:
						  	ToFreqResult
							+ CrossCorrelationOps<
								S,
								T,
								<DspVec<S, T, N, D> as ToFreqResult>::FreqResult>,
						  <DspVec<S, T, N, D> as ToFreqResult>::FreqResult: Vector<T> {
				fn correlate<B>(
						&mut self,
						buffer: &mut B,
						other: &$matrix<<DspVec<S, T, N, D> as ToFreqResult>::FreqResult, S, T>)
						-> VoidResult
                    where B: Buffer<S, T> {
					for (v, o) in self.rows_mut().iter_mut().zip(other.rows()) {
						try!(v.correlate(buffer, o));
					}

					Ok(())
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber> InterpolationOps<S, T>
                    for $matrix<V, S, T>
                    where V: InterpolationOps<S, T> {
				fn interpolatef<B>(
						&mut self,
						buffer: &mut B,
						function: &RealImpulseResponse<T>,
						interpolation_factor: T,
						delay: T,
						conv_len: usize)
				 	where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.interpolatef(buffer, function, interpolation_factor, delay, conv_len);
                    }
				}

				fn interpolatei<B>(
						&mut self,
						buffer: &mut B,
                        function: &RealFrequencyResponse<T>,
                        interpolation_factor: u32) -> VoidResult
					where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        try!(v.interpolatei(buffer, function, interpolation_factor));
                    }

					Ok(())
				}

				fn interpolate<B>(
						&mut self,
						buffer: &mut B,
                        function: Option<&RealFrequencyResponse<T>>,
                        dest_points: usize,
                        delay: T) -> VoidResult
					where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        try!(v.interpolate(buffer, function, dest_points, delay));
                    }

					Ok(())
				}

				fn interpft<B>(
						&mut self,
						buffer: &mut B,
                        dest_points: usize)
					where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.interpft(buffer, dest_points);
                    }
				}

				fn decimatei(
						&mut self,
						decimation_factor: u32,
						delay: u32) {
                    for v in self.rows_mut() {
                        v.decimatei(decimation_factor, delay);
                    }
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber> RealInterpolationOps<S, T>
                    for $matrix<V, S, T>
                    where V: RealInterpolationOps<S, T> {
				fn interpolate_hermite<B>(
						&mut self,
						buffer: &mut B,
						interpolation_factor: T,
						delay: T)
				 	where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.interpolate_hermite(buffer, interpolation_factor, delay);
                    }
				}

				fn interpolate_lin<B>(
						&mut self,
						buffer: &mut B,
						interpolation_factor: T,
						delay: T)
					where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.interpolate_lin(buffer, interpolation_factor, delay);
                    }
				}
			}

			impl<'a, V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
                    Convolution<'a, S, T, &'a RealImpulseResponse<T>>
                    for $matrix<V, S, T>
                    where V: Convolution<'a, S, T, &'a RealImpulseResponse<T>> {
				fn convolve<B>(
						&mut self,
						buffer: &mut B,
						impulse_response: &'a RealImpulseResponse<T>,
						ratio: T,
						len: usize)
				 	where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.convolve(buffer, impulse_response, ratio, len);
                    }
				}
			}

			impl<'a, V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
                    Convolution<'a, S, T, &'a ComplexImpulseResponse<T>>
                    for $matrix<V, S, T>
                    where V: Convolution<'a, S, T, &'a ComplexImpulseResponse<T>> {
				fn convolve<B>(
						&mut self,
						buffer: &mut B,
						impulse_response: &'a ComplexImpulseResponse<T>,
						ratio: T,
						len: usize)
				 	where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.convolve(buffer, impulse_response, ratio, len);
                    }
				}
			}

			impl<'a, V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
                        FrequencyMultiplication<'a, S, T, &'a RealFrequencyResponse<T>>
                    for $matrix<V, S, T>
                    where V: FrequencyMultiplication<'a, S, T, &'a RealFrequencyResponse<T>> {
				fn multiply_frequency_response(
						&mut self,
						frequency_response: &'a RealFrequencyResponse<T>,
						ratio: T) {
                    for v in self.rows_mut() {
                        v.multiply_frequency_response(frequency_response, ratio);
                    }
				}
			}

			impl<'a, V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
                        FrequencyMultiplication<'a, S, T, &'a ComplexFrequencyResponse<T>>
                    for $matrix<V, S, T>
                    where V: FrequencyMultiplication<'a, S, T, &'a ComplexFrequencyResponse<T>> {
				fn multiply_frequency_response(
						&mut self,
						frequency_response: &'a ComplexFrequencyResponse<T>,
						ratio: T) {
                    for v in self.rows_mut() {
                        v.multiply_frequency_response(frequency_response, ratio);
                    }
				}
			}

			impl<S: ToSliceMut<T>, T: RealNumber, N: NumberSpace, D: Domain>
                    ConvolutionOps<S, T, DspVec<S, T, N, D>>
                    for $matrix<DspVec<S, T, N, D>, S, T>
                    where DspVec<S, T, N, D>: ConvolutionOps<S, T, DspVec<S, T, N, D>> {
				fn convolve_signal<B>(
						&mut self,
						buffer: &mut B,
						impulse_response: &DspVec<S, T, N, D>) -> VoidResult
							where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        try!(v.convolve_signal(buffer, impulse_response));
                    }

					Ok(())
				}
			}
		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);

macro_rules! convolve_signal {
    ($self_: expr, $buffer: ident, $impulse_response: ident) => {
        {
            let mut error = None;
            let mut result: Vec<S> = {
                let rows: Vec<&DspVec<S, T, N, D>> = $self_.rows().iter().collect();
                $impulse_response.iter().map(|i| {
                    let mut target = $buffer.get($self_.row_len());
                    let res = DspVec::<S, T, N, D>::convolve_mat(&rows, i, &mut target);
                    match res {
                        Ok(()) => (),
                        Err(reason) => error = Some(reason)
                    };
                    target
                }).collect()
           };

           match error {
               None => (),
               Some(err) => return Err(err)
           }

           for (res, row) in result.iter_mut().zip($self_.rows.iter_mut()) {
               mem::swap(res, &mut row.data)
           }

           while result.len() > 0 {
            $buffer.free(result.pop().expect("Result should not be empty"));
           }

           Ok(())
        }
    }
}

impl<'a, S: ToSliceMut<T>, T: RealNumber, N: NumberSpace, D: Domain>
        ConvolutionOps<S, T, Vec<&'a Vec<&'a DspVec<S, T, N, D>>>>
        for MatrixMxN<DspVec<S, T, N, D>, S, T> {
    fn convolve_signal<B>(
            &mut self,
            buffer: &mut B,
            impulse_response: &Vec<&Vec<&DspVec<S, T, N, D>>>) -> VoidResult
                where B: Buffer<S, T> {
        convolve_signal!(self, buffer, impulse_response)
    }
}

impl<'a, S: ToSliceMut<T>, T: RealNumber, N: NumberSpace, D: Domain>
        ConvolutionOps<S, T, [[&'a DspVec<S, T, N, D>; 2]; 2]>
        for Matrix2xN<DspVec<S, T, N, D>, S, T> {
    fn convolve_signal<B>(
            &mut self,
            buffer: &mut B,
            impulse_response: &[[&'a DspVec<S, T, N, D>; 2]; 2]) -> VoidResult
                where B: Buffer<S, T> {
        convolve_signal!(self, buffer, impulse_response)
    }
}

impl<'a, S: ToSliceMut<T>, T: RealNumber, N: NumberSpace, D: Domain>
        ConvolutionOps<S, T, [[&'a DspVec<S, T, N, D>; 3]; 3]>
        for Matrix3xN<DspVec<S, T, N, D>, S, T> {
    fn convolve_signal<B>(
            &mut self,
            buffer: &mut B,
            impulse_response: &[[&'a DspVec<S, T, N, D>; 3]; 3]) -> VoidResult
                where B: Buffer<S, T> {
        convolve_signal!(self, buffer, impulse_response)
    }
}

impl<'a, S: ToSliceMut<T>, T: RealNumber, N: NumberSpace, D: Domain>
        ConvolutionOps<S, T, [[&'a DspVec<S, T, N, D>; 4]; 4]>
        for Matrix4xN<DspVec<S, T, N, D>, S, T> {
    fn convolve_signal<B>(
            &mut self,
            buffer: &mut B,
            impulse_response: &[[&'a DspVec<S, T, N, D>; 4]; 4]) -> VoidResult
                where B: Buffer<S, T> {
        convolve_signal!(self, buffer, impulse_response)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use basic_dsp_vector::conv_types::*;
    use basic_dsp_vector::*;
    use std::fmt::Debug;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
        where T: RealNumber + Debug
    {
        assert_eq!(left.len(), right.len());
        for i in 0..left.len() {
            if (left[i] - right[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?}", left, right);
            }
        }
    }

    #[test]
    fn convolve_complex_time_and_time32() {
        let res = {
            let len = 11;
            let mut time = vec!(0.0; 2 * len).to_complex_time_vec();
            time[len] = 1.0;
            let time2 = time.clone();
            let mut mat = [time, time2].to_mat();
            let sinc: SincFunction<f32> = SincFunction::new();
            let mut buffer = SingleBuffer::new();
            mat.convolve(&mut buffer,
                          &sinc as &RealImpulseResponse<f32>,
                          0.5,
                          len / 2);
            mat.magnitude()
        };

        let expected = [0.12732396,
                        0.000000027827534,
                        0.21220659,
                        0.000000027827534,
                        0.63661975,
                        1.0,
                        0.63661975,
                        0.000000027827534,
                        0.21220659,
                        0.000000027827534,
                        0.12732396];
        let (res, _) = res.get();
        assert_eq_tol(&res[0][..], &expected, 1e-4);
        assert_eq_tol(&res[1][..], &expected, 1e-4);
    }

    #[test]
    fn delay_convolve() {
        let mut mat = {
            let len = 11;
            let mut time = vec!(0.0; len).to_real_time_vec();
            time[len / 2] = 1.0;
            let time2 = time.clone();
            vec!(time, time2).to_mat()
        };

        let len = 3;
        let empty = vec!(0.0; len).to_real_time_vec();
        let mut delay = vec!(0.0; len).to_real_time_vec();
        delay[0] = 1.0;
        let conv1 = vec!(&delay, &empty);
        let conv2 = vec!(&empty, &delay);
        let conv = vec!(&conv1, &conv2);

        let mut buffer = SingleBuffer::new();
        mat.convolve_signal(&mut buffer, &conv).unwrap();

        let expected = {
            let len = 11;
            let mut time = vec!(0.0; len).to_real_time_vec();
            time[len / 2 - 1] = 1.0;
            let time2 = time.clone();
            vec!(time, time2)
        };

        let (res, _) = mat.get();
        assert_eq_tol(&res[0][..], &expected[0][..], 1e-4);
        assert_eq_tol(&res[1][..], &expected[1][..], 1e-4);
    }

    #[test]
    fn delay_swap_convolve() {
        let mut mat = {
            let len = 11;
            let mut time = vec!(0.0; len).to_real_time_vec();
            time[len / 2] = 0.5;
            let mut time2 = vec!(0.0; len).to_real_time_vec();
            time2[len / 2] = 2.0;
            [time, time2].to_mat()
        };

        let len = 3;
        let empty = vec!(0.0; len).to_real_time_vec();
        let mut delay = vec!(0.0; len).to_real_time_vec();
        delay[0] = 1.0;

        // This impulse response will swap both channels
        // and then delay them
        let conv = [[&empty, &delay], [&delay, &empty]];

        let mut buffer = SingleBuffer::new();
        mat.convolve_signal(&mut buffer, &conv).unwrap();

        let expected = {
            let len = 11;
            let mut time = vec!(0.0; len).to_real_time_vec();
            time[len / 2 - 1] = 2.0;
            let mut time2 = vec!(0.0; len).to_real_time_vec();
            time2[len / 2 - 1] = 0.5;
            vec!(time, time2)
        };

        let (res, _) = mat.get();
        assert_eq_tol(&res[0][..], &expected[0][..], 1e-4);
        assert_eq_tol(&res[1][..], &expected[1][..], 1e-4);
    }
}
