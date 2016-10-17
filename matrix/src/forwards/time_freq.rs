
use basic_dsp_vector::*;
use basic_dsp_vector::window_functions::*;
use basic_dsp_vector::conv_types::*;
use super::super::*;
use TransformContent;
use std::marker;

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

				fn fft_shift<B>(&mut self, buffer: &mut B) where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.fft_shift(buffer);
                    }
				}

				fn ifft_shift<B>(&mut self, buffer: &mut B) where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.ifft_shift(buffer);
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

			impl<S: ToSliceMut<T>, T: RealNumber, N: NumberSpace, D: Domain>
					CrossCorrelationOps<
						S,
						T,
						$matrix<<DspVec<S, T, N, D> as ToFreqResult>::FreqResult, S, T>>
					for $matrix<DspVec<S, T, N, D>, S, T>
					where $matrix<DspVec<S, T, N, D>, S, T>: ToFreqResult,
						  DspVec<S, T, N, D>:
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

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber, C> Convolution<S, T, C>
                    for $matrix<V, S, T>
                    where V: Convolution<S, T, C> {
				fn convolve<B>(
						&mut self,
						buffer: &mut B,
						impulse_response: &C,
						ratio: T,
						len: usize)
				 	where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        v.convolve(buffer, impulse_response, ratio, len);
                    }
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber, C> FrequencyMultiplication<S, T, C>
                    for $matrix<V, S, T>
                    where V: FrequencyMultiplication<S, T, C> {
				fn multiply_frequency_response(
						&mut self,
						frequency_response: &C,
						ratio: T) {
                    for v in self.rows_mut() {
                        v.multiply_frequency_response(frequency_response, ratio);
                    }
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber> ConvolutionOps<S, T, V>
                    for $matrix<V, S, T>
                    where V: ConvolutionOps<S, T, V> {
				fn convolve_vector<B>(
						&mut self,
						buffer: &mut B,
						impulse_response: &V) -> VoidResult
							where B: Buffer<S, T> {
                    for v in self.rows_mut() {
                        try!(v.convolve_vector(buffer, impulse_response));
                    }

					Ok(())
				}
			}
		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
