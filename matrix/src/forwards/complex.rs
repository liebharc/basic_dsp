use basic_dsp_vector::*;
use super::super::*;
use TransformContent;
use std::marker;

macro_rules! add_mat_impl {
    ($($matrix:ident);*) => {
        $(
			impl <V: Vector<T> + ToRealResult, S: ToSlice<T>, T: RealNumber>
				ToRealResult for $matrix<V, S, T>
				where <V as ToRealResult>::RealResult: Vector<T> {
					type RealResult = $matrix<V::RealResult, S, T>;
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber>
					ComplexToRealTransformsOps<T> for $matrix<V, S, T>
					where <V as ToRealResult>::RealResult: Vector<T>,
                          V: ComplexToRealTransformsOps<T> {
				fn magnitude(self) -> Self::RealResult {
					let rows = self.rows.transform(|v|v.magnitude());
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn magnitude_squared(self) -> Self::RealResult {
                    let rows = self.rows.transform(|v|v.magnitude_squared());
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn to_real(self) -> Self::RealResult {
                    let rows = self.rows.transform(|v|v.to_real());
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn to_imag(self) -> Self::RealResult {
                    let rows = self.rows.transform(|v|v.to_imag());
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn phase(self) -> Self::RealResult {
                    let rows = self.rows.transform(|v|v.phase());
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
					ComplexToRealTransformsOpsBuffered<S, T> for $matrix<V, S, T>
					where <V as ToRealResult>::RealResult: Vector<T>,
                          V: ComplexToRealTransformsOpsBuffered<S, T> {
				fn magnitude_b<B>(self, buffer: &mut B) -> Self::RealResult
                    where B: Buffer<S, T> {
					let rows = self.rows.transform(|v|v.magnitude_b(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn magnitude_squared_b<B>(self, buffer: &mut B) -> Self::RealResult
                    where B: Buffer<S, T> {
                    let rows = self.rows.transform(|v|v.magnitude_squared_b(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn to_real_b<B>(self, buffer: &mut B) -> Self::RealResult
                    where B: Buffer<S, T> {
                    let rows = self.rows.transform(|v|v.to_real_b(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn to_imag_b<B>(self, buffer: &mut B) -> Self::RealResult
                    where B: Buffer<S, T> {
                    let rows = self.rows.transform(|v|v.to_imag_b(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}

				fn phase_b<B>(self, buffer: &mut B) -> Self::RealResult
                    where B: Buffer<S, T> {
                    let rows = self.rows.transform(|v|v.phase_b(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber>
					ComplexToRealGetterOps<T> for $matrix<V, S, T>
					where <V as ToRealResult>::RealResult: Vector<T>,
                          <$matrix<V, S, T> as ToRealResult>::RealResult:
                            Matrix<<V as ToRealResult>::RealResult, T>,
                          V: ComplexToRealGetterOps<T> {
				fn get_real(&self, destination: &mut Self::RealResult) {
                    for (o, v) in destination.rows_mut().iter_mut().zip(self.rows()) {
						v.get_real(o);
					}
				}

				fn get_imag(&self, destination: &mut Self::RealResult) {
                    for (o, v) in destination.rows_mut().iter_mut().zip(self.rows()) {
						v.get_imag(o);
					}
				}

				fn get_magnitude(&self, destination: &mut Self::RealResult) {
                    for (o, v) in destination.rows_mut().iter_mut().zip(self.rows()) {
						v.get_imag(o);
					}
				}

				fn get_magnitude_squared(&self, destination: &mut Self::RealResult) {
                    for (o, v) in destination.rows_mut().iter_mut().zip(self.rows()) {
						v.get_imag(o);
					}
				}

				fn get_phase(&self, destination: &mut Self::RealResult) {
                    for (o, v) in destination.rows_mut().iter_mut().zip(self.rows()) {
						v.get_imag(o);
					}
				}

				fn get_real_imag(&self,
                                 real: &mut Self::RealResult,
                                 imag: &mut Self::RealResult) {
                    for ((r, i), v) in real.rows_mut().iter_mut()
                                    .zip(imag.rows_mut())
                                    .zip(self.rows()) {
 						v.get_real_imag(r, i);
 					}
				}

				fn get_mag_phase(&self,
                                 mag: &mut Self::RealResult,
                                 phase: &mut Self::RealResult) {
                    for ((r, i), v) in mag.rows_mut().iter_mut()
                                     .zip(phase.rows_mut())
                                     .zip(self.rows()) {
  						v.get_mag_phase(r, i);
  					}
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber>
					ComplexToRealSetterOps<T> for $matrix<V, S, T>
					where <V as ToRealResult>::RealResult: Vector<T>,
                          <$matrix<V, S, T> as ToRealResult>::RealResult:
                            Matrix<<V as ToRealResult>::RealResult, T>,
                          V: ComplexToRealSetterOps<T> {
				fn set_real_imag(&mut self,
                                 real: &Self::RealResult,
                                 imag: &Self::RealResult) -> VoidResult {
                    for ((v, r), i) in self.rows_mut().iter_mut()
                                    .zip(real.rows())
                                    .zip(imag.rows()) {
 						try!(v.set_real_imag(r, i));
 					}

                    Ok(())
				}

				fn set_mag_phase(&mut self,
                                 mag: &Self::RealResult,
                                 phase: &Self::RealResult) -> VoidResult {
                    for ((v, r), i) in self.rows_mut().iter_mut()
                                     .zip(mag.rows())
                                     .zip(phase.rows()) {
  						try!(v.set_mag_phase(r, i));
  					}

                    Ok(())
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> ComplexOps<T>
                    for $matrix<V, S, T>
                    where V: ComplexOps<T> {
				fn multiply_complex_exponential(&mut self, a: T, b: T) {
                    for v in self.rows_mut() {
                        v.multiply_complex_exponential(a, b);
                    }
				}

				fn conj(&mut self) {
                    for v in self.rows_mut() {
                        v.conj();
                    }
				}
			}
		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
