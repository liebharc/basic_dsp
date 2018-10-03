use super::*;
use std::marker;
use TransformContent;

macro_rules! try_transform {
    ($op: expr, $matrix: ident) => {{
        match $op {
            Ok(rows) => Ok($matrix {
                rows: rows,
                storage_type: marker::PhantomData,
                number_type: marker::PhantomData,
            }),
            Err((r, rows)) => Err((
                r,
                $matrix {
                    rows: rows,
                    storage_type: marker::PhantomData,
                    number_type: marker::PhantomData,
                },
            )),
        }
    }};
}

macro_rules! add_mat_impl {
    ($($matrix:ident);*) => {
        $(
			impl <V: Vector<T> + ToComplexResult, S: ToSlice<T>, T: RealNumber>
				ToComplexResult for $matrix<V, S, T>
				where <V as ToComplexResult>::ComplexResult: Vector<T> {
					type ComplexResult = $matrix<V::ComplexResult, S, T>;
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber>
					RealToComplexTransformsOps<T> for $matrix<V, S, T>
					where <V as ToComplexResult>::ComplexResult: Vector<T>,
                          V: RealToComplexTransformsOps<T> {
				fn to_complex(self) -> TransRes<Self::ComplexResult> {
					let rows = self.rows.transform_res(|v|v.to_complex());
					try_transform!(rows, $matrix)
				}
			}

			impl<V: Vector<T>, S: ToSliceMut<T>, T: RealNumber>
					RealToComplexTransformsOpsBuffered<S, T> for $matrix<V, S, T>
					where <V as ToComplexResult>::ComplexResult: Vector<T>,
                          V: RealToComplexTransformsOpsBuffered<S, T> {
				fn to_complex_b<B>(self, buffer: &mut B) -> Self::ComplexResult
                    where B: for<'b> Buffer<'b, S, T> {
					let rows = self.rows.transform(|v|v.to_complex_b(buffer));
                    $matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> RealOps
                    for $matrix<V, S, T>
                    where V: RealOps {
				fn abs(&mut self) {
                    for v in self.rows_mut() {
                        v.abs();
                    }
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> ModuloOps<T>
                    for $matrix<V, S, T>
                    where V: ModuloOps<T> {
				fn wrap(&mut self, divisor: T) {
                    for v in self.rows_mut() {
                        v.wrap(divisor);
                    }
				}

				fn unwrap(&mut self, divisor: T) {
                    for v in self.rows_mut() {
                        v.unwrap(divisor);
                    }
				}
			}

			impl<S: ToSlice<T>, V: Vector<T> + ApproximatedOps<T>, T: RealNumber>
					ApproximatedOps<T> for $matrix<V, S, T> {
				fn ln_approx(&mut self)  {
					for v in self.rows_mut() {
						v.ln_approx()
					}
				}

				fn exp_approx(&mut self)  {
					for v in self.rows_mut() {
						v.exp_approx()
					}
				}

				fn sin_approx(&mut self)  {
					for v in self.rows_mut() {
						v.sin_approx()
					}
				}

				fn cos_approx(&mut self)  {
					for v in self.rows_mut() {
						v.cos_approx()
					}
				}

				fn log_approx(&mut self, base: T)  {
					for v in self.rows_mut() {
						v.log_approx(base)
					}
				}

				fn expf_approx(&mut self, base: T)  {
					for v in self.rows_mut() {
						v.expf_approx(base)
					}
				}

				fn powf_approx(&mut self, exponent: T)  {
					for v in self.rows_mut() {
						v.powf_approx(exponent)
					}
				}
			}
		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
