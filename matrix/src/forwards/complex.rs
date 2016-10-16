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

			impl <V: Vector<T> + ToComplexResult, S: ToSlice<T>, T: RealNumber>
				ToComplexResult for $matrix<V, S, T>
				where <V as ToComplexResult>::ComplexResult: Vector<T> {
					type ComplexResult = $matrix<V::ComplexResult, S, T>;
			}

			impl<V: Vector<T> + ComplexToRealTransformsOps<T>, S: ToSlice<T>, T: RealNumber>
					ComplexToRealTransformsOps<T> for $matrix<V, S, T>
					where <V as ToRealResult>::RealResult: Vector<T> {
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
		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
