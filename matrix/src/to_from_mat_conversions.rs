use basic_dsp_vector::*;
use super::*;
use std::mem;
use std::marker;

/// Conversion from a generic data type into a dsp matrix which tracks
/// its meta information (domain and number space)
/// only at runtime. See `ToRealMatrix` and
/// `ToComplexMatrix` for alternatives which track most of the meta data
/// with the type system and therefore avoid runtime errors.
pub trait ToDspMatrix<V, T>
    where V: Vector<T>,
		  T: RealNumber {
	type Output: Matrix<V, T>;

    /// Create a new generic matrix.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex matrices with an odd length the resulting value will have a zero length.
    fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output;
}

/// Conversion from a generic data type into a dsp matrix with real data.
pub trait ToRealTimeMatrix<V, T>
    where V: Vector<T>,
		  T: RealNumber {
	type Output: Matrix<V, T>;

    /// Create a new matrix in real number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_time_mat(self) -> Self::Output;
}

/// Conversion from a generic data type into a dsp matrix with real data.
pub trait ToRealFreqMatrix<V, T>
    where V: Vector<T>,
		  T: RealNumber {
	type Output: Matrix<V, T>;

    /// Create a new vector in real number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_freq_mat(self) -> Self::Output;
}

/// Conversion from a generic data type into a dsp vector with complex data.
pub trait ToComplexTimeMatrix<V, T>
    where V: Vector<T>,
		  T: RealNumber {
	type Output: Matrix<V, T>;

    /// Create a new matrix in complex number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex matrices with an odd length the resulting value will have a zero length.
    fn to_complex_time_mat(self) -> Self::Output;
}

/// Conversion from a generic data type into a dsp vector with complex data.
pub trait ToComplexFreqMatrix<V, T>
    where V: Vector<T>,
		  T: RealNumber {
	type Output: Matrix<V, T>;

    /// Create a new matrix in complex number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex matrices with an odd length the resulting value will have a zero length.
    fn to_complex_freq_mat(self) -> Self::Output;
}

/// Retrieves the underlying storage from a matrix.
pub trait FromMatrix<T>
    where T: RealNumber {
    /// Type of the underlying storage of a matrix.
    type Output;

    /// Gets the underlying matrix and the number of elements which
    /// contain valid.
    fn get(self) -> (Self::Output, usize);
}

fn to_mat_mxn<S, F, V, T>(mut source: Vec<S>, conversion: F) -> MatrixMxN<V, T>
	where S: ToSlice<T>,
	 	  V: Vector<T>,
		  T: RealNumber,
		  F: Fn(S) -> V {
	  let mut rows: Vec<V> = Vec::with_capacity(source.len());
	  for _ in 0..source.len() {
	  	let v: S = source.pop().unwrap();
	  	rows.push(conversion(v));
	  }
	  rows.reverse();

	  MatrixMxN {
	  	rows: rows,
	  	number_type: marker::PhantomData
	  }
}

fn to_mat_2xn<S, F, V, T>(mut source: [S; 2], conversion: F) -> Matrix2xN<V, T>
	where S: ToSlice<T>,
	 	  V: Vector<T>,
		  T: RealNumber,
		  F: Fn(S) -> V {
	  unsafe {
		  let first = mem::replace(&mut source[0], mem::uninitialized());
		  let second = mem::replace(&mut source[1], mem::uninitialized());
		  mem::forget(source); // TODO possible memory leak
		  let first = conversion(first);
		  let second = conversion(second);
		  let rows: [V; 2] = [first, second];

		  Matrix2xN {
			  rows: rows,
			  number_type: marker::PhantomData
		  }
	  }
}

fn to_mat_3xn<S, F, V, T>(mut source: [S; 3], conversion: F) -> Matrix3xN<V, T>
	where S: ToSlice<T>,
	 	  V: Vector<T>,
		  T: RealNumber,
		  F: Fn(S) -> V {
	  unsafe {
		  let first = mem::replace(&mut source[0], mem::uninitialized());
		  let second = mem::replace(&mut source[1], mem::uninitialized());
		  let third = mem::replace(&mut source[2], mem::uninitialized());
		  mem::forget(source); // TODO possible memory leak
		  let first = conversion(first);
		  let second = conversion(second);
		  let third = conversion(third);
		  let rows: [V; 3] = [first, second, third];

		  Matrix3xN {
			  rows: rows,
			  number_type: marker::PhantomData
		  }
	  }
}

fn to_mat_4xn<S, F, V, T>(mut source: [S; 4], conversion: F) -> Matrix4xN<V, T>
	where S: ToSlice<T>,
	 	  V: Vector<T>,
		  T: RealNumber,
		  F: Fn(S) -> V {
	  unsafe {
		  let first = mem::replace(&mut source[0], mem::uninitialized());
		  let second = mem::replace(&mut source[1], mem::uninitialized());
		  let third = mem::replace(&mut source[2], mem::uninitialized());
		  let fourth = mem::replace(&mut source[3], mem::uninitialized());
		  mem::forget(source); // TODO possible memory leak
		  let first = conversion(first);
		  let second = conversion(second);
		  let third = conversion(third);
		  let fourth = conversion(fourth);
		  let rows: [V; 4] = [first, second, third, fourth];

		  Matrix4xN {
			  rows: rows,
			  number_type: marker::PhantomData
		  }
	  }
}

impl<T, S> ToDspMatrix<GenDspVec<S, T>, T> for Vec<S>
	where T: RealNumber,
		  S: ToDspVector<T> + ToSlice<T> {
	type Output = MatrixMxN<GenDspVec<S, T>, T>;

	fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output {
		to_mat_mxn(self, |v|v.to_gen_dsp_vec(is_complex, domain))
	}
}

impl<T, S> ToDspMatrix<GenDspVec<S, T>, T> for [S; 2]
	where T: RealNumber,
		  S: ToDspVector<T> + ToSlice<T> {
	type Output = Matrix2xN<GenDspVec<S, T>, T>;

	fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output {
		to_mat_2xn(self, |v|v.to_gen_dsp_vec(is_complex, domain))
	}
}

impl<T, S> ToDspMatrix<GenDspVec<S, T>, T> for [S; 3]
	where T: RealNumber,
		  S: ToDspVector<T> + ToSlice<T> {
	type Output = Matrix3xN<GenDspVec<S, T>, T>;

	fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output {
		to_mat_3xn(self, |v|v.to_gen_dsp_vec(is_complex, domain))
	}
}


impl<T, S> ToDspMatrix<GenDspVec<S, T>, T> for [S; 4]
	where T: RealNumber,
		  S: ToDspVector<T> + ToSlice<T> {
	type Output = Matrix4xN<GenDspVec<S, T>, T>;

	fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output {
		to_mat_4xn(self, |v|v.to_gen_dsp_vec(is_complex, domain))
	}
}

impl<T, S> ToRealTimeMatrix<RealTimeVec<S, T>, T> for Vec<S>
	where T: RealNumber,
		  S: ToRealVector<T> + ToSlice<T> {
	type Output = MatrixMxN<RealTimeVec<S, T>, T>;

	fn to_real_time_mat(self) -> Self::Output {
		to_mat_mxn(self, |v|v.to_real_time_vec())
	}
}

impl<T, S> ToRealTimeMatrix<RealTimeVec<S, T>, T> for [S; 2]
	where T: RealNumber,
		  S: ToRealVector<T> + ToSlice<T> {
	type Output = Matrix2xN<RealTimeVec<S, T>, T>;

	fn to_real_time_mat(self) -> Self::Output {
		to_mat_2xn(self, |v|v.to_real_time_vec())
	}
}

impl<T, S> ToRealTimeMatrix<RealTimeVec<S, T>, T> for [S; 3]
	where T: RealNumber,
		  S: ToRealVector<T> + ToSlice<T> {
	type Output = Matrix3xN<RealTimeVec<S, T>, T>;

	fn to_real_time_mat(self) -> Self::Output {
		to_mat_3xn(self, |v|v.to_real_time_vec())
	}
}

impl<T, S> ToRealTimeMatrix<RealTimeVec<S, T>, T> for [S; 4]
	where T: RealNumber,
		  S: ToRealVector<T> + ToSlice<T> {
	type Output = Matrix4xN<RealTimeVec<S, T>, T>;

	fn to_real_time_mat(self) -> Self::Output {
		to_mat_4xn(self, |v|v.to_real_time_vec())
	}
}

impl<T, S> ToRealFreqMatrix<RealFreqVec<S, T>, T> for Vec<S>
	where T: RealNumber,
		  S: ToRealVector<T> + ToSlice<T> {
	type Output = MatrixMxN<RealFreqVec<S, T>, T>;

	fn to_real_freq_mat(self) -> Self::Output {
		to_mat_mxn(self, |v|v.to_real_freq_vec())
	}
}

impl<T, S> ToRealFreqMatrix<RealFreqVec<S, T>, T> for [S; 2]
	where T: RealNumber,
		  S: ToRealVector<T> + ToSlice<T> {
	type Output = Matrix2xN<RealFreqVec<S, T>, T>;

	fn to_real_freq_mat(self) -> Self::Output {
		to_mat_2xn(self, |v|v.to_real_freq_vec())
	}
}

impl<T, S> ToRealFreqMatrix<RealFreqVec<S, T>, T> for [S; 3]
	where T: RealNumber,
		  S: ToRealVector<T> + ToSlice<T> {
	type Output = Matrix3xN<RealFreqVec<S, T>, T>;

	fn to_real_freq_mat(self) -> Self::Output {
		to_mat_3xn(self, |v|v.to_real_freq_vec())
	}
}

impl<T, S> ToRealFreqMatrix<RealFreqVec<S, T>, T> for [S; 4]
	where T: RealNumber,
		  S: ToRealVector<T> + ToSlice<T> {
	type Output = Matrix4xN<RealFreqVec<S, T>, T>;

	fn to_real_freq_mat(self) -> Self::Output {
		to_mat_4xn(self, |v|v.to_real_freq_vec())
	}
}

impl<T, S> ToComplexTimeMatrix<ComplexTimeVec<S, T>, T> for Vec<S>
	where T: RealNumber,
		  S: ToComplexVector<S, T> + ToSlice<T> {
	type Output = MatrixMxN<ComplexTimeVec<S, T>, T>;

	fn to_complex_time_mat(self) -> Self::Output {
		to_mat_mxn(self, |v|v.to_complex_time_vec())
	}
}

impl<T, S> ToComplexTimeMatrix<ComplexTimeVec<S, T>, T> for [S; 2]
	where T: RealNumber,
		  S: ToComplexVector<S, T> + ToSlice<T> {
	type Output = Matrix2xN<ComplexTimeVec<S, T>, T>;

	fn to_complex_time_mat(self) -> Self::Output {
		to_mat_2xn(self, |v|v.to_complex_time_vec())
	}
}

impl<T, S> ToComplexTimeMatrix<ComplexTimeVec<S, T>, T> for [S; 3]
	where T: RealNumber,
		  S: ToComplexVector<S, T> + ToSlice<T> {
	type Output = Matrix3xN<ComplexTimeVec<S, T>, T>;

	fn to_complex_time_mat(self) -> Self::Output {
		to_mat_3xn(self, |v|v.to_complex_time_vec())
	}
}

impl<T, S> ToComplexTimeMatrix<ComplexTimeVec<S, T>, T> for [S; 4]
	where T: RealNumber,
		  S: ToComplexVector<S, T> + ToSlice<T> {
	type Output = Matrix4xN<ComplexTimeVec<S, T>, T>;

	fn to_complex_time_mat(self) -> Self::Output {
		to_mat_4xn(self, |v|v.to_complex_time_vec())
	}
}

impl<V, T> FromMatrix<T> for MatrixMxN<V, T>
	where T: RealNumber,
		  V: Vector<T> {
	type Output = Vec<V>;

	fn get(self) -> (Self::Output, usize) {
        let len = self.row_len();
		(self.rows, len)
	}
}

impl<V, T> FromMatrix<T> for Matrix2xN<V, T>
	where T: RealNumber,
		  V: Vector<T> {
	type Output = [V; 2];

	fn get(self) -> (Self::Output, usize) {
        let len = self.row_len();
		(self.rows, len)
	}
}

impl<V, T> FromMatrix<T> for Matrix3xN<V, T>
	where T: RealNumber,
		  V: Vector<T> {
	type Output = [V; 3];

	fn get(self) -> (Self::Output, usize) {
        let len = self.row_len();
		(self.rows, len)
	}
}

impl<V, T> FromMatrix<T> for Matrix4xN<V, T>
	where T: RealNumber,
		  V: Vector<T> {
	type Output = [V; 4];

	fn get(self) -> (Self::Output, usize) {
        let len = self.row_len();
		(self.rows, len)
	}
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use basic_dsp_vector::*;

    #[test]
    fn to_gen_dsp_mat_test() {
        let mat: MatrixMxN<_, _> = vec!(vec!(0.0, 1.0), vec!(2.0, 3.0)).to_gen_dsp_mat(false, DataDomain::Time);
		assert_eq!(&mat.rows[0][..], &[0.0, 1.0]);
		assert_eq!(&mat.rows[1][..], &[2.0, 3.0]);

		let mat: Matrix2xN<_, _> = [vec!(0.0, 1.0), vec!(2.0, 3.0)].to_gen_dsp_mat(false, DataDomain::Time);
		assert_eq!(&mat.rows[0][..], &[0.0, 1.0]);
		assert_eq!(&mat.rows[1][..], &[2.0, 3.0]);
	}
}
