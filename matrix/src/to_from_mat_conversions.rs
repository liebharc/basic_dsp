use super::*;
use std::marker;
use TransformContent;

/// Conversion from a collection of vectors to a matrix.
pub trait ToMatrix<V, T>
where
    V: Vector<T>,
    T: RealNumber,
{
    type Output: Matrix<V, T>;

    /// Create a new matrix from a collection of vectors.
    fn to_mat(self) -> Self::Output;
}

/// Conversion from a generic data type into a dsp matrix which tracks
/// its meta information (domain and number space)
/// only at runtime. See `ToRealMatrix` and
/// `ToComplexMatrix` for alternatives which track most of the meta data
/// with the type system and therefore avoid runtime errors.
pub trait ToDspMatrix<V, T>
where
    V: Vector<T>,
    T: RealNumber,
{
    type Output: Matrix<V, T>;

    /// Create a new generic matrix.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex matrices with an odd length the resulting value will have a zero length.
    fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output;
}

/// Conversion from a generic data type into a dsp matrix with real data.
pub trait ToRealTimeMatrix<V, T>
where
    V: Vector<T>,
    T: RealNumber,
{
    type Output: Matrix<V, T>;

    /// Create a new matrix in real number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_time_mat(self) -> Self::Output;
}

/// Conversion from a generic data type into a dsp matrix with real data.
pub trait ToRealFreqMatrix<V, T>
where
    V: Vector<T>,
    T: RealNumber,
{
    type Output: Matrix<V, T>;

    /// Create a new vector in real number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    fn to_real_freq_mat(self) -> Self::Output;
}

/// Conversion from a generic data type into a dsp vector with complex data.
pub trait ToComplexTimeMatrix<V, T>
where
    V: Vector<T>,
    T: RealNumber,
{
    type Output: Matrix<V, T>;

    /// Create a new matrix in complex number space and time domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex matrices with an odd length the resulting value will have a zero length.
    fn to_complex_time_mat(self) -> Self::Output;
}

/// Conversion from a generic data type into a dsp vector with complex data.
pub trait ToComplexFreqMatrix<V, T>
where
    V: Vector<T>,
    T: RealNumber,
{
    type Output: Matrix<V, T>;

    /// Create a new matrix in complex number space and frequency domain.
    /// `delta` can be changed after construction with a call of `set_delta`.
    ///
    /// For complex matrices with an odd length the resulting value will have a zero length.
    fn to_complex_freq_mat(self) -> Self::Output;
}

/// Retrieves the underlying storage from a matrix.
pub trait FromMatrix<T>
where
    T: RealNumber,
{
    /// Type of the underlying storage of a matrix.
    type Output;

    /// Gets the underlying matrix and the number of elements which
    /// contain valid.
    fn get(self) -> (Self::Output, usize);
}

fn to_mat_mxn<S, F, V, T>(source: Vec<S>, conversion: F) -> MatrixMxN<V, S, T>
where
    S: ToSlice<T>,
    V: Vector<T>,
    T: RealNumber,
    F: Fn(S) -> V,
{
    let rows = source.transform(conversion);
    MatrixMxN {
        rows: rows,
        storage_type: marker::PhantomData,
        number_type: marker::PhantomData,
    }
}

fn to_mat_2xn<S, F, V, T>(source: [S; 2], conversion: F) -> Matrix2xN<V, S, T>
where
    S: ToSlice<T>,
    V: Vector<T>,
    T: RealNumber,
    F: Fn(S) -> V,
{
    let rows = source.transform(conversion);
    Matrix2xN {
        rows: rows,
        storage_type: marker::PhantomData,
        number_type: marker::PhantomData,
    }
}

fn to_mat_3xn<S, F, V, T>(source: [S; 3], conversion: F) -> Matrix3xN<V, S, T>
where
    S: ToSlice<T>,
    V: Vector<T>,
    T: RealNumber,
    F: Fn(S) -> V,
{
    let rows = source.transform(conversion);
    Matrix3xN {
        rows: rows,
        storage_type: marker::PhantomData,
        number_type: marker::PhantomData,
    }
}

fn to_mat_4xn<S, F, V, T>(source: [S; 4], conversion: F) -> Matrix4xN<V, S, T>
where
    S: ToSlice<T>,
    V: Vector<T>,
    T: RealNumber,
    F: Fn(S) -> V,
{
    let rows = source.transform(conversion);
    Matrix4xN {
        rows: rows,
        storage_type: marker::PhantomData,
        number_type: marker::PhantomData,
    }
}

impl<S, T, N, D> ToMatrix<DspVec<S, T, N, D>, T> for Vec<DspVec<S, T, N, D>>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = MatrixMxN<DspVec<S, T, N, D>, S, T>;

    fn to_mat(self) -> Self::Output {
        MatrixMxN {
            rows: self,
            storage_type: marker::PhantomData,
            number_type: marker::PhantomData,
        }
    }
}

impl<S, T, N, D> ToMatrix<DspVec<S, T, N, D>, T> for [DspVec<S, T, N, D>; 2]
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = Matrix2xN<DspVec<S, T, N, D>, S, T>;

    fn to_mat(self) -> Self::Output {
        Matrix2xN {
            rows: self,
            storage_type: marker::PhantomData,
            number_type: marker::PhantomData,
        }
    }
}

impl<S, T, N, D> ToMatrix<DspVec<S, T, N, D>, T> for [DspVec<S, T, N, D>; 3]
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = Matrix3xN<DspVec<S, T, N, D>, S, T>;

    fn to_mat(self) -> Self::Output {
        Matrix3xN {
            rows: self,
            storage_type: marker::PhantomData,
            number_type: marker::PhantomData,
        }
    }
}

impl<S, T, N, D> ToMatrix<DspVec<S, T, N, D>, T> for [DspVec<S, T, N, D>; 4]
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    type Output = Matrix4xN<DspVec<S, T, N, D>, S, T>;

    fn to_mat(self) -> Self::Output {
        Matrix4xN {
            rows: self,
            storage_type: marker::PhantomData,
            number_type: marker::PhantomData,
        }
    }
}

impl<T, S> ToDspMatrix<GenDspVec<S, T>, T> for Vec<S>
where
    T: RealNumber,
    S: ToDspVector<T> + ToSlice<T>,
{
    type Output = MatrixMxN<GenDspVec<S, T>, S, T>;

    fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output {
        to_mat_mxn(self, |v| v.to_gen_dsp_vec(is_complex, domain))
    }
}

impl<T, S> ToDspMatrix<GenDspVec<S, T>, T> for [S; 2]
where
    T: RealNumber,
    S: ToDspVector<T> + ToSlice<T>,
{
    type Output = Matrix2xN<GenDspVec<S, T>, S, T>;

    fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output {
        to_mat_2xn(self, |v| v.to_gen_dsp_vec(is_complex, domain))
    }
}

impl<T, S> ToDspMatrix<GenDspVec<S, T>, T> for [S; 3]
where
    T: RealNumber,
    S: ToDspVector<T> + ToSlice<T>,
{
    type Output = Matrix3xN<GenDspVec<S, T>, S, T>;

    fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output {
        to_mat_3xn(self, |v| v.to_gen_dsp_vec(is_complex, domain))
    }
}

impl<T, S> ToDspMatrix<GenDspVec<S, T>, T> for [S; 4]
where
    T: RealNumber,
    S: ToDspVector<T> + ToSlice<T>,
{
    type Output = Matrix4xN<GenDspVec<S, T>, S, T>;

    fn to_gen_dsp_mat(self, is_complex: bool, domain: DataDomain) -> Self::Output {
        to_mat_4xn(self, |v| v.to_gen_dsp_vec(is_complex, domain))
    }
}

impl<T, S> ToRealTimeMatrix<RealTimeVec<S, T>, T> for Vec<S>
where
    T: RealNumber,
    S: ToRealVector<T> + ToSlice<T>,
{
    type Output = MatrixMxN<RealTimeVec<S, T>, S, T>;

    fn to_real_time_mat(self) -> Self::Output {
        to_mat_mxn(self, |v| v.to_real_time_vec())
    }
}

impl<T, S> ToRealTimeMatrix<RealTimeVec<S, T>, T> for [S; 2]
where
    T: RealNumber,
    S: ToRealVector<T> + ToSlice<T>,
{
    type Output = Matrix2xN<RealTimeVec<S, T>, S, T>;

    fn to_real_time_mat(self) -> Self::Output {
        to_mat_2xn(self, |v| v.to_real_time_vec())
    }
}

impl<T, S> ToRealTimeMatrix<RealTimeVec<S, T>, T> for [S; 3]
where
    T: RealNumber,
    S: ToRealVector<T> + ToSlice<T>,
{
    type Output = Matrix3xN<RealTimeVec<S, T>, S, T>;

    fn to_real_time_mat(self) -> Self::Output {
        to_mat_3xn(self, |v| v.to_real_time_vec())
    }
}

impl<T, S> ToRealTimeMatrix<RealTimeVec<S, T>, T> for [S; 4]
where
    T: RealNumber,
    S: ToRealVector<T> + ToSlice<T>,
{
    type Output = Matrix4xN<RealTimeVec<S, T>, S, T>;

    fn to_real_time_mat(self) -> Self::Output {
        to_mat_4xn(self, |v| v.to_real_time_vec())
    }
}

impl<T, S> ToRealFreqMatrix<RealFreqVec<S, T>, T> for Vec<S>
where
    T: RealNumber,
    S: ToRealVector<T> + ToSlice<T>,
{
    type Output = MatrixMxN<RealFreqVec<S, T>, S, T>;

    fn to_real_freq_mat(self) -> Self::Output {
        to_mat_mxn(self, |v| v.to_real_freq_vec())
    }
}

impl<T, S> ToRealFreqMatrix<RealFreqVec<S, T>, T> for [S; 2]
where
    T: RealNumber,
    S: ToRealVector<T> + ToSlice<T>,
{
    type Output = Matrix2xN<RealFreqVec<S, T>, S, T>;

    fn to_real_freq_mat(self) -> Self::Output {
        to_mat_2xn(self, |v| v.to_real_freq_vec())
    }
}

impl<T, S> ToRealFreqMatrix<RealFreqVec<S, T>, T> for [S; 3]
where
    T: RealNumber,
    S: ToRealVector<T> + ToSlice<T>,
{
    type Output = Matrix3xN<RealFreqVec<S, T>, S, T>;

    fn to_real_freq_mat(self) -> Self::Output {
        to_mat_3xn(self, |v| v.to_real_freq_vec())
    }
}

impl<T, S> ToRealFreqMatrix<RealFreqVec<S, T>, T> for [S; 4]
where
    T: RealNumber,
    S: ToRealVector<T> + ToSlice<T>,
{
    type Output = Matrix4xN<RealFreqVec<S, T>, S, T>;

    fn to_real_freq_mat(self) -> Self::Output {
        to_mat_4xn(self, |v| v.to_real_freq_vec())
    }
}

impl<T, S> ToComplexTimeMatrix<ComplexTimeVec<S, T>, T> for Vec<S>
where
    T: RealNumber,
    S: ToComplexVector<S, T> + ToSlice<T>,
{
    type Output = MatrixMxN<ComplexTimeVec<S, T>, S, T>;

    fn to_complex_time_mat(self) -> Self::Output {
        to_mat_mxn(self, |v| v.to_complex_time_vec())
    }
}

impl<T, S> ToComplexTimeMatrix<ComplexTimeVec<S, T>, T> for [S; 2]
where
    T: RealNumber,
    S: ToComplexVector<S, T> + ToSlice<T>,
{
    type Output = Matrix2xN<ComplexTimeVec<S, T>, S, T>;

    fn to_complex_time_mat(self) -> Self::Output {
        to_mat_2xn(self, |v| v.to_complex_time_vec())
    }
}

impl<T, S> ToComplexTimeMatrix<ComplexTimeVec<S, T>, T> for [S; 3]
where
    T: RealNumber,
    S: ToComplexVector<S, T> + ToSlice<T>,
{
    type Output = Matrix3xN<ComplexTimeVec<S, T>, S, T>;

    fn to_complex_time_mat(self) -> Self::Output {
        to_mat_3xn(self, |v| v.to_complex_time_vec())
    }
}

impl<T, S> ToComplexTimeMatrix<ComplexTimeVec<S, T>, T> for [S; 4]
where
    T: RealNumber,
    S: ToComplexVector<S, T> + ToSlice<T>,
{
    type Output = Matrix4xN<ComplexTimeVec<S, T>, S, T>;

    fn to_complex_time_mat(self) -> Self::Output {
        to_mat_4xn(self, |v| v.to_complex_time_vec())
    }
}

impl<V, S, T> FromMatrix<T> for MatrixMxN<V, S, T>
where
    T: RealNumber,
    V: Vector<T>,
    S: ToSlice<T>,
{
    type Output = Vec<V>;

    fn get(self) -> (Self::Output, usize) {
        let len = self.row_len();
        (self.rows, len)
    }
}

impl<V, S, T> FromMatrix<T> for Matrix2xN<V, S, T>
where
    T: RealNumber,
    V: Vector<T>,
    S: ToSlice<T>,
{
    type Output = [V; 2];

    fn get(self) -> (Self::Output, usize) {
        let len = self.row_len();
        (self.rows, len)
    }
}

impl<V, S, T> FromMatrix<T> for Matrix3xN<V, S, T>
where
    T: RealNumber,
    V: Vector<T>,
    S: ToSlice<T>,
{
    type Output = [V; 3];

    fn get(self) -> (Self::Output, usize) {
        let len = self.row_len();
        (self.rows, len)
    }
}

impl<V, S, T> FromMatrix<T> for Matrix4xN<V, S, T>
where
    T: RealNumber,
    V: Vector<T>,
    S: ToSlice<T>,
{
    type Output = [V; 4];

    fn get(self) -> (Self::Output, usize) {
        let len = self.row_len();
        (self.rows, len)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn to_gen_dsp_mat_test() {
        let mat: MatrixMxN<_, _, _> =
            vec![vec![0.0, 1.0], vec![2.0, 3.0]].to_gen_dsp_mat(false, DataDomain::Time);
        assert_eq!(&mat.rows[0][..], &[0.0, 1.0]);
        assert_eq!(&mat.rows[1][..], &[2.0, 3.0]);

        let mat: Matrix2xN<_, _, _> =
            [vec![0.0, 1.0], vec![2.0, 3.0]].to_gen_dsp_mat(false, DataDomain::Time);
        assert_eq!(&mat.rows[0][..], &[0.0, 1.0]);
        assert_eq!(&mat.rows[1][..], &[2.0, 3.0]);
    }
}
